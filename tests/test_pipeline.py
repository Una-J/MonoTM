import csv
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from monotm.config import RunConfig, SAEConfig
from monotm.pipeline import run

torch.set_num_threads(1)


class FixtureChat:
    """Test double only; constant positive predictions have balanced F1=2/3."""
    def __init__(self):
        self.calls = 0

    def complete(self, prompt, model, *, validator=None, **kwargs):
        self.calls += 1
        text = "FINAL: synthetic test feature label" if model == "interpreter" else "PREDICTION: 1"
        if validator is not None:
            assert validator(text)
        return text


class PipelineFixture:
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name)
        self.input = self.folder / "input.csv"
        with self.input.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["id", "text"])
            for i in range(64):
                writer.writerow([f"doc-{i:03}", f"Synthetic document number {i} about one of several test themes."])
        self.embeddings = self.folder / "embeddings.npy"
        np.save(self.embeddings, np.random.default_rng(4).normal(size=(64, 8)).astype(np.float32))
        self.config = RunConfig(n_topics=3, seed=42, device="cpu", interpreter_model="interpreter", predictor_model="predictor",
                                mixture=SAEConfig(width=1, k=3, steps=15, batch_size=16, monitor_every=5),
                                interpretation=SAEConfig(width=1, k=2, steps=15, batch_size=16, monitor_every=5),
                                min_support=4, max_examples=2, typical_examples=1, zero_examples=2,
                                validation_examples=3, min_validation_examples=2, is_threshold=.6,
                                llm_workers=1, lda_max_iter=4, em_max_iter=20)
        self.chat = FixtureChat()

    def tearDown(self):
        self.temporary.cleanup()

    def fit(self, output="run", config=None):
        return run(self.input, self.folder / output, config=config or self.config,
                   id_column="id", embeddings_path=self.embeddings, chat_client=self.chat)


class PipelineTests(PipelineFixture, unittest.TestCase):
    def test_full_pipeline_and_resume(self):
        result = self.fit()
        self.assertEqual(result.theta.shape, (64, 3))
        self.assertEqual(result.beta.shape[0], 3)
        np.testing.assert_allclose(result.theta.sum(1), 1)
        np.testing.assert_allclose(result.beta.sum(1), 1)
        manifest = json.loads((result.output_dir / "run.json").read_text())
        self.assertEqual(manifest["status"], "completed")
        ids = json.loads((result.output_dir / "document_ids.json").read_text())
        self.assertEqual(ids, [f"doc-{i:03}" for i in range(64)])
        records = json.loads((result.output_dir / "cache/validated_features.json").read_text())
        for record in records:
            if record["status"] == "validated":
                ex = record["examples"]
                self.assertFalse(set(ex["maximum"] + ex["typical"] + ex["zeros"]) &
                                 set(ex["validation_positive"] + ex["validation_negative"]))
        calls = self.chat.calls
        with patch("monotm.pipeline.fit_sae", side_effect=AssertionError("should reuse training")):
            resumed = self.fit()
        self.assertEqual(calls, self.chat.calls)
        np.testing.assert_array_equal(result.theta, resumed.theta)
        self.assertFalse((result.output_dir / "run.lock").exists())

    def test_validation_cutoff_does_not_change_mixtures(self):
        low = self.fit()
        high_config = replace(self.config, is_threshold=.9)
        with self.assertRaisesRegex(ValueError, "No features passed"):
            self.fit("strict", high_config)
        np.testing.assert_array_equal(low.theta, np.load(self.folder / "strict/theta.npy"))
        self.assertFalse((self.folder / "strict/run.lock").exists())

    def test_changed_input_is_rejected_without_overwriting_manifest(self):
        result = self.fit()
        before = (result.output_dir / "run.json").read_bytes()
        self.input.write_text(self.input.read_text().replace("document number 0", "CHANGED document number 0"))
        with self.assertRaisesRegex(ValueError, "different corpus"):
            self.fit()
        self.assertEqual(before, (result.output_dir / "run.json").read_bytes())

    def test_missing_key_fails_before_training(self):
        with patch.dict(os.environ, {}, clear=True), patch("monotm.pipeline.fit_sae", side_effect=AssertionError("must fail first")):
            with self.assertRaisesRegex(ValueError, "API_KEY"):
                run(self.input, self.folder / "no-key", config=self.config, embeddings_path=self.embeddings)

    def test_existing_non_run_directory_is_preserved(self):
        target = self.folder / "unrelated"
        target.mkdir()
        (target / "important.txt").write_text("keep")
        with self.assertRaisesRegex(ValueError, "empty"):
            run(self.input, target, config=self.config, embeddings_path=self.embeddings, chat_client=self.chat)
        self.assertEqual((target / "important.txt").read_text(), "keep")
        self.assertFalse((target / "run.lock").exists())
