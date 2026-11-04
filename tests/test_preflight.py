from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from monotm.cli import main
from monotm.preflight import check_run, format_report
from monotm.preflight import _package_version
from monotm.pipeline import run
from test_pipeline import PipelineFixture


class PreflightTests(PipelineFixture, unittest.TestCase):
    def check(self, **kwargs):
        options = dict(config=self.config, id_column="id", embeddings_path=self.embeddings)
        options.update(kwargs)
        return check_run(self.input, self.folder / "new-run", **options)

    def test_read_only_check_lists_actual_settings_without_network_or_training(self):
        with patch.dict(os.environ, {"MONOTM_API_KEY": "private-check-token"}, clear=True), \
                patch("monotm.llm.request.urlopen", side_effect=AssertionError("no API call")), \
                patch("monotm.embeddings.embed_documents", side_effect=AssertionError("no embedding model")), \
                patch("monotm.pipeline.fit_sae", side_effect=AssertionError("no training")):
            report = self.check()
        self.assertTrue(report["ready"], report["errors"])
        self.assertFalse((self.folder / "new-run").exists())
        self.assertEqual(report["input"]["documents"], 64)
        self.assertEqual(report["input"]["embeddings_shape"], [64, 8])
        self.assertEqual(report["hardware"]["selected_device"], "cpu")
        self.assertEqual(report["config"]["interpreter_model"], "interpreter")
        self.assertEqual(report["llm"]["key_source"], "MONOTM_API_KEY")
        self.assertNotIn("private-check-token", json.dumps(report) + format_report(report))

    def test_missing_key_is_reported_before_output_creation(self):
        with patch.dict(os.environ, {}, clear=True):
            report = self.check()
        self.assertFalse(report["ready"])
        self.assertTrue(any("API_KEY" in e for e in report["errors"]))
        self.assertFalse((self.folder / "new-run").exists())

    def test_cuda_detection_and_explicit_cpu_override(self):
        props = SimpleNamespace(name="Test GPU", total_memory=8 * 2**30)
        with patch.dict(os.environ, {"MONOTM_API_KEY": "test-token"}), \
                patch("torch.cuda.is_available", return_value=True), \
                patch("torch.cuda.device_count", return_value=1), \
                patch("torch.cuda.get_device_properties", return_value=props), \
                patch("torch.cuda.mem_get_info", return_value=(3 * 2**30, 8 * 2**30)), \
                patch("torch.cuda.current_device", return_value=0):
            automatic = self.check(config=replace(self.config, device="auto"))
            explicit = self.check(config=replace(self.config, device="cpu"))
            bad_index = self.check(config=replace(self.config, device="cuda:5"))
        self.assertTrue(automatic["ready"], automatic["errors"])
        self.assertEqual(automatic["hardware"]["selected_device"], "cuda:0")
        self.assertEqual(automatic["hardware"]["gpus"][0]["free_memory_gib"], 3)
        self.assertEqual(explicit["hardware"]["selected_device"], "cpu")
        self.assertFalse(bad_index["ready"])
        self.assertTrue(any("index 5" in e for e in bad_index["errors"]))

    def test_unavailable_cuda_and_mps_are_blocking_for_new_training(self):
        with patch("torch.cuda.is_available", return_value=False), patch("torch.backends.mps.is_available", return_value=False):
            cuda = self.check(config=replace(self.config, device="cuda"))
            mps = self.check(config=replace(self.config, device="mps"))
        self.assertTrue(any("CUDA was requested" in e for e in cuda["errors"]))
        self.assertTrue(any("MPS was requested" in e for e in mps["errors"]))

    def test_cached_config_dimension_checks_k_without_loading_model(self):
        model_dir = self.folder / "local-model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text('{"hidden_size": 2}')
        with patch.dict(os.environ, {"MONOTM_API_KEY": "test-token"}), \
                patch("monotm.embeddings.embed_documents", side_effect=AssertionError("no model")):
            report = self.check(config=replace(self.config, embedding_model=str(model_dir)), embeddings_path=None)
        self.assertEqual(report["embedding_dimension"], 2)
        self.assertTrue(any("dictionary size 2 is smaller than K=3" in e for e in report["errors"]))

    def test_missing_accelerate_is_caught_before_model_loading(self):
        model_dir = self.folder / "local-model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text('{"hidden_size": 8}')
        versions = lambda name: None if name == "accelerate" else _package_version(name)
        with patch.dict(os.environ, {"MONOTM_API_KEY": "test-token"}), \
                patch("monotm.preflight._package_version", side_effect=versions), \
                patch("monotm.embeddings.embed_documents", side_effect=AssertionError("do not load")):
            report = self.check(config=replace(self.config, embedding_model=str(model_dir)), embeddings_path=None)
        self.assertFalse(report["ready"])
        self.assertTrue(any("accelerate" in e for e in report["errors"]))

    def test_input_and_embedding_errors_are_visible_in_report(self):
        config = replace(self.config, n_topics=100)
        with patch.dict(os.environ, {"MONOTM_API_KEY": "test-token"}):
            report = self.check(config=config)
        self.assertFalse(report["ready"])
        self.assertTrue(any("n_topics" in e for e in report["errors"]))
        ids = self.folder / "bad_ids.json"
        ids.write_text(json.dumps(["wrong"] * 64))
        with patch.dict(os.environ, {"MONOTM_API_KEY": "test-token"}):
            report = self.check(embedding_ids_path=ids)
        self.assertTrue(any("IDs must match" in e for e in report["errors"]))

    def test_invalid_endpoint_does_not_disclose_url_credentials(self):
        config = replace(self.config, base_url="https://name:private-url-password@example.test/v1?key=private-query-key")
        with patch.dict(os.environ, {"MONOTM_API_KEY": "test-token"}):
            report = self.check(config=config)
        encoded = json.dumps(report) + format_report(report)
        self.assertFalse(report["ready"])
        self.assertNotIn("private-url-password", encoded)
        self.assertNotIn("private-query-key", encoded)
        self.assertEqual(report["llm"]["endpoint"], "<invalid endpoint>")

    def test_completed_run_check_reuses_stages_without_api_key(self):
        def reply(req, **kwargs):
            model = json.loads(req.data)["model"]
            text = "FINAL: synthetic feature label" if model == "interpreter" else "PREDICTION: 1"
            return io.BytesIO(json.dumps({"choices": [{"message": {"content": text}}]}).encode())
        output = self.folder / "completed"
        with patch.dict(os.environ, {"MONOTM_API_KEY": "test-token"}), patch("monotm.llm.request.urlopen", side_effect=reply):
            run(self.input, output, config=self.config, id_column="id", embeddings_path=self.embeddings)
        manifest = (output / "run.json").read_bytes()
        with patch.dict(os.environ, {}, clear=True), patch("monotm.llm.request.urlopen", side_effect=AssertionError("no request")):
            report = check_run(self.input, output, config=self.config, id_column="id", embeddings_path=self.embeddings)
            mismatch = check_run(self.input, output, config=replace(self.config, n_topics=2), id_column="id", embeddings_path=self.embeddings)
        self.assertTrue(report["ready"], report["errors"])
        self.assertFalse(report["llm"]["requests_needed"])
        self.assertEqual(set(report["stages"].values()), {"reuse"})
        self.assertEqual((output / "run.json").read_bytes(), manifest)
        self.assertFalse(mismatch["ready"])
        self.assertTrue(any("different corpus, configuration" in e for e in mismatch["errors"]))

    def test_cli_check_and_run_resolve_identical_overrides(self):
        config = self.folder / "settings.json"
        config.write_text(json.dumps(self.config.to_dict()))
        options = ["--input", str(self.input), "--output", str(self.folder / "new"), "--id-column", "id",
                   "--embeddings", str(self.embeddings), "--config", str(config), "--topics", "2",
                   "--interpreter-model", "chosen-model", "--device", "cpu"]
        out = io.StringIO()
        with patch.dict(os.environ, {"MONOTM_API_KEY": "test-token"}), redirect_stdout(out):
            self.assertEqual(main(["check", *options, "--json"]), 0)
        report = json.loads(out.getvalue())
        with patch.dict(os.environ, {"MONOTM_API_KEY": "test-token"}), redirect_stdout(io.StringIO()), \
                patch("monotm.pipeline.run", return_value=SimpleNamespace(topics=[], output_dir=self.folder)) as fit:
            self.assertEqual(main(["run", *options]), 0)
        self.assertEqual(fit.call_args.kwargs["config"].to_dict(), report["config"])
        self.assertEqual(report["config"]["n_topics"], 2)
        self.assertEqual(report["config"]["interpreter_model"], "chosen-model")

    def test_dry_run_cannot_train_or_create_output(self):
        config = self.folder / "settings.json"
        config.write_text(json.dumps(self.config.to_dict()))
        output = self.folder / "dry-run"
        out = io.StringIO()
        args = ["run", "--input", str(self.input), "--id-column", "id", "--topics", "3",
                "--output", str(output), "--embeddings", str(self.embeddings), "--config", str(config), "--dry-run", "--json"]
        with patch.dict(os.environ, {"MONOTM_API_KEY": "test-token"}), redirect_stdout(out), \
                patch("monotm.pipeline.run", side_effect=AssertionError("must not run")):
            self.assertEqual(main(args), 0)
        self.assertTrue(json.loads(out.getvalue())["ready"])
        self.assertFalse(output.exists())

    def test_normal_cli_run_blocks_before_training_when_key_missing(self):
        config = self.folder / "settings.json"
        config.write_text(json.dumps(self.config.to_dict()))
        with patch.dict(os.environ, {}, clear=True), redirect_stdout(io.StringIO()) as out, \
                patch("monotm.pipeline.run", side_effect=AssertionError("must not run")):
            code = main(["run", "--input", str(self.input), "--id-column", "id", "--topics", "3", "--config", str(config),
                         "--embeddings", str(self.embeddings), "--output", str(self.folder / "blocked")])
        self.assertEqual(code, 1)
        self.assertIn("Fix before running", out.getvalue())
        self.assertFalse((self.folder / "blocked").exists())
