from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation

from monotm.cli import main
from monotm.config import GenerationConfig, RunConfig
from monotm.lda_config import LDA_ESTIMATOR, LDA_REFERENCE_URL
from monotm.pipeline import run, run_fingerprint
from monotm.preflight import check_run, format_report
from monotm.sae import SAEFeatures
from monotm.topics import fit_mixtures, make_lda
from test_pipeline import PipelineFixture


class LDAParameterTests(unittest.TestCase):
    def test_notebook_exposes_every_constructor_parameter(self):
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root / "examples/start_here.ipynb").read_text())
        namespace = {"Path": Path, "RunConfig": RunConfig, "GenerationConfig": GenerationConfig}
        for cell in notebook["cells"]:
            source = "".join(cell["source"])
            if cell["cell_type"] == "code" and any(marker in source for marker in (
                    "INPUT_CSV =", "SKLEARN_LDA_PARAMS =", "config = RunConfig(")):
                exec(compile(source, "start_here.ipynb", "exec"), namespace)
        expected = set(LatentDirichletAllocation().get_params(deep=False))
        self.assertEqual(set(namespace["SKLEARN_LDA_PARAMS"]), expected)
        namespace["config"].validate()
        self.assertEqual(set(make_lda(namespace["config"]).get_params(deep=False)), expected)

    def test_every_parameter_reaches_estimator_and_online_fitting(self):
        parameters = dict(n_components=2, doc_topic_prior=0.3, topic_word_prior=0.2,
                          learning_method="online", learning_decay=0.6, learning_offset=16.0,
                          max_iter=3, batch_size=2, evaluate_every=1, total_samples=128.0,
                          perp_tol=0.3, mean_change_tol=0.002, max_doc_update_iter=8,
                          n_jobs=None, verbose=0, random_state=15)
        config = RunConfig(lda_params=parameters)
        self.assertEqual(make_lda(config).get_params(deep=False), parameters)
        self.assertEqual(config.n_topics, 2)
        features = SAEFeatures(np.tile([0, 1], (8, 1)),
                               np.array([[8, 1], [6, 2], [7, 1], [5, 2], [1, 8], [2, 6], [1, 7], [2, 5.]]), 2)
        with patch("monotm.topics.LatentDirichletAllocation", wraps=LatentDirichletAllocation) as estimator:
            theta, components, _, _ = fit_mixtures(features, config)
        self.assertEqual(estimator.call_args.kwargs, parameters)
        self.assertEqual(theta.shape, (8, 2))
        self.assertEqual(components.shape, (2, 2))
        np.testing.assert_allclose(theta.sum(axis=1), 1)

    def test_alias_precedence_and_json_roundtrip(self):
        config = RunConfig.from_dict({"n_topics": 9, "lda_max_iter": 5, "lda_jobs": 1, "seed": 42,
                                     "lda_params": {"n_components": 3, "max_iter": 2, "n_jobs": None, "random_state": 7}})
        self.assertEqual((config.n_topics, config.lda_max_iter, config.lda_jobs, config.seed), (3, 2, None, 42))
        self.assertEqual(config.lda_parameters()["random_state"], 7)
        self.assertEqual(RunConfig.from_dict(config.to_dict()), config)
        config = RunConfig(n_topics=5, seed=19, lda_max_iter=13, lda_jobs=2)
        self.assertEqual(make_lda(config).get_params()["random_state"], 19)
        self.assertEqual(config.lda_parameters()["n_components"], 5)

    def test_invalid_values_are_rejected_before_fitting(self):
        invalid = [None, [], {"n_components": 0}, {"doc_topic_prior": -0.1}, {"topic_word_prior": 1.5},
                   {"learning_method": "gibbs"}, {"learning_decay": 1.1}, {"learning_offset": 0},
                   {"max_iter": -1}, {"batch_size": 0}, {"evaluate_every": 0.5}, {"total_samples": 0},
                   {"perp_tol": -1}, {"mean_change_tol": float("nan")}, {"max_doc_update_iter": -1},
                   {"n_jobs": 0}, {"verbose": -1}, {"random_state": 2**32}]
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                RunConfig.from_dict({"lda_params": values})

    def test_config_command_lists_complete_scikit_learn_dictionary(self):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(["config"]), 0)
        values = json.loads(output.getvalue())
        self.assertEqual(set(values["lda_params"]), set(LatentDirichletAllocation().get_params()))
        self.assertEqual(values["lda_params"]["max_iter"], 200)
        self.assertEqual(values["lda_params"]["n_components"], 20)
        self.assertFalse({"n_topics", "lda_max_iter", "lda_jobs"} & set(values))


class LDAPipelineTests(PipelineFixture, unittest.TestCase):
    def test_full_dictionary_matches_run_identity_and_changed_prior_invalidates_it(self):
        from monotm.data import load_corpus
        corpus = load_corpus(self.input, id_column="id")
        baseline = run_fingerprint(corpus, self.config, self.embeddings)
        explicit = replace(self.config, lda_params=self.config.lda_parameters())
        explicit.validate()
        self.assertEqual(baseline, run_fingerprint(corpus, explicit, self.embeddings))
        changed = replace(self.config, lda_params={"doc_topic_prior": 0.2})
        changed.validate()
        self.assertNotEqual(baseline, run_fingerprint(corpus, changed, self.embeddings))

    def test_check_shows_all_effective_parameters_and_applicability(self):
        config = replace(self.config, lda_params={"n_components": 2, "learning_method": "online", "n_jobs": None,
                                                  "random_state": None, "evaluate_every": 1, "doc_topic_prior": 0.2})
        with patch.dict(os.environ, {"MONOTM_API_KEY": "test-key"}), \
                patch("monotm.llm.request.urlopen", side_effect=AssertionError("no network")):
            report = check_run(self.input, config=config, id_column="id", embeddings_path=self.embeddings)
        self.assertTrue(report["ready"], report["errors"])
        self.assertEqual(report["lda"]["parameters"], make_lda(config).get_params())
        self.assertEqual(report["lda"]["inactive_parameters"], ["total_samples"])
        self.assertIn('"doc_topic_prior": 0.2', format_report(report))
        self.assertIn("Scikit-learn LDA parameters:", format_report(report))
        self.assertEqual(report["lda"]["estimator"], LDA_ESTIMATOR)
        self.assertIn(LDA_ESTIMATOR, format_report(report))
        self.assertIn(LDA_REFERENCE_URL, format_report(report))

    def test_invalid_keyword_blocks_before_training_and_writes(self):
        config = replace(self.config, lda_params={"not_a_parameter": 1})
        output = self.folder / "bad"
        with patch("monotm.pipeline.fit_sae", side_effect=AssertionError("no training")):
            with self.assertRaisesRegex(ValueError, "not_a_parameter"):
                run(self.input, output, config=config, id_column="id", embeddings_path=self.embeddings)
            with patch.dict(os.environ, {"MONOTM_API_KEY": "test-key"}):
                report = check_run(self.input, output, config=config, id_column="id", embeddings_path=self.embeddings)
        self.assertFalse(output.exists())
        self.assertFalse(report["ready"])
        self.assertTrue(any("not_a_parameter" in error for error in report["errors"]))

    def test_cli_dictionary_topic_count_and_explicit_override(self):
        config = replace(self.config, lda_params={"n_components": 3, "max_iter": 5})
        config_file = self.folder / "settings.json"
        config_file.write_text(json.dumps(config.to_dict()))
        args = ["run", "--input", str(self.input), "--output", str(self.folder / "new"), "--id-column", "id",
                "--embeddings", str(self.embeddings), "--config", str(config_file), "--dry-run", "--json"]
        for suffix, expected in (([], 3), (["--topics", "2"], 2)):
            output = io.StringIO()
            with patch.dict(os.environ, {"MONOTM_API_KEY": "test-key"}), redirect_stdout(output):
                self.assertEqual(main([*args, *suffix]), 0)
            report = json.loads(output.getvalue())
            self.assertEqual(report["config"]["n_topics"], expected)
            self.assertEqual(report["lda"]["parameters"]["n_components"], expected)
        self.assertFalse((self.folder / "new").exists())

    def test_topic_count_is_shared_by_lda_and_descriptor_outputs(self):
        config = replace(self.config, lda_params={"n_components": 2, "doc_topic_prior": 0.2, "max_iter": 3})
        result = self.fit(config=config)
        self.assertEqual(result.theta.shape, (64, 2))
        self.assertEqual(result.beta.shape[0], 2)
        manifest = json.loads((result.output_dir / "run.json").read_text())
        self.assertEqual(manifest["lda_parameters"], make_lda(config).get_params())
        before = manifest["fingerprint"]
        with patch("monotm.pipeline.fit_sae", side_effect=AssertionError("no retraining")):
            self.fit(config=config)
        self.assertEqual(json.loads((result.output_dir / "run.json").read_text())["fingerprint"], before)
