import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

import numpy as np
from scipy.sparse import csr_matrix
import torch

from monotm.config import RunConfig, SAEConfig
from monotm.data import Corpus, load_corpus, load_embeddings
from monotm.embeddings import mean_pool
from monotm.interpretation import compute_scores, parse_label, parse_prediction, select_examples, validate_features
from monotm.llm import ChatClient, LLMError
from monotm.sae import SAEFeatures, TopKSAE
from monotm.topics import feature_matrix, fit_descriptors, fit_mixtures


class InputTests(unittest.TestCase):
    def test_csv_keeps_ids_and_quoted_multiline_text(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "input.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["key", "body"])
                for i in range(4):
                    writer.writerow([f"00{i}", f"문서 {i},\nsecond line"])
            corpus = load_corpus(path, "body", "key")
            self.assertEqual(corpus.ids, ["000", "001", "002", "003"])
            self.assertIn("\n", corpus.texts[0])
            self.assertEqual(load_corpus(path, "body").ids, ["0", "1", "2", "3"])

    def test_rejects_empty_text_and_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "input.csv"
            for body, expected in [("id,text\na,x\na,y\nc,z\nd,w\n", "unique"),
                                   ("id,text\na,x\nb,\nc,z\nd,w\n", "Empty text")]:
                path.write_text(body)
                with self.assertRaisesRegex(ValueError, expected):
                    load_corpus(path, id_column="id")

    def test_embedding_alignment_and_invalid_numbers(self):
        with tempfile.TemporaryDirectory() as folder:
            path, ids = Path(folder) / "e.npy", Path(folder) / "ids.json"
            corpus = Corpus(["a", "b", "c", "d"], ["a", "b", "c", "d"])
            array = np.arange(12).reshape(4, 3)
            np.save(path, array)
            ids.write_text(json.dumps(["d", "c", "b", "a"]))
            np.testing.assert_array_equal(load_embeddings(path, corpus, ids), array[::-1])
            np.save(path, np.array(1))
            with self.assertRaisesRegex(ValueError, "matrix|array"):
                load_embeddings(path, corpus, ids)
            array = array.astype(float)
            array[0, 0] = np.nan
            np.save(path, array)
            with self.assertRaisesRegex(ValueError, "finite"):
                load_embeddings(path, corpus)

    def test_config_unknown_keys_and_nested_defaults(self):
        config = RunConfig.from_dict({"interpretation": {"steps": 9}})
        self.assertEqual(config.interpretation.k, 16)
        self.assertEqual(config.interpretation.steps, 9)
        with self.assertRaisesRegex(ValueError, "Unknown"):
            RunConfig.from_dict({"topcis": 3})
        with self.assertRaisesRegex(ValueError, "threshold"):
            RunConfig.from_dict({"is_threshold": 1.1})


class AlgorithmTests(unittest.TestCase):
    def test_pooling_ignores_padding_and_normalizes(self):
        hidden = torch.tensor([[[99., 99.], [3., 0.], [0., 4.]], [[2., 0.], [99., 99.], [99., 99.]]])
        mask = torch.tensor([[0, 1, 1], [1, 0, 0]])
        np.testing.assert_allclose(mean_pool(hidden, mask).numpy(), [[.6, .8], [1., 0.]], atol=1e-6)

    def test_topk_applies_gain_after_selection(self):
        model = TopKSAE(2, 2, 1)
        with torch.no_grad():
            model.decoder.copy_(torch.eye(2))
            model.bias.zero_()
            model.log_gain.copy_(torch.log(torch.tensor([1., 100.])))
        idx, val = model.sparse_codes(torch.tensor([[2., 1.]]))
        self.assertEqual(idx.item(), 0)
        self.assertEqual(val.item(), 2)

    def test_zero_topk_slots_are_not_active(self):
        features = SAEFeatures(np.array([[0, 1], [1, 0], [0, 1], [0, 1]]),
                               np.array([[0., 2.], [0., 0.], [3., 0.], [0., 4.]]), 2)
        matrix = feature_matrix(features)
        np.testing.assert_array_equal(matrix.toarray(), [[0, 2], [0, 0], [3, 0], [0, 4]])
        theta, _, _, has_features = fit_mixtures(features, RunConfig(n_topics=2, lda_max_iter=3))
        self.assertEqual(theta.shape, (4, 2))
        self.assertFalse(has_features[1])
        np.testing.assert_allclose(theta.sum(1), 1)

    def test_fixed_theta_em_probabilities_and_objective(self):
        theta = np.array([[.9, .1], [.8, .2], [.1, .9], [.2, .8]])
        before = theta.copy()
        counts = csr_matrix([[8., 1., 0.], [7., 1., 0.], [1., 9., 0.], [1., 8., 0.]])
        beta, metrics = fit_descriptors(counts, theta, max_iter=60, batch_size=2)
        np.testing.assert_array_equal(theta, before)
        np.testing.assert_allclose(beta.sum(1), 1)
        self.assertTrue((beta > 0).all())
        self.assertGreater(beta[0, 0], beta[0, 1])
        self.assertGreater(beta[1, 1], beta[1, 0])
        self.assertTrue((np.diff([h["map_objective"] for h in metrics["history"]]) >= -1e-8).all())
        with self.assertRaisesRegex(ValueError, "aligned"):
            fit_descriptors(counts, theta[:2])


class InterpretationTests(unittest.TestCase):
    def config(self):
        return RunConfig(min_support=4, max_examples=2, typical_examples=1, zero_examples=2,
                         validation_examples=3, min_validation_examples=2)

    def test_examples_are_disjoint_and_truly_active(self):
        activation = np.array([3, 4, 2, 1, 5, 6, 0, 0, 0, 0, 0, 0.])
        examples = select_examples(activation, self.config(), np.random.default_rng(42))
        label_rows = set(examples["maximum"] + examples["typical"] + examples["zeros"])
        validation_rows = set(examples["validation_positive"] + examples["validation_negative"])
        self.assertFalse(label_rows & validation_rows)
        self.assertTrue((activation[examples["validation_positive"]] > 0).all())
        self.assertTrue((activation[examples["validation_negative"]] == 0).all())
        self.assertEqual(len(examples["validation_positive"]), len(examples["validation_negative"]))

    def test_insufficient_background_is_not_validated(self):
        self.assertIsNone(select_examples(np.ones(12), self.config(), np.random.default_rng(1)))

    def test_feature_limit_does_not_select_ubiquitous_unvalidatable_features(self):
        class Chat:
            def complete(self, prompt, model, *, validator=None, **kwargs):
                return "FINAL: feature description" if model == "i" else "PREDICTION: 1"
        config = self.config()
        config.max_features, config.interpreter_model, config.predictor_model = 1, "i", "p"
        idx = np.tile([0, 1], (12, 1))
        values = np.column_stack([np.ones(12), np.r_[np.ones(6), np.zeros(6)]])
        records = validate_features(SAEFeatures(idx, values, 2), Corpus([str(i) for i in range(12)], ["text"] * 12), config, Chat())
        self.assertEqual([r["feature_id"] for r in records], [1])
        self.assertEqual(records[0]["status"], "validated")

    def test_scores_and_strict_parsing(self):
        self.assertEqual(compute_scores([1, 0], [1, 0])["is_score"], 1)
        self.assertAlmostEqual(compute_scores([1, 1], [1, 0])["is_score"], 2/3)
        self.assertEqual(compute_scores([0, 0], [1, 0])["is_score"], 0)
        self.assertEqual(parse_prediction("PREDICTION: 1"), 1)
        self.assertEqual(parse_prediction("**PREDICTION: 0**"), 0)
        self.assertIsNone(parse_prediction("PREDICTION: 1.5"))
        self.assertIsNone(parse_prediction("PREDICTION: -1"))
        self.assertEqual(parse_label("FINAL: **sports broadcasting complaints**"), "sports broadcasting complaints")
        self.assertEqual(parse_label("`FINAL: sports broadcasting complaints`"), "sports broadcasting complaints")

    def test_provider_failure_stops_further_feature_calls(self):
        class FailedChat:
            calls = 0
            def complete(self, *args, **kwargs):
                self.calls += 1
                raise LLMError("provider unavailable")
        config = self.config()
        config.llm_workers = 1
        chat = FailedChat()
        idx = np.tile([0, 1], (12, 1))
        values = np.column_stack([np.r_[np.ones(6), np.zeros(6)], np.r_[np.zeros(6), np.ones(6)]])
        records = validate_features(SAEFeatures(idx, values, 2), Corpus([str(i) for i in range(12)], ["text"] * 12), config, chat)
        self.assertTrue(any(r["status"] == "llm_failed" for r in records))
        self.assertEqual(chat.calls, 1)


class ClientTests(unittest.TestCase):
    @staticmethod
    def response(content):
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": content}}]}).encode())

    def test_successful_cache_is_model_and_prompt_specific(self):
        with tempfile.TemporaryDirectory() as folder:
            client = ChatClient(base_url="https://example.test/v1", api_key="private-test-key", cache_dir=folder)
            with patch("monotm.llm.request.urlopen", side_effect=lambda *a, **k: self.response("PREDICTION: 1")) as http:
                self.assertEqual(client.complete("a", "m"), "PREDICTION: 1")
                client.complete("a", "m")
                client.complete("b", "m")
                self.assertEqual(http.call_count, 2)
            for file in Path(folder).glob("*.json"):
                self.assertNotIn("private-test-key", file.read_text())

    def test_malformed_response_is_retried_not_accepted(self):
        with tempfile.TemporaryDirectory() as folder:
            client = ChatClient(base_url="https://example.test/v1", api_key="x", cache_dir=folder, retries=2)
            with patch("monotm.llm.request.urlopen", side_effect=[self.response("invalid"), self.response("PREDICTION: 0")]), patch("monotm.llm.time.sleep"):
                self.assertEqual(client.complete("p", "m", validator=lambda s: parse_prediction(s) is not None), "PREDICTION: 0")
            self.assertEqual(len(list(Path(folder).glob("*.json"))), 1)

    def test_auth_failure_is_not_retried_or_cached(self):
        with tempfile.TemporaryDirectory() as folder:
            client = ChatClient(base_url="https://example.test/v1", api_key="x", cache_dir=folder)
            with patch("monotm.llm.request.urlopen", side_effect=HTTPError("x", 401, "", {}, None)) as http:
                with self.assertRaisesRegex(LLMError, "401"):
                    client.complete("p", "m")
                self.assertEqual(http.call_count, 1)
            self.assertFalse(list(Path(folder).glob("*.json")))
