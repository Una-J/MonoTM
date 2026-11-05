from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from monotm.cli import main
from monotm.config import GenerationConfig, RunConfig
from monotm.generation import compatibility_issues
from monotm.llm import ChatClient, LLMError
from monotm.pipeline import _file_hash, run
from monotm.preflight import check_run, format_report
from monotm.recovery import PROTECTED, inspect_run, retry_features
from test_pipeline import PipelineFixture


def response(text, finish="stop", usage=None):
    return io.BytesIO(json.dumps({"choices": [{"message": {"content": text}, "finish_reason": finish}],
                                 "usage": usage}).encode())


class GenerationTests(unittest.TestCase):
    def test_partial_config_roundtrip_and_validation(self):
        config = RunConfig.from_dict({"interpreter_generation": {"temperature": None, "token_limit": 4096}})
        self.assertIsNone(config.interpreter_generation.temperature)
        self.assertEqual(config.predictor_generation.temperature, 0)
        self.assertEqual(RunConfig.from_dict(config.to_dict()), config)
        for values in ({"temperature": True}, {"temperature": float("nan")}, {"temperature": -1},
                       {"temperature": "none"}, {"token_limit": 0}, {"token_limit": True},
                       {"token_limit_parameter": "max_output_tokens"}, {"reasoning_effort": "invalid"},
                       {"unknown": 3}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                GenerationConfig.from_dict(values)
        with self.assertRaises(ValueError):
            RunConfig.from_dict({"interpreter_generation": None})

    def test_known_provider_rules_do_not_apply_to_proxy_aliases(self):
        options = GenerationConfig()
        issues = compatibility_issues("https://api.openai.com/v1", "o3-2025-04-16", options, "interpreter")
        self.assertTrue(any("max_completion_tokens" in issue for issue in issues))
        self.assertTrue(any("temperature" in issue for issue in issues))
        self.assertFalse(compatibility_issues("https://proxy.test/v1", "o3", options, "interpreter"))
        self.assertFalse(compatibility_issues("https://api.openai.com/v1", "custom-alias", options, "interpreter"))
        reasoning = GenerationConfig(temperature=None, token_limit_parameter="max_completion_tokens", reasoning_effort="low")
        self.assertFalse(compatibility_issues("https://api.openai.com/v1", "o3", reasoning, "interpreter"))
        sampling = replace(reasoning, temperature=0, reasoning_effort="none")
        self.assertFalse(compatibility_issues("https://api.openai.com/v1", "gpt-5.2", sampling, "interpreter"))
        self.assertTrue(compatibility_issues("https://api.openai.com/v1", "gpt-5.2", replace(sampling, reasoning_effort="high"), "interpreter"))
        self.assertTrue(compatibility_issues("https://api.openai.com/v1", "gpt-5-mini", sampling, "interpreter"))
        self.assertTrue(compatibility_issues("https://api.anthropic.com/v1", "claude-opus-4-7-20260416", options, "predictor"))

    def test_requests_use_stage_options_and_cache_only_identical_payloads(self):
        with tempfile.TemporaryDirectory() as folder:
            client = ChatClient(base_url="https://provider.test/v1", api_key="test-key", cache_dir=folder,
                                stage_options={"interpreter": GenerationConfig(temperature=None, token_limit=4096,
                                               token_limit_parameter="max_completion_tokens", reasoning_effort="low")})
            requests = []
            def reply(req, **kwargs):
                requests.append(json.loads(req.data))
                return response("PREDICTION: 1")
            with patch("monotm.llm.request.urlopen", side_effect=reply):
                client.complete("same", "shared-model", context={"stage": "interpreter"})
                client.complete("same", "shared-model", context={"stage": "predictor"})
                client.complete("same", "shared-model", context={"stage": "interpreter"})
                client.stage_options["interpreter"] = replace(client.stage_options["interpreter"], reasoning_effort="high")
                client.complete("same", "shared-model", context={"stage": "interpreter"})
            self.assertEqual(len(requests), 3)
            self.assertNotIn("temperature", requests[0])
            self.assertNotIn("max_tokens", requests[0])
            self.assertEqual(requests[0]["max_completion_tokens"], 4096)
            self.assertEqual(requests[0]["reasoning_effort"], "low")
            self.assertEqual(requests[1]["temperature"], 0)
            self.assertEqual(requests[1]["max_tokens"], 1024)
            self.assertNotIn("reasoning_effort", requests[1])
            self.assertEqual(requests[2]["reasoning_effort"], "high")
            self.assertEqual(client.cache_hits, 1)

    def test_rejected_parameter_is_explained_without_fallback_or_retry(self):
        for param in ("temperature", "max_tokens", "reasoning_effort"):
            with self.subTest(param=param), tempfile.TemporaryDirectory() as folder:
                client = ChatClient(base_url="https://proxy.test/v1", api_key="test-key", cache_dir=folder)
                body = json.dumps({"error": {"param": param, "message": "Unsupported parameter: " + param}}).encode()
                failure = HTTPError("url", 400, "invalid", {}, io.BytesIO(body))
                with patch("monotm.llm.request.urlopen", side_effect=failure) as http:
                    with self.assertRaisesRegex(LLMError, "predictor_generation") as caught:
                        client.complete("prompt", "alias", context={"stage": "predictor"})
                self.assertEqual(caught.exception.kind, "unsupported_parameter")
                self.assertEqual(http.call_count, 1)
                attempt = json.loads(next((Path(folder) / "attempts").glob("*.json")).read_text())
                self.assertEqual(attempt["request"]["temperature"], 0)
                self.assertEqual(attempt["http_status"], 400)
                self.assertFalse(list(Path(folder).glob("*.json")))

    def test_known_bad_options_fail_before_network(self):
        with tempfile.TemporaryDirectory() as folder:
            client = ChatClient(base_url="https://api.openai.com/v1", api_key="test-key", cache_dir=folder)
            with patch("monotm.llm.request.urlopen", side_effect=AssertionError("no request")):
                with self.assertRaisesRegex(LLMError, "temperature"):
                    client.complete("prompt", "o3", context={"stage": "interpreter"})

    def test_truncated_replies_are_recorded_once_and_never_cached(self):
        for text in (None, "", "FINAL: apparently complete but truncated label"):
            with self.subTest(text=text), tempfile.TemporaryDirectory() as folder:
                client = ChatClient(base_url="https://provider.test/v1", api_key="test-key", cache_dir=folder,
                                    stage_options={"interpreter": GenerationConfig(temperature=None,
                                                   token_limit_parameter="max_completion_tokens")})
                usage = {"completion_tokens": 1024, "completion_tokens_details": {"reasoning_tokens": 1024}}
                with patch("monotm.llm.request.urlopen", side_effect=lambda *a, **kw: response(text, "length", usage)) as http:
                    with self.assertRaisesRegex(LLMError, "truncated") as caught:
                        client.complete("prompt", "model", context={"stage": "interpreter"})
                self.assertEqual(caught.exception.kind, "token_limit")
                self.assertEqual(http.call_count, 1)
                attempt = json.loads(next((Path(folder) / "attempts").glob("*.json")).read_text())
                self.assertEqual(attempt["status"], "truncated")
                self.assertEqual(attempt["usage"], usage)
                self.assertFalse(list(Path(folder).glob("*.json")))

    def test_previously_cached_truncated_reply_is_not_reused(self):
        with tempfile.TemporaryDirectory() as folder:
            client = ChatClient(base_url="https://provider.test/v1", api_key="test-key", cache_dir=folder)
            with patch("monotm.llm.request.urlopen", side_effect=lambda *a, **kw: response("PREDICTION: 1")):
                client.complete("prompt", "model")
            cache = next(Path(folder).glob("*.json"))
            saved = json.loads(cache.read_text())
            saved["finish_reason"] = "length"
            cache.write_text(json.dumps(saved))
            with patch("monotm.llm.request.urlopen", side_effect=lambda *a, **kw: response("PREDICTION: 0")) as http:
                self.assertEqual(client.complete("prompt", "model"), "PREDICTION: 0")
                self.assertEqual(http.call_count, 1)


class GenerationPipelineTests(PipelineFixture, unittest.TestCase):
    def test_truncated_run_can_be_repaired_with_larger_stage_budget_without_retraining(self):
        config = replace(self.config, max_features=1, interpreter_generation=GenerationConfig(
            temperature=None, token_limit_parameter="max_completion_tokens", reasoning_effort="low"))
        root = self.folder / "truncated"
        with patch.dict(os.environ, {"MONOTM_API_KEY": "test-key"}), \
                patch("monotm.llm.request.urlopen", side_effect=lambda *a, **kw: response(None, "length")) as http:
            with self.assertRaisesRegex(LLMError, "truncated"):
                run(self.input, root, config=config, id_column="id", embeddings_path=self.embeddings)
        self.assertEqual(http.call_count, 1)
        feature = inspect_run(root)["features"][0]
        self.assertEqual(feature["error_kind"], "token_limit")
        self.assertEqual(feature["interpreter"]["request_parameters"],
                         {"max_completion_tokens": 1024, "reasoning_effort": "low"})
        hashes = {name: _file_hash(root / name) for name in PROTECTED if (root / name).exists()}
        requests = []
        def reply(req, **kwargs):
            data = json.loads(req.data)
            requests.append(data)
            return response("FINAL: complete semantic feature label" if data["model"] == "interpreter" else "PREDICTION: 1")
        with patch.dict(os.environ, {"MONOTM_API_KEY": "test-key"}), patch("monotm.llm.request.urlopen", side_effect=reply), \
                patch("monotm.pipeline.fit_sae", side_effect=AssertionError("no SAE")), \
                patch("monotm.pipeline.fit_mixtures", side_effect=AssertionError("no LDA")):
            repaired = retry_features(root, self.input, failed=True, interpreter_generation={"token_limit": 4096})
        self.assertEqual(repaired["status"], "completed")
        self.assertEqual(requests[0]["max_completion_tokens"], 4096)
        self.assertNotIn("temperature", requests[0])
        self.assertTrue(all(r["max_tokens"] == 1024 for r in requests[1:]))
        self.assertEqual(hashes, {name: _file_hash(root / name) for name in hashes})

    def test_check_and_cli_show_exact_fields_and_explicit_omission(self):
        config_file = self.folder / "settings.json"
        config_file.write_text(json.dumps(self.config.to_dict()))
        args = ["check", "--config", str(config_file), "--input", str(self.input), "--id-column", "id",
                "--embeddings", str(self.embeddings), "--interpreter-temperature", "none",
                "--interpreter-token-limit-parameter", "max_completion_tokens", "--interpreter-token-limit", "4096",
                "--interpreter-reasoning-effort", "low", "--predictor-reasoning-effort", "none", "--json"]
        output = io.StringIO()
        with patch.dict(os.environ, {"MONOTM_API_KEY": "test-key"}), redirect_stdout(output), \
                patch("monotm.llm.request.urlopen", side_effect=AssertionError("no network")):
            self.assertEqual(main(args), 0)
        report = json.loads(output.getvalue())
        self.assertEqual(report["llm"]["request_parameters"]["interpreter"],
                         {"max_completion_tokens": 4096, "reasoning_effort": "low"})
        self.assertEqual(report["llm"]["request_parameters"]["predictor"]["reasoning_effort"], "none")
        self.assertIn("temperature: omitted", format_report(report))
        self.assertIsNone(report["config"]["interpreter_generation"]["temperature"])

    def test_invalid_known_combination_blocks_before_training(self):
        config = replace(self.config, base_url="https://api.openai.com/v1", interpreter_model="o3")
        with patch.dict(os.environ, {"MONOTM_API_KEY": "test-key"}), \
                patch("monotm.pipeline.fit_sae", side_effect=AssertionError("no training")):
            report = check_run(self.input, self.folder / "new", config=config, id_column="id", embeddings_path=self.embeddings)
            self.assertFalse(report["ready"])
            self.assertTrue(any("interpreter_generation.temperature" in e for e in report["errors"]))
            with self.assertRaisesRegex(ValueError, "max_completion_tokens"):
                run(self.input, self.folder / "new", config=config, id_column="id", embeddings_path=self.embeddings)
        self.assertFalse((self.folder / "new/models/mixture_sae.pt").exists())

    def test_retry_changes_generation_without_training_and_records_effective_options(self):
        root = self.fit().output_dir
        feature = inspect_run(root)["features"][0]["feature_id"]
        hashes = {name: _file_hash(root / name) for name in PROTECTED if (root / name).exists()}
        requests = []
        def reply(req, **kwargs):
            data = json.loads(req.data)
            requests.append(data)
            return response("FINAL: revised semantic feature label" if data["model"] == "interpreter" else "PREDICTION: 1")
        with patch.dict(os.environ, {"MONOTM_API_KEY": "test-key"}), patch("monotm.llm.request.urlopen", side_effect=reply), \
                patch("monotm.pipeline.fit_sae", side_effect=AssertionError("no SAE")), \
                patch("monotm.pipeline.fit_mixtures", side_effect=AssertionError("no LDA")):
            result = retry_features(root, self.input, feature_ids=[feature], interpreter_generation={
                "temperature": None, "token_limit_parameter": "max_completion_tokens", "token_limit": 4096, "reasoning_effort": "low"})
        self.assertNotIn("temperature", requests[0])
        self.assertEqual(requests[0]["max_completion_tokens"], 4096)
        self.assertTrue(all(r["max_tokens"] == 1024 and r["temperature"] == 0 for r in requests[1:]))
        record = next(r for r in result["features"] if r["feature_id"] == feature)
        self.assertEqual(record["interpreter"]["request_parameters"],
                         {"max_completion_tokens": 4096, "reasoning_effort": "low"})
        self.assertIsNone(result["recovery_history"][-1]["generation"]["interpreter"]["temperature"])
        self.assertEqual(hashes, {name: _file_hash(root / name) for name in hashes})

    def test_invalid_retry_options_preserve_successful_run(self):
        root = self.fit().output_dir
        feature = inspect_run(root)["features"][0]["feature_id"]
        before = (root / "run.json").read_bytes()
        topics = (root / "topics.json").read_bytes()
        with patch("monotm.llm.request.urlopen", side_effect=AssertionError("no network")):
            with self.assertRaisesRegex(ValueError, "temperature"):
                retry_features(root, self.input, feature_ids=[feature], base_url="https://api.openai.com/v1",
                               interpreter_model="o3", interpreter_generation={"token_limit_parameter": "max_completion_tokens"})
        self.assertEqual((root / "run.json").read_bytes(), before)
        self.assertEqual((root / "topics.json").read_bytes(), topics)

    def test_cli_retry_distinguishes_omitted_and_none_reasoning(self):
        with patch("monotm.recovery.retry_features", return_value={"output_dir": "run"}) as retry, redirect_stdout(io.StringIO()):
            self.assertEqual(main(["retry", "--run", "run", "--input", "input.csv", "--features", "3",
                                   "--interpreter-temperature", "none", "--interpreter-reasoning-effort", "omit",
                                   "--predictor-reasoning-effort", "none", "--predictor-token-limit", "4096"]), 0)
        self.assertEqual(retry.call_args.kwargs["interpreter_generation"], {"temperature": None, "reasoning_effort": None})
        self.assertEqual(retry.call_args.kwargs["predictor_generation"], {"reasoning_effort": "none", "token_limit": 4096})
