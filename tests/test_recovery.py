from contextlib import redirect_stdout, redirect_stderr
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from monotm.cli import main
from monotm.interpretation import parse_prediction
from monotm.llm import ChatClient, LLMError
from monotm.recovery import PROTECTED, inspect_run, retry_features
from monotm.pipeline import _file_hash, run
from test_pipeline import PipelineFixture


class RepairChat:
    base_url = "https://new-provider.test/v1"
    max_tokens = 2048

    def __init__(self, fail=False, prediction=1):
        self.calls = []
        self.fail, self.prediction = fail, prediction

    def complete(self, prompt, model, *, validator=None, context=None, refresh=False):
        self.calls.append({"prompt": prompt, "model": model, "context": context, "refresh": refresh})
        if self.fail:
            raise LLMError("temporary test failure")
        return "FINAL: repaired semantic feature description" if context["stage"] == "interpreter" else f"PREDICTION: {self.prediction}"


class RecoveryTests(PipelineFixture, unittest.TestCase):
    def test_selected_model_change_revalidates_without_training_or_changing_other_features(self):
        result = self.fit()
        root = result.output_dir
        before = inspect_run(root)["features"]
        feature = before[0]["feature_id"]
        hashes = {p: _file_hash(root / p) for p in PROTECTED if (root / p).exists()}
        chat = RepairChat()
        with patch("monotm.pipeline.fit_sae", side_effect=AssertionError("no SAE training")), \
                patch("monotm.pipeline.fit_mixtures", side_effect=AssertionError("no LDA training")):
            repaired = retry_features(root, self.input, feature_ids=[feature], interpreter_model="new-interpreter",
                                      predictor_model="new-predictor", chat_client=chat)
            # A normal resume must use the repaired vocabulary, too.
            self.fit()
        self.assertEqual(hashes, {p: _file_hash(root / p) for p in hashes})
        after = repaired["features"]
        self.assertEqual([r for r in before if r["feature_id"] != feature],
                         [r for r in after if r["feature_id"] != feature])
        changed = next(r for r in after if r["feature_id"] == feature)
        self.assertEqual(changed["label"], "repaired semantic feature description")
        self.assertEqual(changed["interpreter"]["model"], "new-interpreter")
        self.assertEqual(changed["predictor"]["model"], "new-predictor")
        self.assertEqual(changed["examples"], before[0]["examples"])
        self.assertEqual(len(chat.calls), 1 + 2 * changed["validation_per_class"])
        self.assertTrue(all(c["refresh"] and c["context"]["feature_id"] == feature for c in chat.calls))
        self.assertEqual(chat.calls[0]["model"], "new-interpreter")
        self.assertTrue(all(c["model"] == "new-predictor" for c in chat.calls[1:]))
        self.assertIn(changed["label"], (root / "vocabulary.csv").read_text())
        operation = repaired["recovery_history"][-1]
        self.assertTrue(operation["trained_artifacts_unchanged"])
        archive = root / "cache/repairs" / operation["operation_id"]
        self.assertTrue((archive / "before/topics.json").exists())

    def test_predictor_only_keeps_label_and_recomputes_score_and_vocabulary(self):
        root = self.fit().output_dir
        before = inspect_run(root)["features"]
        feature = before[0]["feature_id"]
        chat = RepairChat(prediction=0)
        result = retry_features(root, self.input, feature_ids=[feature], stage="predictor",
                                predictor_model="changed-predictor", chat_client=chat)
        record = next(r for r in result["features"] if r["feature_id"] == feature)
        self.assertEqual(record["label"], before[0]["label"])
        self.assertEqual(record["interpreter"], before[0]["interpreter"])
        self.assertEqual(record["is_score"], 0)
        self.assertTrue(all(c["context"]["stage"] == "predictor" for c in chat.calls))
        import csv
        with (root / "vocabulary.csv").open() as handle:
            self.assertNotIn(str(feature), [r["feature_id"] for r in csv.DictReader(handle)])

    def test_manual_label_is_always_revalidated(self):
        root = self.fit().output_dir
        feature = inspect_run(root)["features"][0]["feature_id"]
        chat = RepairChat()
        result = retry_features(root, self.input, feature_ids=[feature], label="manually revised feature", chat_client=chat)
        record = next(r for r in result["features"] if r["feature_id"] == feature)
        self.assertEqual(record["interpreter"]["source"], "manual")
        self.assertTrue(all(c["context"]["stage"] == "predictor" for c in chat.calls))
        self.assertTrue(all("manually revised feature" in c["prompt"] for c in chat.calls))
        self.assertEqual(record["status"], "validated")

    def test_failed_manual_label_validation_resumes_without_replacing_manual_text(self):
        root = self.fit().output_dir
        feature = inspect_run(root)["features"][0]["feature_id"]
        with self.assertRaisesRegex(LLMError, "incomplete"):
            retry_features(root, self.input, feature_ids=[feature], label="keep this manual description",
                           chat_client=RepairChat(fail=True))
        chat = RepairChat()
        retry_features(root, self.input, failed=True, chat_client=chat)
        record = inspect_run(root, feature_id=feature)["features"][0]
        self.assertEqual(record["label"], "keep this manual description")
        self.assertTrue(all(c["context"]["stage"] == "predictor" for c in chat.calls))

    def test_descriptor_only_resume_needs_no_key_or_llm_or_training(self):
        root = self.fit().output_dir
        feature = inspect_run(root)["features"][0]["feature_id"]
        with patch("monotm.recovery.rebuild_descriptors", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                retry_features(root, self.input, feature_ids=[feature], chat_client=RepairChat())
        self.assertFalse((root / "topics.json").exists())
        with patch.dict(os.environ, {}, clear=True), \
                patch("monotm.recovery.ChatClient", side_effect=AssertionError("no API client")), \
                patch("monotm.pipeline.fit_sae", side_effect=AssertionError("no training")):
            result = retry_features(root, self.input, failed=True)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["recovery_history"][-1]["feature_ids"], [])

    def test_partial_failure_can_resume_with_new_model_without_sae_or_lda(self):
        # Stop the original run during its first LLM request.
        original = self.chat.complete
        self.chat.complete = lambda *a, **k: (_ for _ in ()).throw(LLMError("test outage"))
        with self.assertRaisesRegex(LLMError, "incomplete"):
            self.fit()
        self.chat.complete = original
        root = self.folder / "run"
        initial = inspect_run(root)
        self.assertTrue(any(r["status"] == "llm_failed" for r in initial["features"]))
        self.assertGreater(len(initial["features"]), 1)
        hashes = {p: _file_hash(root / p) for p in PROTECTED if (root / p).exists()}
        chat = RepairChat()
        with patch("monotm.pipeline.fit_sae", side_effect=AssertionError("no SAE")), \
                patch("monotm.pipeline.fit_mixtures", side_effect=AssertionError("no LDA")):
            result = retry_features(root, self.input, failed=True, interpreter_model="replacement", chat_client=chat)
        self.assertEqual(result["status"], "completed")
        self.assertTrue(all(r["status"] == "validated" for r in result["features"]))
        self.assertTrue(all(not c["refresh"] for c in chat.calls))
        self.assertEqual(hashes, {p: _file_hash(root / p) for p in hashes})
        self.assertFalse((root / "cache/validation.partial.json").exists())

    def test_http_partial_run_reuses_successes_and_inspection_matches_feature_and_document(self):
        self.config.max_features = 1
        self.config.llm_retries = 1
        root = self.folder / "run"
        request_count = 0

        def http_reply(req, **kwargs):
            nonlocal request_count
            request_count += 1
            if request_count == 3:
                raise HTTPError(req.full_url, 401, "test failure", {}, io.BytesIO(b"test auth error body"))
            payload = json.loads(req.data)
            text = "FINAL: HTTP fixture semantic label" if payload["model"] == "interpreter" else "PREDICTION: 1"
            return AttemptHistoryTests.response(text)

        with patch.dict(os.environ, {"MONOTM_API_KEY": "test-private-credential"}), \
                patch("monotm.llm.request.urlopen", side_effect=http_reply):
            with self.assertRaisesRegex(LLMError, "incomplete"):
                run(self.input, root, config=self.config, id_column="id", embeddings_path=self.embeddings)
            feature = inspect_run(root)["features"][0]["feature_id"]
            details = inspect_run(root, feature_id=feature)
            failed = next(a for a in details["attempts"] if a["status"] == "failed")
            self.assertEqual(failed["context"]["feature_id"], feature)
            self.assertEqual(failed["context"]["stage"], "predictor")
            self.assertIn(failed["context"]["document_id"], details["features"][0]["examples"]["validation_positive"])
            self.assertEqual(failed["raw_response"], "test auth error body")
            with patch("monotm.pipeline.fit_sae", side_effect=AssertionError("no training")):
                retry_features(root, self.input, failed=True)
        # Interpreter and first Predictor request were already accepted: only
        # the remaining five balanced validation requests reach the endpoint.
        self.assertEqual(request_count, 3 + 5)
        details = inspect_run(root, feature_id=feature)
        self.assertEqual(sum(a["status"] == "cache_hit" for a in details["attempts"]), 2)
        self.assertEqual(details["features"][0]["status"], "validated")

    def test_single_repair_does_not_publish_topics_with_other_pending_features(self):
        self.config.max_features = 2
        self.chat = RepairChat(fail=True)
        with self.assertRaisesRegex(LLMError, "incomplete"):
            self.fit()
        root = self.folder / "run"
        feature = inspect_run(root)["features"][0]["feature_id"]
        with self.assertRaisesRegex(LLMError, "incomplete"):
            retry_features(root, self.input, feature_ids=[feature], chat_client=RepairChat())
        saved = inspect_run(root)
        self.assertEqual(next(r for r in saved["features"] if r["feature_id"] == feature)["status"], "validated")
        self.assertFalse((root / "topics.json").exists())
        chat = RepairChat()
        retry_features(root, self.input, failed=True, chat_client=chat)
        self.assertTrue(all(c["context"]["feature_id"] != feature for c in chat.calls))

    def test_failed_repair_archives_previous_reports_and_removes_stale_descriptors(self):
        root = self.fit().output_dir
        feature = inspect_run(root)["features"][0]["feature_id"]
        old_topics = (root / "topics.json").read_bytes()
        with self.assertRaisesRegex(LLMError, "incomplete"):
            retry_features(root, self.input, feature_ids=[feature], chat_client=RepairChat(fail=True))
        self.assertFalse((root / "topics.json").exists())
        self.assertFalse((root / "beta.npy").exists())
        self.assertFalse((root / "run.lock").exists())
        result = inspect_run(root)
        operation = result["recovery_history"][-1]
        archive = root / "cache/repairs" / operation["operation_id"]
        self.assertEqual((archive / "before/topics.json").read_bytes(), old_topics)
        self.assertEqual(operation["status"], "failed")
        retry_features(root, self.input, failed=True, chat_client=RepairChat())
        self.assertTrue((root / "topics.json").exists())

    def test_input_mismatch_and_invalid_selection_do_not_modify_run(self):
        root = self.fit().output_dir
        before = (root / "run.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "candidate"):
            retry_features(root, self.input, feature_ids=[9999], chat_client=RepairChat())
        self.input.write_text(self.input.read_text().replace("document number 0", "changed document number 0"))
        with self.assertRaisesRegex(ValueError, "identical"):
            retry_features(root, self.input, feature_ids=[0], chat_client=RepairChat())
        self.assertEqual(before, (root / "run.json").read_bytes())
        self.assertTrue((root / "topics.json").exists())

    def test_inspect_cli_exports_readable_diagnostics_without_llm_calls(self):
        root = self.fit().output_dir
        record = inspect_run(root)["features"][0]
        feature = record["feature_id"]
        target = self.folder / "diagnostics.json"
        with redirect_stdout(io.StringIO()), patch("monotm.llm.request.urlopen", side_effect=AssertionError("read-only")):
            self.assertEqual(main(["inspect", "--run", str(root), "--feature", str(feature), "--output", str(target)]), 0)
        data = json.loads(target.read_text())
        self.assertEqual(data["features"][0], record)
        self.assertIn("attempts", data)
        with redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
            self.assertEqual(main(["inspect", "--run", str(root), "--output", str(target)]), 1)

    def test_retry_cli_passes_model_and_token_overrides(self):
        root = self.fit().output_dir
        feature = inspect_run(root)["features"][0]["feature_id"]
        with patch("monotm.recovery.ChatClient", return_value=RepairChat()) as client, redirect_stdout(io.StringIO()):
            code = main(["retry", "--run", str(root), "--input", str(self.input), "--features", str(feature),
                         "--interpreter-model", "i2", "--predictor-model", "p2", "--max-tokens", "2048"])
        self.assertEqual(code, 0)
        self.assertEqual(client.call_args.kwargs["max_tokens"], 2048)


class AttemptHistoryTests(unittest.TestCase):
    @staticmethod
    def response(content):
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": content}, "finish_reason": "length"}]}).encode())

    def test_failed_raw_reply_prompt_and_context_persist_then_refresh_bypasses_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            client = ChatClient(base_url="https://provider.test/v1", api_key="private-token-for-test", cache_dir=folder,
                                retries=2, max_tokens=2048)
            context = {"feature_id": 17, "stage": "predictor", "document_id": "doc-4"}
            with patch("monotm.llm.request.urlopen", side_effect=[self.response("unfinished reasoning"), self.response("PREDICTION: 0"),
                                                                  self.response("PREDICTION: 1")]) as http, patch("monotm.llm.time.sleep"):
                self.assertEqual(client.complete("original prompt", "model-v2", validator=lambda s: parse_prediction(s) is not None,
                                                 context=context), "PREDICTION: 0")
                self.assertEqual(client.complete("original prompt", "model-v2", context=context), "PREDICTION: 0")
                self.assertEqual(client.complete("original prompt", "model-v2", context=context, refresh=True), "PREDICTION: 1")
                self.assertEqual(http.call_count, 3)
            attempts = [json.loads(p.read_text()) for p in (Path(folder) / "attempts").glob("*.json")]
            failure = next(a for a in attempts if a["status"] == "failed")
            self.assertIn("unfinished reasoning", failure["raw_response"])
            self.assertEqual(failure["response"], "unfinished reasoning")
            self.assertEqual(failure["request"]["messages"][0]["content"], "original prompt")
            self.assertEqual(failure["request"]["max_tokens"], 2048)
            self.assertEqual(failure["finish_reason"], "length")
            self.assertEqual(failure["context"], context)
            self.assertEqual(len(attempts), 4)
            for path in Path(folder).rglob("*.json"):
                self.assertNotIn("private-token-for-test", path.read_text())

    def test_http_error_body_is_retained_and_credentials_redacted(self):
        with tempfile.TemporaryDirectory() as folder:
            secret = "private-test-credential"
            client = ChatClient(base_url="https://provider.test/v1", api_key=secret, cache_dir=folder)
            failure = HTTPError("url", 400, "bad request", {}, io.BytesIO(f"invalid model; echoed {secret}".encode()))
            with patch("monotm.llm.request.urlopen", side_effect=failure) as http:
                with self.assertRaisesRegex(LLMError, "400"):
                    client.complete("p", "missing-model", context={"feature_id": 2, "stage": "interpreter"})
            attempt = json.loads(next((Path(folder) / "attempts").glob("*.json")).read_text())
            self.assertIn("invalid model", attempt["raw_response"])
            self.assertNotIn(secret, json.dumps(attempt))
            self.assertEqual(attempt["http_status"], 400)
            self.assertEqual(http.call_count, 1)
