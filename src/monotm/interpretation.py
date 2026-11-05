"""Interpreter labels tested on disjoint, balanced activation examples."""
from __future__ import annotations

from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from copy import deepcopy
import logging
from pathlib import Path
import re
import threading
import numpy as np

from .llm import LLMError
from .topics import feature_matrix

logger = logging.getLogger(__name__)
PROMPT_DIR = Path(__file__).with_name("prompts")


def parse_label(text):
    matches = re.findall(r"(?im)^\s*[`*]*FINAL:\s*(.+?)\s*$", text or "")
    if not matches:
        return None
    label = re.sub(r"[`*]", "", matches[-1]).strip()
    return label if label and len(label) <= 300 else None


def parse_prediction(text):
    matches = re.findall(r"(?im)^\s*[`*]*PREDICTION:\s*[`*]*([01])\s*[`*]*\s*$", text or "")
    return int(matches[-1]) if matches else None


def select_examples(activation, config, rng):
    positive = np.flatnonzero(activation > 0)
    negative = np.flatnonzero(activation == 0)
    if len(positive) < config.min_support:
        return None
    # Reserve enough examples for validation before selecting label examples.
    label_capacity = len(positive) - config.min_validation_examples
    zero_count = min(config.zero_examples, len(negative) - config.min_validation_examples)
    if label_capacity < 1 or zero_count < 1:
        return None
    ranked = positive[np.argsort(-activation[positive], kind="stable")]
    maximum = ranked[:min(config.max_examples, label_capacity)]
    remaining = ranked[len(maximum):]
    typical_count = min(config.typical_examples, label_capacity - len(maximum), len(remaining))
    typical = np.array([], dtype=int)
    if typical_count:
        low, high = np.quantile(activation[remaining], [0.3, 0.7])
        pool = remaining[(activation[remaining] >= low) & (activation[remaining] <= high)]
        if len(pool) < typical_count:
            pool = remaining
        typical = rng.choice(pool, size=typical_count, replace=False)
    zeros = rng.choice(negative, size=zero_count, replace=False)
    val_positive = np.setdiff1d(positive, np.concatenate([maximum, typical]))
    val_negative = np.setdiff1d(negative, zeros)
    n = min(config.validation_examples, len(val_positive), len(val_negative))
    if n < config.min_validation_examples:
        return None
    return {"maximum": maximum.tolist(), "typical": typical.tolist(), "zeros": zeros.tolist(),
            "validation_positive": rng.choice(val_positive, size=n, replace=False).tolist(),
            "validation_negative": rng.choice(val_negative, size=n, replace=False).tolist()}


def compute_scores(predictions, ground_truth):
    pred, true = np.asarray(predictions), np.asarray(ground_truth)
    if pred.shape != true.shape or pred.size == 0 or not np.isin(pred, [0, 1]).all() or not np.isin(true, [0, 1]).all():
        raise ValueError("Validation requires equally sized nonempty binary predictions and targets")
    tp = int(((pred == 1) & (true == 1)).sum())
    fp = int(((pred == 1) & (true == 0)).sum())
    fn = int(((pred == 0) & (true == 1)).sum())
    tn = int(((pred == 0) & (true == 0)).sum())
    return {"is_score": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def candidate_features(features, config):
    matrix = feature_matrix(features)
    support = np.asarray((matrix > 0).sum(axis=0)).ravel()
    eligible = ((support >= config.min_support) & (support >= config.min_validation_examples + 1)
                & (matrix.shape[0] - support >= config.min_validation_examples + 1))
    candidates = np.flatnonzero(eligible)
    candidates = sorted(candidates.tolist(), key=lambda f: (-support[f], f))
    if config.max_features is not None:
        candidates = candidates[:config.max_features]
    return candidates


def pending_records(features, config):
    support = np.asarray((feature_matrix(features) > 0).sum(axis=0)).ravel()
    return [{"feature_id": int(f), "support": int(support[f]), "label": "", "is_score": 0.0,
             "status": "pending"} for f in candidate_features(features, config)]


def validate_features(features, corpus, config, chat, *, feature_ids=None, existing=None,
                      stage="both", refresh=False, labels=None, on_record=None, operation_id="run", prompt_dir=None):
    """Evaluate selected features with saved example IDs and incremental checkpoints.

    stage='predictor' retains the current label. All other modes revalidate the
    newly generated (or manually supplied) label on the same held-out documents.
    """
    if stage not in ("both", "predictor"):
        raise ValueError("stage must be both or predictor")
    matrix = feature_matrix(features).tocsc()
    support = np.diff(matrix.indptr)
    candidates = candidate_features(features, config) if feature_ids is None else list(feature_ids)
    previous = {r["feature_id"]: r for r in (existing or [])}
    labels = labels or {}
    if stage == "predictor" and any(not previous.get(f, {}).get("label") for f in candidates):
        raise ValueError("Predictor-only retry requires an existing label for every selected feature")
    prompt_dir = Path(prompt_dir) if prompt_dir else PROMPT_DIR
    interpreter_template = (prompt_dir / "interpreter.txt").read_text(encoding="utf-8")
    predictor_template = (prompt_dir / "predictor.txt").read_text(encoding="utf-8")
    stop = threading.Event()

    def checkpoint(record):
        if on_record is not None:
            on_record(deepcopy(record))

    def evaluate(feature):
        activation = np.asarray(matrix[:, feature].toarray()).ravel()
        rng = np.random.default_rng(np.random.SeedSequence([config.seed, int(feature)]))
        old = previous.get(feature, {})
        if old.get("examples"):
            lookup = {identifier: row for row, identifier in enumerate(corpus.ids)}
            try:
                examples = {key: [lookup[identifier] for identifier in ids] for key, ids in old["examples"].items()}
                positive = examples["maximum"] + examples["typical"] + examples["validation_positive"]
                negative = examples["zeros"] + examples["validation_negative"]
                all_rows = positive + negative
                if (len(set(all_rows)) != len(all_rows) or not examples["maximum"] or not examples["zeros"]
                        or len(examples["validation_positive"]) != len(examples["validation_negative"])
                        or len(examples["validation_positive"]) < config.min_validation_examples
                        or not (activation[positive] > 0).all() or not (activation[negative] == 0).all()):
                    raise ValueError("Saved example IDs no longer match disjoint activation/validation sets")
            except KeyError:
                raise ValueError("Saved examples contain missing groups or unknown document IDs") from None
        else:
            examples = select_examples(activation, config, rng)
        record = {"feature_id": int(feature), "support": int(support[feature]), "label": "", "is_score": 0.0,
                  "operation_id": operation_id, "status": "pending"}
        if examples is None:
            record["status"] = "insufficient_examples"
            checkpoint(record)
            return record
        examples_by_id = {key: [corpus.ids[i] for i in rows] for key, rows in examples.items()}
        record["examples"] = examples_by_id
        def provenance(role):
            result = {"base_url": getattr(chat, "base_url", config.base_url), "operation_id": operation_id}
            if hasattr(chat, "generation_settings"):
                settings = chat.generation_settings(role)
                result["generation"] = settings
                result["request_parameters"] = {settings["token_limit_parameter"]: settings["token_limit"]}
                for key in ("temperature", "reasoning_effort"):
                    if settings[key] is not None:
                        result["request_parameters"][key] = settings[key]
            else:
                result["max_tokens"] = getattr(chat, "max_tokens", None)
            return result
        context = {"feature_id": int(feature), "operation_id": operation_id}

        def document(row):
            return re.sub(r"\s+", " ", corpus.texts[row])[:config.max_chars_per_document]

        def format_rows(rows):
            return "\n\n".join(f"[activation={activation[i]:.4f}] {document(i)}" for i in rows)

        prompt = interpreter_template
        for token, key in (("<max activating examples>", "maximum"), ("<typical activating examples>", "typical"),
                           ("<zero activating examples>", "zeros")):
            prompt = prompt.replace(token, format_rows(examples[key]))
        try:
            if stop.is_set():
                record["status"] = "cancelled"
                return record
            record["status"] = "validating"
            if feature in labels:
                label = labels[feature]
                record["interpreter"] = {"source": "manual", "operation_id": operation_id}
            elif stage == "predictor" or (not refresh and old.get("interpreter", {}).get("source") == "manual" and old.get("label")):
                label = old["label"]
                record["interpreter"] = deepcopy(old.get("interpreter", {"source": "previous_run"}))
            else:
                record["interpreter"] = {**provenance("interpreter"), "model": config.interpreter_model, "source": "llm"}
                checkpoint(record)
                label = parse_label(chat.complete(prompt, config.interpreter_model,
                                                  validator=lambda s: parse_label(s) is not None,
                                                  context={**context, "stage": "interpreter"}, refresh=refresh))
            if label is None:
                raise LLMError("Interpreter did not provide a label")
            record["label"] = label
            record["predictor"] = {**provenance("predictor"), "model": config.predictor_model, "source": "llm"}
            predictions, truth = [], []
            record.update(predictions=predictions, ground_truth=truth, validation_document_ids=[])
            checkpoint(record)
            for key, target in (("validation_positive", 1), ("validation_negative", 0)):
                for row in examples[key]:
                    if stop.is_set():
                        record["status"] = "cancelled"
                        return record
                    predictor_prompt = predictor_template.replace("<description>", label).replace("<text>", document(row))
                    prediction = parse_prediction(chat.complete(predictor_prompt, config.predictor_model,
                                                               validator=lambda s: parse_prediction(s) is not None,
                                                               context={**context, "stage": "predictor", "document_id": corpus.ids[row]},
                                                               refresh=refresh))
                    if prediction is None:
                        raise LLMError("Predictor did not return a binary prediction")
                    predictions.append(prediction)
                    truth.append(target)
                    record["validation_document_ids"].append(corpus.ids[row])
                    checkpoint(record)
            record.update(compute_scores(predictions, truth))
            record.update(status="validated", validation_per_class=len(examples["validation_positive"]),
                          predictions=predictions, ground_truth=truth)
        except LLMError as exc:
            stop.set()
            record.update(status="llm_failed", error=str(exc), error_kind=exc.kind)
        finally:
            checkpoint(record)
        return record

    records = []
    pool = ThreadPoolExecutor(max_workers=config.llm_workers)
    futures = {pool.submit(evaluate, feature): feature for feature in candidates}
    try:
        for future in as_completed(futures):
            try:
                record = future.result()
            except CancelledError:
                feature = futures[future]
                record = {"feature_id": int(feature), "support": int(support[feature]), "label": "",
                          "is_score": 0.0, "status": "pending"}
            records.append(record)
            logger.info("Feature validation %s/%s: feature=%s status=%s IS=%.3f",
                        len(records), len(candidates), record["feature_id"], record["status"], record["is_score"])
            if stop.is_set():
                for pending in futures:
                    pending.cancel()
    finally:
        stop.set()
        for pending in futures:
            pending.cancel()
        pool.shutdown(wait=True)
    return sorted(records, key=lambda r: r["feature_id"])
