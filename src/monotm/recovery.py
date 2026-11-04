"""Inspect and repair LLM stages using immutable trained features and mixtures."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from uuid import uuid4

import numpy as np

from .config import RunConfig
from .data import load_corpus, write_json
from .interpretation import PROMPT_DIR, validate_features
from .llm import ChatClient
from .pipeline import (_file_hash, _versions, finish_validation, rebuild_descriptors,
                       saved_validation_records, validation_checkpoint, validation_records, write_feature_scores)
from .sae import SAEFeatures

TERMINAL = ("validated", "insufficient_examples")
DESCRIPTORS = ("beta.npy", "topics.json", "topic_features.csv", "vocabulary.csv",
               "models/descriptor_counts.npz", "models/descriptor_em.json")
PROTECTED = ("theta.npy", "document_ids.json", "cache/embeddings.npy",
             "cache/mixture_activations.npz", "cache/interpretation_activations.npz",
             "models/mixture_sae.pt", "models/interpretation_sae.pt",
             "models/lda_components.npy", "models/mixture_feature_ids.npy")


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def inspect_run(output_dir, *, feature_id=None):
    """Return feature records and request diagnostics.

    A feature-specific view contains the full prompts and raw HTTP response
    bodies for every attempt. The summary omits these potentially long fields.
    """
    output_dir = Path(output_dir).expanduser().resolve()
    manifest = _read_json(output_dir / "run.json")
    records = saved_validation_records(output_dir)
    result = {"output_dir": str(output_dir), "status": manifest["status"],
              "status_counts": dict(Counter(r["status"] for r in records)),
              "is_threshold": manifest["config"]["is_threshold"],
              "low_score_feature_ids": [r["feature_id"] for r in records if r["status"] == "validated"
                                        and r["is_score"] < manifest["config"]["is_threshold"]],
              "features": records, "recovery_history": manifest.get("recovery_history", [])}
    if feature_id is not None:
        result["features"] = [r for r in records if r["feature_id"] == feature_id]
        if not result["features"]:
            raise ValueError(f"Feature {feature_id} is not a validation candidate in this run")
        attempts = [_read_json(p) for p in (output_dir / "cache/llm/attempts").glob("*.json") if not p.name.startswith(".")]
        result["attempts"] = sorted((a for a in attempts if a.get("context", {}).get("feature_id") == feature_id),
                                    key=lambda a: (a["timestamp"], a["attempt_id"]))
        if not result["attempts"]:
            result["note"] = "No LLM request attempts are recorded for this feature."
    return result


def retry_features(output_dir, input_csv, *, feature_ids=None, failed=False, stage="both",
                   interpreter_model=None, predictor_model=None, base_url=None,
                   refresh=None, max_tokens=1024, label=None, text_column=None,
                   id_column=None, chat_client=None):
    """Repair selected LLM stages and rebuild B, without embeddings/SAE/LDA fitting.

    Explicit feature IDs default to fresh requests, including well-formed cached
    responses. failed=True resumes unfinished features using accepted responses.
    Every new label is validated; predictor-only mode keeps its existing label.
    """
    output_dir = Path(output_dir).expanduser().resolve()
    if not (output_dir / "run.json").is_file():
        raise ValueError("--run must be an existing MonoTM run directory")
    if failed == (feature_ids is not None):
        raise ValueError("Select either feature_ids or failed=True")
    if stage not in ("both", "predictor"):
        raise ValueError("stage must be both or predictor")
    if stage == "predictor" and (label is not None or interpreter_model is not None):
        raise ValueError("Predictor-only retry retains its label; omit --label and --interpreter-model")
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
        raise ValueError("max_tokens must be a positive integer")
    lock_path = output_dir / "run.lock"
    try:
        lock = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise ValueError(f"Run directory is locked: {lock_path}") from None
    os.close(lock)
    manifest, operation, records = None, None, None
    try:
        manifest = _read_json(output_dir / "run.json")
        original_config = RunConfig.from_dict(manifest["config"])
        config = deepcopy(original_config)
        columns = manifest.get("input_columns", {})
        corpus = load_corpus(input_csv, text_column or columns.get("text", "text"),
                             id_column if id_column is not None else columns.get("id"))
        if (corpus.fingerprint != manifest["input_fingerprint"]
                or corpus.ids != _read_json(output_dir / "document_ids.json")):
            raise ValueError("Recovery input must have the identical text, IDs and row order as the original corpus")
        if not {"mixtures", "interpretation"}.issubset(manifest["completed_stages"]):
            raise ValueError("Saved interpretation SAE activations and LDA mixtures are required; finish those stages with monotm run first")
        features = SAEFeatures.load(output_dir / "cache/interpretation_activations.npz", len(corpus.ids))
        theta = np.load(output_dir / "theta.npy", allow_pickle=False)
        if (theta.shape != (len(corpus.ids), config.n_topics) or not np.isfinite(theta).all()
                or (theta < 0).any() or not np.allclose(theta.sum(axis=1), 1)):
            raise ValueError("Saved topic mixtures have invalid dimensions or probabilities")
        records = validation_records(output_dir, features, original_config)
        by_id = {r["feature_id"]: r for r in records}
        selected = ([r["feature_id"] for r in records if r["status"] not in TERMINAL]
                    if failed else sorted(set(feature_ids)))
        descriptors_missing = ("descriptors" not in manifest["completed_stages"]
                               or not all((output_dir / name).is_file() for name in DESCRIPTORS))
        can_rebuild = (failed and descriptors_missing and any(r["status"] == "validated"
                       and r["is_score"] >= original_config.is_threshold for r in records))
        if not selected and not can_rebuild:
            raise ValueError("No unfinished features. Select --features to retry a validated or low-scoring label")
        if any(f not in by_id for f in selected):
            raise ValueError("Every selected feature must be a validation candidate in this run; use monotm inspect for IDs")
        if stage == "predictor" and any(not by_id[f].get("label") for f in selected):
            raise ValueError("Predictor-only retry requires an existing label for every selected feature")
        if label is not None:
            label = label.strip()
            if failed or len(selected) != 1 or not label or len(label) > 300:
                raise ValueError("--label requires exactly one --features ID and a nonempty label of at most 300 characters")
        for name, value in (("interpreter_model", interpreter_model), ("predictor_model", predictor_model), ("base_url", base_url)):
            if value is not None:
                setattr(config, name, value)
        config.validate()
        refresh = not failed if refresh is None else refresh
        chat = chat_client
        if chat is None and selected:
            chat = ChatClient(base_url=config.base_url,
                              api_key=os.environ.get("MONOTM_API_KEY") or os.environ.get("DEEPINFRA_API_KEY"),
                              cache_dir=output_dir / "cache/llm", timeout=config.llm_timeout,
                              retries=config.llm_retries, max_tokens=max_tokens)
        prompt_dir = output_dir / "cache/prompts"
        if not all((prompt_dir / name).is_file() for name in ("interpreter.txt", "predictor.txt")):
            prompt_dir = PROMPT_DIR  # Use bundled templates when no saved snapshot is present.
        source_hashes = {name: _file_hash(output_dir / name) for name in PROTECTED if (output_dir / name).exists()}
        operation_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8]
        archive = output_dir / "cache/repairs" / operation_id
        archive.mkdir(parents=True)
        write_json(archive / "before.json", {"manifest": manifest, "features": records})
        for name in (*DESCRIPTORS, "feature_scores.csv"):
            source = output_dir / name
            if source.is_file():
                target = archive / "before" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        operation = {"operation_id": operation_id, "feature_ids": selected, "stage": stage,
                     "refresh": refresh, "interpreter_model": config.interpreter_model,
                     "predictor_model": config.predictor_model, "base_url": config.base_url,
                     "max_tokens": max_tokens, "manual_label": label is not None, "status": "running",
                     "source_hashes": source_hashes, "versions": _versions(),
                     "prompt_hashes": {p.name: _file_hash(p) for p in prompt_dir.glob("*.txt")}}
        manifest.setdefault("recovery_history", []).append(operation)
        manifest["completed_stages"] = [s for s in manifest["completed_stages"] if s not in ("validation", "descriptors")]
        manifest.update(status="repairing")
        manifest.pop("error", None)
        manifest.pop("n_validated_features", None)
        write_json(output_dir / "run.json", manifest)
        # Invalidate descriptors until validation and estimation complete.
        for name in DESCRIPTORS:
            (output_dir / name).unlink(missing_ok=True)
        (output_dir / "cache/validated_features.json").unlink(missing_ok=True)
        previous = deepcopy(records)
        for record in records:
            if record["feature_id"] in selected:
                record.update(status="pending", is_score=0.0)
        checkpoint = validation_checkpoint(output_dir, records)
        if selected:
            validate_features(features, corpus, config, chat, feature_ids=selected, existing=previous,
                              stage=stage, refresh=refresh, labels={selected[0]: label} if label is not None else None,
                              on_record=checkpoint, operation_id=operation_id, prompt_dir=prompt_dir)
        finish_validation(output_dir, manifest, records)
        beta, topics, n_kept = rebuild_descriptors(output_dir, features, theta, records, original_config)
        for name, digest in source_hashes.items():
            if _file_hash(output_dir / name) != digest:
                raise RuntimeError(f"Trained artifact unexpectedly changed during recovery: {name}")
        operation.update(status="completed", trained_artifacts_unchanged=True)
        manifest.update(status="completed", n_validated_features=n_kept)
        manifest["completed_stages"].append("descriptors")
        write_json(archive / "after.json", {"operation": operation, "features": records})
        write_json(output_dir / "run.json", manifest)
        return inspect_run(output_dir)
    except BaseException as exc:
        # Invalid input/arguments do not alter a previous successful run.
        if operation is not None:
            operation.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
            manifest.update(status=operation["status"], error=str(exc))
            write_feature_scores(output_dir, records)
            write_json(archive / "after.json", {"operation": operation, "features": records})
            write_json(output_dir / "run.json", manifest)
        raise
    finally:
        lock_path.unlink()
