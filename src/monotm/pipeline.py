"""One ordered document corpus throughout the complete MonoTM pipeline."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import metadata
import json
import logging
import os
from pathlib import Path
import platform
import threading
from uuid import uuid4

import numpy as np
from scipy import sparse

from . import __version__
from .config import GenerationConfig, RunConfig
from .data import load_corpus, load_embeddings, validate_corpus_size, write_csv, write_json
from .embeddings import embed_documents
from .interpretation import PROMPT_DIR, pending_records, validate_features
from .llm import ChatClient, LLMError
from .sae import SAEFeatures, fit_sae
from .topics import feature_matrix, fit_descriptors, fit_mixtures

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    output_dir: Path
    theta: np.ndarray
    beta: np.ndarray
    topics: list[dict]


def _file_hash(path):
    if path is None:
        return None
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _versions():
    result = {"python": platform.python_version(), "monotm": __version__}
    for package in ("numpy", "scipy", "scikit-learn", "torch", "transformers"):
        try:
            result[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            pass
    return result


def run_fingerprint(corpus, config, embeddings_path=None, embedding_ids_path=None, chat_client=None, embedder=None):
    """The same run identity is used by read-only checks and execution."""
    versions = _versions()
    identity_config = config.to_dict()
    for role in ("interpreter", "predictor"):
        key = role + "_generation"
        if identity_config[key] == GenerationConfig().to_dict():
            identity_config.pop(key)
    identity = {"corpus": corpus.fingerprint, "config": identity_config, "versions": versions,
                "embedding_file": _file_hash(embeddings_path), "embedding_ids": _file_hash(embedding_ids_path),
                "prompts": {p.name: _file_hash(p) for p in sorted(PROMPT_DIR.glob("*.txt"))},
                "custom_chat": type(chat_client).__qualname__ if chat_client is not None else None,
                "custom_embedder": getattr(embedder, "__qualname__", None)}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(), versions


def saved_validation_records(output_dir):
    """Read accepted records plus the active incremental validation checkpoint."""
    records = {}
    for name in ("validated_features.json", "validation.partial.json"):
        path = output_dir / "cache" / name
        if path.exists():
            value = json.loads(path.read_text(encoding="utf-8"))
            # Checkpoints may contain a record list or a progress index.
            baseline = value["records"] if isinstance(value, dict) else value
            for record in baseline:
                records[record["feature_id"]] = record
            if isinstance(value, dict):
                checkpoint_id = value["checkpoint_id"]
                if len(checkpoint_id) != 32 or any(c not in "0123456789abcdef" for c in checkpoint_id):
                    raise ValueError("Invalid validation checkpoint ID")
                for file in (output_dir / "cache/validation_progress" / checkpoint_id).glob("*.json"):
                    if not file.stem.isdecimal():
                        continue  # Ignore temporary files from an in-progress atomic write.
                    record = json.loads(file.read_text(encoding="utf-8"))
                    records[record["feature_id"]] = record
    return sorted(records.values(), key=lambda r: r["feature_id"])


def validation_records(output_dir, features, config):
    """Merge saved checkpoints with all eligible feature IDs."""
    records = {r["feature_id"]: r for r in pending_records(features, config)}
    for record in saved_validation_records(output_dir):
        if record["feature_id"] in records:
            records[record["feature_id"]] = record
    return sorted(records.values(), key=lambda r: r["feature_id"])


def validation_checkpoint(output_dir, records):
    lock = threading.Lock()
    positions = {record["feature_id"]: i for i, record in enumerate(records)}
    checkpoint_id = uuid4().hex
    progress = output_dir / "cache/validation_progress" / checkpoint_id
    progress.mkdir(parents=True)
    write_json(output_dir / "cache/validation.partial.json", {"checkpoint_id": checkpoint_id, "records": records})

    def checkpoint(record):
        with lock:
            records[positions[record["feature_id"]]] = record
            # Per-feature atomic writes keep checkpoint I/O proportional to progress.
            write_json(progress / f"{record['feature_id']}.json", record)

    return checkpoint


def finish_validation(output_dir, manifest, records):
    write_feature_scores(output_dir, records)
    if any(r["status"] not in ("validated", "insufficient_examples") for r in records):
        failure = next((r for r in records if r.get("error")), None)
        detail = f" Feature {failure['feature_id']}: {failure['error']}" if failure else ""
        raise LLMError("Feature validation is incomplete." + detail
                       + " Use monotm inspect and monotm retry --failed; completed training is saved.")
    write_json(output_dir / "cache/validated_features.json", records)
    (output_dir / "cache/validation.partial.json").unlink(missing_ok=True)
    if "validation" not in manifest["completed_stages"]:
        manifest["completed_stages"].append("validation")
    write_json(output_dir / "run.json", manifest)
    return records


def write_feature_scores(output_dir, records):
    score_fields = ["feature_id", "label", "support", "is_score", "status", "validation_per_class", "tp", "fp", "fn", "tn"]
    write_csv(output_dir / "feature_scores.csv", ({key: r.get(key, "") for key in score_fields} for r in records), score_fields)


def rebuild_descriptors(output_dir, interpretation_features, theta, records, config):
    """Reestimate only B and its reports; never fit SAEs or LDA."""
    models = output_dir / "models"
    kept = sorted((r for r in records if r["status"] == "validated" and r["label"]
                   and r["support"] >= config.min_support and r["is_score"] >= config.is_threshold), key=lambda r: r["feature_id"])
    if not kept:
        raise ValueError("No features passed validation. Mixtures are saved in theta.npy. Inspect feature_scores.csv; retry selected features with monotm retry, or use a new run with adjusted training settings.")
    counts = feature_matrix(interpretation_features)[:, [r["feature_id"] for r in kept]]
    counts = counts.multiply(np.array([r["is_score"] for r in kept])[None, :]).tocsr()
    logger.info("Learning semantic descriptors with fixed topic mixtures")
    beta, metrics = fit_descriptors(counts, theta, max_iter=config.em_max_iter,
                                   tolerance=config.em_tolerance, batch_size=config.em_batch_size)
    topics, rows = [], []
    for topic in range(config.n_topics):
        descriptors = []
        for rank, column in enumerate(np.argsort(-beta[topic], kind="stable")[:config.top_features], 1):
            r = kept[int(column)]
            row = {"topic": topic, "rank": rank, "feature_id": r["feature_id"], "label": r["label"],
                   "probability": float(beta[topic, column]), "is_score": r["is_score"], "support": r["support"]}
            rows.append(row)
            descriptors.append({key: value for key, value in row.items() if key != "topic"})
        topics.append({"topic": topic, "mean_document_weight": float(theta[:, topic].mean()), "descriptors": descriptors})
    np.save(output_dir / "beta.npy", beta, allow_pickle=False)
    sparse.save_npz(models / "descriptor_counts.npz", counts)
    write_json(models / "descriptor_em.json", metrics)
    write_json(output_dir / "topics.json", topics)
    write_csv(output_dir / "topic_features.csv", rows,
              ["topic", "rank", "feature_id", "label", "probability", "is_score", "support"])
    write_csv(output_dir / "vocabulary.csv",
              ({"column": i, "feature_id": r["feature_id"], "label": r["label"], "is_score": r["is_score"], "support": r["support"]}
               for i, r in enumerate(kept)), ["column", "feature_id", "label", "is_score", "support"])
    return beta, topics, len(kept)


def run(input_csv, output_dir, *, config=None, text_column="text", id_column=None,
        embeddings_path=None, embedding_ids_path=None, chat_client=None, embedder=None):
    """Fit MonoTM to a CSV and write aligned matrices and human-readable topics.

    Optional embeddings are a finite .npy matrix in CSV order. With an ID JSON
    file they are explicitly reordered by ID. Optional chat_client and embedder
    arguments select custom inference backends.
    Reusing a run directory requires an identical corpus, settings and runtime.
    """
    config = config or RunConfig()
    config.validate()
    corpus = load_corpus(input_csv, text_column, id_column)
    validate_corpus_size(len(corpus.ids), config)
    if embedding_ids_path and not embeddings_path:
        raise ValueError("embedding_ids_path requires embeddings_path")
    fingerprint, versions = run_fingerprint(corpus, config, embeddings_path, embedding_ids_path, chat_client, embedder)
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run.json"
    lock_path = output_dir / "run.lock"
    try:
        lock = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise ValueError(f"Run directory is locked: {lock_path}. If no process is running, remove that stale lock.") from None
    os.close(lock)
    manifest = None
    try:
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("fingerprint") != fingerprint:
                manifest = None
                raise ValueError("Output directory belongs to a different corpus, configuration or runtime. Choose a new --output directory.")
        else:
            other_files = [p for p in output_dir.iterdir() if p != lock_path]
            if other_files:
                raise ValueError("Output directory must be empty or an existing MonoTM run directory")
            manifest = {"fingerprint": fingerprint, "config": config.to_dict(), "versions": versions,
                        "input_fingerprint": corpus.fingerprint, "n_documents": len(corpus.ids),
                        "input_columns": {"text": text_column, "id": id_column},
                        "completed_stages": [], "validation_protocol": "disjoint label and balanced validation documents",
                        "embedding_source": "precomputed" if embeddings_path else config.embedding_model}
        manifest["status"] = "running"
        manifest.pop("error", None)
        write_json(manifest_path, manifest)
        cache = output_dir / "cache"
        cache.mkdir(exist_ok=True)
        models = output_dir / "models"
        models.mkdir(exist_ok=True)
        prompts = cache / "prompts"
        prompts.mkdir(exist_ok=True)
        for prompt in PROMPT_DIR.glob("*.txt"):
            if not (prompts / prompt.name).exists():
                (prompts / prompt.name).write_text(prompt.read_text(encoding="utf-8"), encoding="utf-8")
        write_json(output_dir / "document_ids.json", corpus.ids)

        def done(stage):
            if stage not in manifest["completed_stages"]:
                manifest["completed_stages"].append(stage)
            write_json(manifest_path, manifest)

        def cached(stage, *paths):
            return stage in manifest["completed_stages"] and all(Path(p).is_file() for p in paths)

        # Fail before model downloads/training when a new run lacks credentials.
        validation_path = cache / "validated_features.json"
        chat = chat_client
        if not cached("validation", validation_path) and chat is None:
            issues = config.generation_issues()
            if issues:
                raise ValueError(" ".join(issues))
            chat = ChatClient(base_url=config.base_url,
                              api_key=os.environ.get("MONOTM_API_KEY") or os.environ.get("DEEPINFRA_API_KEY"),
                              cache_dir=cache / "llm", timeout=config.llm_timeout, retries=config.llm_retries,
                              stage_options={role: getattr(config, role + "_generation") for role in ("interpreter", "predictor")})

        embedding_cache = cache / "embeddings.npy"
        if cached("embeddings", embedding_cache):
            logger.info("Reusing cached document embeddings")
            embeddings = load_embeddings(embedding_cache, corpus)
        else:
            if embeddings_path:
                embeddings = load_embeddings(embeddings_path, corpus, embedding_ids_path)
                truncated = np.zeros(len(corpus.ids), dtype=bool)
            else:
                embeddings, truncated = (embedder or embed_documents)(corpus.texts, config)
                embeddings = np.asarray(embeddings, dtype=np.float32)
                if embeddings.ndim != 2 or embeddings.shape[0] != len(corpus.ids) or not np.isfinite(embeddings).all():
                    raise ValueError("Embedding backend returned invalid or misaligned document vectors")
            # Validate both SAE dimensions before starting either training run.
            for role in ("mixture", "interpretation"):
                sae = getattr(config, role)
                if int(sae.width * embeddings.shape[1]) < sae.k:
                    raise ValueError(f"{role} SAE dictionary is smaller than K. Adjust its width or K for dimension {embeddings.shape[1]}.")
            np.save(embedding_cache, embeddings, allow_pickle=False)
            manifest["embedding_dimension"] = int(embeddings.shape[1])
            manifest["truncated_documents"] = int(np.sum(truncated)) if not embeddings_path else None
            done("embeddings")

        def get_features(role, seed):
            path = cache / f"{role}_activations.npz"
            checkpoint = models / f"{role}_sae.pt"
            if cached(role, path, checkpoint):
                logger.info("Reusing %s SAE", role)
                return SAEFeatures.load(path, len(corpus.ids))
            logger.info("Training %s SAE", role)
            features, metrics = fit_sae(embeddings, getattr(config, role), seed=seed,
                                        device=config.device, checkpoint_path=checkpoint)
            features.save(path)
            write_json(models / f"{role}_training.json", metrics)
            done(role)
            return features

        mixture_features = get_features("mixture", config.seed)
        theta_path = output_dir / "theta.npy"
        if cached("mixtures", theta_path, cache / "mixture_has_features.npy"):
            theta = np.load(theta_path, allow_pickle=False)
            has_features = np.load(cache / "mixture_has_features.npy", allow_pickle=False)
        else:
            logger.info("Estimating topic mixtures with full-feature LDA")
            theta, components, feature_ids, has_features = fit_mixtures(mixture_features, config)
            np.save(theta_path, theta, allow_pickle=False)
            np.save(models / "lda_components.npy", components, allow_pickle=False)
            np.save(models / "mixture_feature_ids.npy", feature_ids, allow_pickle=False)
            np.save(cache / "mixture_has_features.npy", has_features, allow_pickle=False)
            done("mixtures")
        if theta.shape != (len(corpus.ids), config.n_topics) or not np.isfinite(theta).all() or not np.allclose(theta.sum(1), 1):
            raise ValueError("Cached topic mixtures have invalid dimensions or probabilities")
        write_csv(output_dir / "document_topics.csv",
                  ({"document_id": identifier, **{f"topic_{t}": float(theta[i, t]) for t in range(config.n_topics)}}
                   for i, identifier in enumerate(corpus.ids)),
                  ["document_id"] + [f"topic_{t}" for t in range(config.n_topics)])
        write_csv(output_dir / "documents.csv",
                  ({"document_id": identifier, "top_topic": int(theta[i].argmax()),
                    "top_topic_probability": float(theta[i].max()), "has_mixture_features": bool(has_features[i])}
                   for i, identifier in enumerate(corpus.ids)),
                  ["document_id", "top_topic", "top_topic_probability", "has_mixture_features"])

        interpretation_features = get_features("interpretation", (config.seed + 1) % 2**32)
        if cached("validation", validation_path):
            logger.info("Reusing validated feature labels and scores")
            records = json.loads(validation_path.read_text(encoding="utf-8"))
        else:
            records = validation_records(output_dir, interpretation_features, config)
            checkpoint = validation_checkpoint(output_dir, records)
            todo = [r["feature_id"] for r in records if r["status"] not in ("validated", "insufficient_examples")]
            validate_features(interpretation_features, corpus, config, chat, feature_ids=todo,
                              existing=records, on_record=checkpoint, prompt_dir=prompts)
            records = finish_validation(output_dir, manifest, records)
        write_feature_scores(output_dir, records)
        if cached("descriptors", output_dir / "beta.npy", output_dir / "topics.json"):
            beta = np.load(output_dir / "beta.npy", allow_pickle=False)
            topics = json.loads((output_dir / "topics.json").read_text(encoding="utf-8"))
            n_kept = beta.shape[1]
        else:
            beta, topics, n_kept = rebuild_descriptors(output_dir, interpretation_features, theta, records, config)
            done("descriptors")
        manifest.update(status="completed", n_validated_features=n_kept,
                        documents_without_mixture_features=int((~has_features).sum()))
        write_json(manifest_path, manifest)
        logger.info("Finished: %s topics, %s validated features; outputs in %s", config.n_topics, n_kept, output_dir)
        return RunResult(output_dir, theta, beta, topics)
    except BaseException as exc:
        if manifest is not None:
            manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
            write_json(manifest_path, manifest)
        raise
    finally:
        lock_path.unlink()
