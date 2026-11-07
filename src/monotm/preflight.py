"""Read-only configuration and local environment checks before execution."""
from __future__ import annotations

from importlib import metadata
import json
import os
from pathlib import Path
import platform
from urllib.parse import urlparse

from .config import RunConfig


def _package_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _cached_dimension(config):
    """Read only a local config.json; never import model code or contact the Hub."""
    path = Path(config.embedding_model).expanduser() / "config.json"
    if not path.is_file():
        try:
            from huggingface_hub import try_to_load_from_cache
            cached = try_to_load_from_cache(config.embedding_model, "config.json",
                                             revision=config.embedding_revision or "main")
            path = Path(cached) if isinstance(cached, str) else None
        except (ImportError, ValueError, OSError):
            path = None
    if path and path.is_file():
        try:
            values = json.loads(path.read_text(encoding="utf-8"))
            values = values.get("text_config", values)
            dimension = values.get("hidden_size", values.get("d_model"))
            if isinstance(dimension, int) and dimension >= 2:
                return dimension
        except (ValueError, OSError, AttributeError):
            pass
    return None


def _hardware(requested, required, errors, notes):
    result = {"requested_device": requested, "selected_device": None, "cuda_available": False,
              "mps_available": False, "gpus": [], "needed_for_pending_stages": required}
    try:
        import torch
        from .embeddings import resolve_device
        result.update(torch_version=torch.__version__, cuda_build=torch.version.cuda,
                      cuda_available=torch.cuda.is_available(),
                      mps_available=bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()))
        if result["cuda_available"]:
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                gpu = {"index": index, "name": props.name, "total_memory_gib": round(props.total_memory / 2**30, 2)}
                try:
                    free, _ = torch.cuda.mem_get_info(index)
                    gpu["free_memory_gib"] = round(free / 2**30, 2)
                except (RuntimeError, OSError):
                    gpu["free_memory_gib"] = None
                result["gpus"].append(gpu)
        device = resolve_device(requested)
        result["selected_device"] = (f"cuda:{torch.cuda.current_device() if device.index is None else device.index}"
                                     if device.type == "cuda" else str(device))
        if device.type not in ("cpu", "cuda", "mps"):
            notes.append(f"Availability of device type {device.type} is not checked locally.")
        if requested == "auto" and result["mps_available"] and not result["cuda_available"]:
            notes.append("auto selects CUDA or CPU; Apple MPS requires explicit --device mps.")
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        (errors if required else notes).append(str(exc) if required else f"Device not needed for cached stages: {exc}")
    return result


def check_run(input_csv=None, output_dir=None, *, config=None, text_column="text", id_column=None,
              embeddings_path=None, embedding_ids_path=None):
    """Inspect the effective settings without downloads, training, API calls or writes.

    With an input and an existing output, compare the same run fingerprint used
    by execution. Precomputed matrices are memory-mapped for header/ID checks;
    full-file hashing is needed only to identify an existing run. The check
    excludes array values, model access, memory estimates, and authentication.
    """
    config = config or RunConfig()
    config.validate()
    errors, notes = [], []
    report = {"ready": False, "scope": "local checks only", "config": config.to_dict(),
              "python": platform.python_version(), "errors": errors, "notes": notes,
              "input": {"path": str(Path(input_csv).expanduser().resolve()) if input_csv else None,
                        "text_column": text_column, "id_column": id_column},
              "output": {"path": str(Path(output_dir).expanduser().resolve()) if output_dir else None,
                         "state": "not selected"}, "embedding_dimension": None}
    packages = ("numpy", "scipy", "scikit-learn", "torch", "transformers", "sentencepiece", "accelerate")
    report["packages"] = {name: _package_version(name) for name in packages}
    core_missing = [name for name in packages[:4] if report["packages"][name] is None]
    if core_missing:
        errors.append("Missing core packages: " + ", ".join(core_missing) + "; install MonoTM first.")
    corpus, dimension, manifest = None, None, None
    if input_csv and not core_missing:
        try:
            from .data import embedding_layout, load_corpus, validate_corpus_size
            corpus = load_corpus(Path(input_csv).expanduser(), text_column, id_column)
            report["input"]["documents"] = len(corpus.ids)
            validate_corpus_size(len(corpus.ids), config)
            if embeddings_path:
                matrix, _ = embedding_layout(Path(embeddings_path).expanduser(), corpus, embedding_ids_path)
                dimension = int(matrix.shape[1])
                report["input"]["embeddings_shape"] = list(matrix.shape)
                notes.append("Precomputed embedding header/IDs checked; numerical finiteness is checked during run.")
        except (ValueError, OSError, RuntimeError, ImportError) as exc:
            errors.append(str(exc))
    elif not input_csv:
        notes.append("No input selected; provide --input to check documents and resume identity.")
    if embedding_ids_path and not embeddings_path:
        errors.append("embedding_ids_path requires embeddings_path")
    output = Path(output_dir).expanduser().resolve() if output_dir else None
    if output is not None:
        try:
            if output.exists() and not output.is_dir():
                raise ValueError("Output must be a directory")
            report["output"]["state"] = "new directory" if not output.exists() else "empty directory"
            if (output / "run.lock").exists():
                raise ValueError("Run directory is locked; check whether another process is running")
            if (output / "run.json").is_file():
                report["output"]["state"] = "existing run; identity not checked"
                if corpus is not None and not errors:
                    from .pipeline import run_fingerprint
                    saved = json.loads((output / "run.json").read_text(encoding="utf-8"))
                    fingerprint, _ = run_fingerprint(corpus, config, embeddings_path, embedding_ids_path)
                    if saved.get("fingerprint") != fingerprint:
                        raise ValueError("Output belongs to a different corpus, configuration or runtime; choose another --output")
                    manifest = saved
                    report["output"].update(state="matching run", status=saved["status"])
            elif output.is_dir() and any(output.iterdir()):
                raise ValueError("Output directory must be empty or an existing MonoTM run directory")
        except (ValueError, OSError, ImportError, RuntimeError, KeyError) as exc:
            errors.append(str(exc))

    def reused(stage, *files):
        return bool(manifest and stage in manifest["completed_stages"] and all((output / f).is_file() for f in files))

    stages = {"embeddings": "reuse" if reused("embeddings", "cache/embeddings.npy") else ("precomputed" if embeddings_path else "generate"),
              "mixture_sae": "reuse" if reused("mixture", "cache/mixture_activations.npz", "models/mixture_sae.pt") else "train",
              "topic_mixtures": "reuse" if reused("mixtures", "theta.npy", "cache/mixture_has_features.npy") else "fit on CPU",
              "interpretation_sae": "reuse" if reused("interpretation", "cache/interpretation_activations.npz", "models/interpretation_sae.pt") else "train",
              "validation": "reuse" if reused("validation", "cache/validated_features.json") else "LLM requests",
              "descriptors": "reuse" if reused("descriptors", "beta.npy", "topics.json") else "fit on CPU"}
    report["stages"] = stages
    if dimension is None and manifest:
        dimension = manifest.get("embedding_dimension")
    if dimension is None and not embeddings_path:
        dimension = _cached_dimension(config)
    report["embedding_dimension"] = dimension
    report["sae"] = {}
    for role in ("mixture", "interpretation"):
        sae = getattr(config, role)
        width = int(sae.width * dimension) if dimension is not None else None
        report["sae"][role] = {"width_multiplier": sae.width, "dictionary_size": width, "k": sae.k, "max_steps": sae.steps}
        if width is not None and width < sae.k:
            errors.append(f"{role} SAE dictionary size {width} is smaller than K={sae.k}; adjust width or K")
    device_needed = stages["embeddings"] == "generate" or any(stages[s] == "train" for s in ("mixture_sae", "interpretation_sae"))
    report["hardware"] = _hardware(config.device, device_needed, errors, notes)
    if stages["embeddings"] == "generate":
        missing = [name for name in packages[4:] if report["packages"][name] is None]
        if missing:
            errors.append("Missing embedding packages: " + ", ".join(missing) + "; install with pip install '.[embeddings]'.")
        if report["packages"]["transformers"] not in (None, "4.51.0"):
            errors.append("This embedding loader requires transformers==4.51.0; reinstall MonoTM's embeddings extra.")
        if dimension is None:
            notes.append("Embedding dimension/SAE dictionary sizes are unknown until a model config is available locally.")
    key_name = next((name for name in ("MONOTM_API_KEY", "DEEPINFRA_API_KEY") if os.environ.get(name)), None)
    key_present = bool(key_name and os.environ[key_name].strip())
    try:
        from .llm import validate_base_url
        endpoint = validate_base_url(config.base_url)
    except (ValueError, ImportError) as exc:
        endpoint = "<invalid endpoint>"
        report["config"]["base_url"] = endpoint
        errors.append(str(exc))
    report["llm"] = {"endpoint": endpoint, "key_present": key_present, "key_source": key_name,
                     "requests_needed": stages["validation"] != "reuse",
                     "location": "loopback endpoint" if urlparse(endpoint).hostname in ("localhost", "127.0.0.1", "::1") else "configured endpoint",
                     "device": "managed by endpoint, independent of --device",
                     "requests_per_feature_max": 1 + 2 * config.validation_examples}
    report["llm"]["request_parameters"] = {role: getattr(config, role + "_generation").request_parameters()
                                              for role in ("interpreter", "predictor")}
    if report["llm"]["requests_needed"] and endpoint != "<invalid endpoint>":
        errors.extend(config.generation_issues())
    if report["llm"]["requests_needed"] and not key_present:
        errors.append("Set MONOTM_API_KEY or DEEPINFRA_API_KEY for the configured chat endpoint.")
    notes.append("Model availability/downloads, endpoint connectivity/authentication and GPU memory fit are not checked.")
    notes.append("Generation compatibility checks cover documented model IDs on official endpoints; aliases and other models are checked by the provider during execution.")
    report["ready"] = not errors
    return report


def format_report(report):
    """A terminal/notebook view of the exact resolved run settings."""
    config, hardware, stages = report["config"], report["hardware"], report["stages"]
    input_info, output = report["input"], report["output"]
    lines = ["MonoTM setup", "", f"Input: {input_info['path'] or '(not selected)'}",
             f"Documents: {input_info.get('documents', 'not checked')}; topics: {config['n_topics']}",
             f"Columns: text={input_info['text_column']!r}, id={input_info['id_column']!r}",
             f"Output: {output['path'] or '(not selected)'} ({output['state']})",
             f"Runtime: Python {report['python']}; PyTorch {hardware.get('torch_version', 'unavailable')}; CUDA build {hardware.get('cuda_build')}",
             f"Device: {hardware['requested_device']} -> {hardware['selected_device'] or 'unavailable'}",
             f"CUDA available: {hardware['cuda_available']}; Apple MPS available: {hardware['mps_available']}"]
    for gpu in hardware["gpus"]:
        lines.append(f"  GPU {gpu['index']}: {gpu['name']}; VRAM {gpu['total_memory_gib']} GiB total, {gpu.get('free_memory_gib')} GiB free")
    lines += ["", "Stage / model / execution"]
    embedding = f"precomputed matrix {input_info.get('embeddings_shape', '')}" if stages["embeddings"] == "precomputed" else config["embedding_model"]
    lines.append(f"  Embeddings: {embedding} [{stages['embeddings']}; device={hardware['selected_device']}]")
    for role in ("mixture", "interpretation"):
        sae = report["sae"][role]
        lines.append(f"  {role.title()} SAE: dictionary={sae['dictionary_size'] or 'inferred later'}, width={sae['width_multiplier']} x dimension, K={sae['k']}, max steps={sae['max_steps']} [{stages[role + '_sae']}]")
    lines += [f"  LDA: topics={config['n_topics']}, max iterations={config['lda_max_iter']}, CPU workers={config['lda_jobs']} [{stages['topic_mixtures']}]",
              f"  Shared random seed: {config['seed']}",
              f"  Descriptor EM: {stages['descriptors']}",
              f"  Interpreter: {config['interpreter_model']}", f"  Predictor: {config['predictor_model']}",
              f"  LLM endpoint: {report['llm']['endpoint']} ({report['llm']['location']}; {stages['validation']})",
              "  LLM device is managed by the endpoint, independent of --device.",
              f"API key: {'registered via ' + report['llm']['key_source'] if report['llm']['key_present'] else 'not registered'} (value hidden)",
              f"Validation: support >= {config['min_support']}, IS >= {config['is_threshold']}, feature cap={config['max_features'] or 'all eligible'}",
              f"LLM work: up to {report['llm']['requests_per_feature_max']} requests per feature before cache/retries"]
    for role in ("interpreter", "predictor"):
        parameters = report["llm"]["request_parameters"][role]
        lines.append(f"  {role.title()} request parameters: {json.dumps(parameters, sort_keys=True)}")
        if "temperature" not in parameters:
            lines.append(f"    {role.title()} temperature: omitted (provider default)")
    if report["notes"]:
        lines += ["", "Check scope:"] + [f"  - {note}" for note in report["notes"]]
    if report["errors"]:
        lines += ["", "Fix before running:"] + [f"  - {error}" for error in report["errors"]]
    lines += ["", "Local checks passed." if report["ready"] else "Local checks found blocking issues."]
    return "\n".join(lines)
