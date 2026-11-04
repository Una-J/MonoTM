"""Mean-pooled, L2-normalized Hugging Face document embeddings."""
from __future__ import annotations

import logging
import numpy as np

logger = logging.getLogger(__name__)


def mean_pool(hidden, attention_mask):
    from torch.nn import functional as F
    hidden = hidden.float().masked_fill(~attention_mask[..., None].bool(), 0)
    pooled = hidden.sum(1) / attention_mask.sum(1)[:, None].clamp_min(1)
    return F.normalize(pooled, dim=-1)


def resolve_device(name):
    import torch
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if device.type == "cuda" and device.index is not None and device.index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device index {device.index} is outside the available device count")
    if device.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise ValueError("MPS was requested but is unavailable")
    return device


def embed_documents(texts, config):
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install embedding dependencies with pip install '.[embeddings]', or supply --embeddings") from exc
    import torch
    device = resolve_device(config.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    options = {"trust_remote_code": True}
    if config.embedding_revision:
        options["revision"] = config.embedding_revision
    tokenizer = AutoTokenizer.from_pretrained(config.embedding_model, **options)
    # Nemotron uses left padding; encoder models retain their tokenizer's
    # configured padding side and position IDs.
    if config.embedding_model == "nvidia/llama-embed-nemotron-8b":
        tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("Embedding tokenizer needs a padding or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(config.embedding_model, torch_dtype=dtype,
                                     attn_implementation=config.embedding_attention, **options).to(device).eval()
    result, truncated = [], []
    try:
        with torch.no_grad():
            for start in range(0, len(texts), config.embedding_batch_size):
                batch_text = texts[start:start + config.embedding_batch_size]
                lengths = tokenizer(batch_text, truncation=False, padding=False)["input_ids"]
                truncated.extend(len(tokens) > config.embedding_max_length for tokens in lengths)
                batch = tokenizer(batch_text, padding=True, truncation=True,
                                  max_length=config.embedding_max_length, return_tensors="pt").to(device)
                result.append(mean_pool(model(**batch).last_hidden_state, batch["attention_mask"]).cpu().numpy())
                logger.info("Embedded %s/%s documents", min(start + len(batch_text), len(texts)), len(texts))
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return np.concatenate(result).astype(np.float32), np.asarray(truncated, dtype=bool)
