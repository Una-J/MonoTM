"""Corpus-trained tied Top-K SAE with post-selection positive gains."""
from __future__ import annotations

from dataclasses import dataclass
import logging
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .embeddings import resolve_device

logger = logging.getLogger(__name__)


class TopKSAE(nn.Module):
    def __init__(self, dimension, width, k):
        super().__init__()
        self.decoder = nn.Parameter(torch.randn(dimension, width) / np.sqrt(dimension))
        self.bias = nn.Parameter(torch.zeros(width))
        self.log_gain = nn.Parameter(torch.zeros(width))
        self.k = k

    def sparse_codes(self, x):
        values, indices = torch.topk(F.relu(x @ self.decoder + self.bias), self.k, dim=-1)
        return indices, values * self.log_gain.exp()[indices]

    def forward(self, x):
        indices, values = self.sparse_codes(x)
        h = torch.zeros((len(x), self.decoder.shape[1]), device=x.device, dtype=x.dtype)
        h.scatter_(1, indices, values)
        return h @ self.decoder.T

    @torch.no_grad()
    def renormalize(self):
        self.decoder.div_(self.decoder.norm(dim=0, keepdim=True).clamp_min(1e-8))


@dataclass
class SAEFeatures:
    indices: np.ndarray
    values: np.ndarray
    width: int

    def validate(self, n_documents=None):
        if self.indices.ndim != 2 or self.values.shape != self.indices.shape:
            raise ValueError("SAE index/value arrays must have the same two-dimensional shape")
        if n_documents is not None and len(self.indices) != n_documents:
            raise ValueError("SAE activations do not match the corpus row count")
        if self.width < 1 or not np.issubdtype(self.indices.dtype, np.integer):
            raise ValueError("SAE feature IDs must be integers within the dictionary")
        if (self.indices < 0).any() or (self.indices >= self.width).any():
            raise ValueError("SAE feature IDs are outside the dictionary")
        if not np.isfinite(self.values).all() or (self.values < 0).any():
            raise ValueError("SAE activations must be finite and nonnegative")

    def save(self, path):
        np.savez_compressed(path, indices=self.indices, values=self.values, width=np.array(self.width))

    @classmethod
    def load(cls, path, n_documents=None):
        with np.load(path, allow_pickle=False) as data:
            result = cls(data["indices"], data["values"], int(data["width"]))
        result.validate(n_documents)
        return result


def normalized_mse(target, prediction):
    error = F.mse_loss(prediction, target)
    baseline = F.mse_loss(target, target.mean(0, keepdim=True).expand_as(target))
    return error / baseline.clamp_min(1e-8)


def fit_sae(embeddings, config, *, seed, device, checkpoint_path):
    config.validate()
    if embeddings.ndim != 2 or len(embeddings) < 4 or not np.isfinite(embeddings).all():
        raise ValueError("SAE training requires at least four finite embedding rows")
    dimension = embeddings.shape[1]
    width = int(config.width * dimension)
    if width < config.k:
        raise ValueError(f"SAE width {width} is smaller than K={config.k}. Lower K or increase width.")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    target_device = resolve_device(device)
    mean = embeddings.mean(0).astype(np.float32)
    scale = (embeddings.std(0) + 1e-8).astype(np.float32)
    x = torch.from_numpy(((embeddings - mean) / scale).astype(np.float32))
    model = TopKSAE(dimension, width, config.k).to(target_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.999))
    generator = torch.Generator().manual_seed(seed)
    monitor_rows = torch.randperm(len(x), generator=generator)[:max(2, int(round(len(x) * 0.1)))]
    monitor = x[monitor_rows]
    best_loss, best_state, significant_best = float("inf"), None, None
    stale, step, history = 0, 0, []
    batch_size = min(config.batch_size, len(x))
    while step < config.steps:
        order = torch.randperm(len(x), generator=generator)
        stop = False
        for offset in range(0, len(x), batch_size):
            batch = x[order[offset:offset + batch_size]]
            if len(batch) < 2:
                continue
            batch = batch.to(target_device)
            optimizer.zero_grad(set_to_none=True)
            loss = normalized_mse(batch, model(batch))
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite SAE training loss; reduce learning_rate")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.renormalize()
            step += 1
            if step % config.monitor_every == 0 or step == config.steps:
                model.eval()
                with torch.no_grad():
                    scores = []
                    for start in range(0, len(monitor), max(2, min(512, len(monitor)))):
                        m = monitor[start:start + 512].to(target_device)
                        if len(m) >= 2:
                            scores.append(float(normalized_mse(m, model(m)).cpu()))
                    score = float(np.mean(scores))
                history.append({"step": step, "monitor_nmse": score})
                logger.info("SAE N=%s K=%s step=%s monitor NMSE=%.5f", config.width, config.k, step, score)
                if score < best_loss:
                    best_loss = score
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                if significant_best is None or (significant_best - score) / max(significant_best, 1e-12) >= config.min_relative_improvement:
                    significant_best, stale = score, 0
                else:
                    stale += 1
                model.train()
                if stale >= config.patience:
                    stop = True
            if stop or step >= config.steps:
                break
        if stop:
            break
    if best_state is None:
        raise RuntimeError("SAE training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    indices, values = [], []
    with torch.no_grad():
        for offset in range(0, len(x), config.inference_batch_size):
            idx, val = model.sparse_codes(x[offset:offset + config.inference_batch_size].to(target_device))
            indices.append(idx.cpu().numpy().astype(np.int32))
            values.append(val.cpu().numpy().astype(np.float32))
    torch.save({"state_dict": best_state, "mean": torch.from_numpy(mean), "scale": torch.from_numpy(scale),
                "dimension": dimension, "width": width, "k": config.k, "seed": seed,
                "training_config": vars(config)}, checkpoint_path)
    features = SAEFeatures(np.concatenate(indices), np.concatenate(values), width)
    features.validate(len(x))
    metrics = {"steps_completed": step, "best_monitor_nmse": best_loss,
               "monitor_is_training_subset": True, "history": history,
               "positive_activation_fraction": float(np.mean(features.values > 0))}
    del model, optimizer
    if target_device.type == "cuda":
        torch.cuda.empty_cache()
    return features, metrics
