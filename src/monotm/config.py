"""Configuration for embeddings, autoencoders, validation, and topics."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import math
from typing import Any


@dataclass
class SAEConfig:
    width: float = 0.5
    k: int = 256
    steps: int = 300000
    batch_size: int = 1024
    learning_rate: float = 1e-4
    monitor_every: int = 1000
    patience: int = 3
    min_relative_improvement: float = 0.002
    inference_batch_size: int = 256

    def validate(self):
        if not math.isfinite(self.width) or self.width <= 0:
            raise ValueError("SAE width must be finite and positive")
        for key in ("k", "steps", "batch_size", "monitor_every", "patience", "inference_batch_size"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"SAE {key} must be a positive integer")
        if self.batch_size < 2:
            raise ValueError("SAE batch_size must be at least 2")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("SAE learning_rate must be finite and positive")
        if not 0 <= self.min_relative_improvement < 1:
            raise ValueError("SAE min_relative_improvement must be in [0, 1)")


@dataclass
class RunConfig:
    n_topics: int = 20
    seed: int = 42
    device: str = "auto"
    embedding_model: str = "nvidia/llama-embed-nemotron-8b"
    embedding_revision: str | None = None
    embedding_attention: str = "eager"
    embedding_batch_size: int = 1
    embedding_max_length: int = 32768
    mixture: SAEConfig = field(default_factory=SAEConfig)
    interpretation: SAEConfig = field(default_factory=lambda: SAEConfig(width=2, k=16))
    min_support: int = 30
    max_features: int | None = None
    max_examples: int = 10
    typical_examples: int = 10
    zero_examples: int = 10
    validation_examples: int = 30
    min_validation_examples: int = 5
    max_chars_per_document: int = 8000
    is_threshold: float = 0.8
    interpreter_model: str = "google/gemma-3-27b-it"
    predictor_model: str = "google/gemma-3-12b-it"
    base_url: str = "https://api.deepinfra.com/v1/openai"
    llm_workers: int = 4
    llm_timeout: float = 180
    llm_retries: int = 3
    lda_max_iter: int = 200
    lda_jobs: int = 1
    em_max_iter: int = 200
    em_tolerance: float = 1e-6
    em_batch_size: int = 4096
    top_features: int = 10

    def validate(self):
        self.mixture.validate()
        self.interpretation.validate()
        for key in ("n_topics", "embedding_batch_size", "embedding_max_length", "min_support",
                    "max_examples", "zero_examples", "validation_examples", "min_validation_examples",
                    "max_chars_per_document", "llm_workers", "llm_retries", "lda_max_iter", "em_max_iter",
                    "em_batch_size", "top_features"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if not isinstance(self.typical_examples, int) or self.typical_examples < 0:
            raise ValueError("typical_examples must be a nonnegative integer")
        if self.max_features is not None and (not isinstance(self.max_features, int) or self.max_features < 1):
            raise ValueError("max_features must be a positive integer or null")
        if self.min_validation_examples > self.validation_examples:
            raise ValueError("min_validation_examples cannot exceed validation_examples")
        if not 0 <= self.is_threshold <= 1:
            raise ValueError("is_threshold must be in [0, 1]")
        if not math.isfinite(self.em_tolerance) or self.em_tolerance <= 0:
            raise ValueError("em_tolerance must be finite and positive")
        if not math.isfinite(self.llm_timeout) or self.llm_timeout <= 0:
            raise ValueError("llm_timeout must be finite and positive")
        if self.lda_jobs == 0 or not isinstance(self.lda_jobs, int):
            raise ValueError("lda_jobs must be a nonzero integer")
        for key in ("device", "embedding_model", "interpreter_model", "predictor_model", "base_url"):
            value = getattr(self, key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{key} must be a nonempty string")
        if self.embedding_revision is not None and not isinstance(self.embedding_revision, str):
            raise ValueError("embedding_revision must be a string or null")
        if self.embedding_attention not in ("eager", "flash_attention_2"):
            raise ValueError("embedding_attention must be eager or flash_attention_2")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]):
        if not isinstance(values, dict):
            raise ValueError("Configuration must be a JSON object")
        values = dict(values)
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration keys: {', '.join(sorted(unknown))}")
        for key in ("mixture", "interpretation"):
            if key in values:
                if not isinstance(values[key], dict):
                    raise ValueError(f"{key} must be a JSON object")
                base = asdict(cls().mixture if key == "mixture" else cls().interpretation)
                unknown_sae = set(values[key]) - set(base)
                if unknown_sae:
                    raise ValueError(f"Unknown {key} SAE keys: {', '.join(sorted(unknown_sae))}")
                base.update(values[key])
                values[key] = SAEConfig(**base)
        try:
            config = cls(**values)
            config.validate()
        except (TypeError, AttributeError) as exc:
            raise ValueError(f"Incorrect value type in JSON configuration: {exc}") from None
        return config
