"""JSON-compatible scikit-learn LDA constructor parameters."""
from __future__ import annotations

import json
import math


LDA_ESTIMATOR = "sklearn.decomposition.LatentDirichletAllocation"
LDA_REFERENCE_URL = "https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html"


LDA_DEFAULTS = {
    "n_components": 20,
    "doc_topic_prior": None,
    "topic_word_prior": None,
    "learning_method": "batch",
    "learning_decay": 0.7,
    "learning_offset": 10.0,
    "max_iter": 200,
    "batch_size": 128,
    "evaluate_every": -1,
    "total_samples": 1e6,
    "perp_tol": 0.1,
    "mean_change_tol": 0.001,
    "max_doc_update_iter": 100,
    "n_jobs": 1,
    "verbose": 0,
    "random_state": 42,
}


def lda_parameters(config, *, overrides=True):
    values = {**LDA_DEFAULTS, "n_components": config.n_topics, "max_iter": config.lda_max_iter,
              "n_jobs": config.lda_jobs, "random_state": config.seed}
    if overrides:
        if not isinstance(config.lda_params, dict):
            raise ValueError("lda_params must be a JSON object of scikit-learn LDA parameters")
        values.update(config.lda_params)
    return values


def validate_lda_parameters(values):
    try:
        json.dumps(values, allow_nan=False)
    except (ValueError, TypeError):
        raise ValueError("lda_params must contain JSON-compatible values; use an integer or null for random_state") from None
    for name, minimum in (("n_components", 1), ("max_iter", 0), ("batch_size", 1),
                          ("max_doc_update_iter", 0), ("verbose", 0)):
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"lda_params.{name} must be an integer >= {minimum}")
    if values["learning_method"] not in ("batch", "online"):
        raise ValueError("lda_params.learning_method must be batch or online")
    for name, minimum, maximum, inclusive in (
        ("doc_topic_prior", 0, 1, True), ("topic_word_prior", 0, 1, True),
        ("learning_decay", 0, 1, True), ("learning_offset", 1, None, True),
        ("total_samples", 0, None, False), ("perp_tol", 0, None, True), ("mean_change_tol", 0, None, True),
    ):
        value = values[name]
        if name in ("doc_topic_prior", "topic_word_prior") and value is None:
            continue
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                or (value < minimum if inclusive else value <= minimum)
                or (maximum is not None and value > maximum)):
            bounds = f">= {minimum}" if inclusive else f"> {minimum}"
            if maximum is not None:
                bounds += f" and <= {maximum}"
            raise ValueError(f"lda_params.{name} must be finite and {bounds}")
    if isinstance(values["evaluate_every"], bool) or not isinstance(values["evaluate_every"], int):
        raise ValueError("lda_params.evaluate_every must be an integer")
    jobs = values["n_jobs"]
    if jobs is not None and (isinstance(jobs, bool) or not isinstance(jobs, int) or jobs == 0):
        raise ValueError("lda_params.n_jobs must be null or a nonzero integer")
    state = values["random_state"]
    if state is not None and (isinstance(state, bool) or not isinstance(state, int) or not 0 <= state < 2**32):
        raise ValueError("lda_params.random_state must be null or an integer in [0, 2**32)")
