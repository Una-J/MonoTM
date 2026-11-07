"""Full-feature LDA followed by fixed-mixture descriptor estimation."""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import LatentDirichletAllocation


def feature_matrix(features):
    features.validate()
    rows = np.repeat(np.arange(len(features.indices)), features.indices.shape[1])
    values, columns = features.values.ravel(), features.indices.ravel()
    positive = values > 0
    matrix = csr_matrix((values[positive], (rows[positive], columns[positive])),
                        shape=(len(features.indices), features.width), dtype=np.float64)
    matrix.sum_duplicates()
    matrix.sort_indices()
    return matrix


def make_lda(config):
    """Resolve constructor options before training or model downloads."""
    config.validate()
    try:
        return LatentDirichletAllocation(**config.lda_parameters())
    except TypeError as exc:
        raise ValueError(f"Invalid scikit-learn LDA parameters: {exc}") from None


def fit_mixtures(features, config):
    lda = make_lda(config)
    matrix = feature_matrix(features)
    feature_ids = np.flatnonzero(np.asarray(matrix.sum(0)).ravel() > 0)
    if not len(feature_ids):
        raise ValueError("Mixture SAE has no positive activations; adjust its configuration")
    matrix = matrix[:, feature_ids]
    theta = lda.fit_transform(matrix)
    return theta, lda.components_, feature_ids, np.asarray(matrix.sum(1)).ravel() > 0


def fit_descriptors(counts, theta, *, max_iter=200, tolerance=1e-6, batch_size=4096):
    """Dirichlet-smoothed EM; theta is copied and never updated.

    eta = 1/T + mean(nonzero features per document)/V controls smoothing.
    The M-step adds eta as a pseudocount, hence convergence is measured on
    the corresponding MAP objective, not on the unregularized likelihood.
    batch_size bounds the number of nonzero entries expanded per operation.
    """
    theta = np.array(theta, dtype=np.float64, copy=True)
    counts = counts.tocsr().astype(np.float64)
    counts.sum_duplicates()
    counts.eliminate_zeros()
    if theta.ndim != 2 or counts.shape[0] != theta.shape[0] or counts.shape[1] < 1:
        raise ValueError("Descriptor counts and theta must have aligned, nonempty dimensions")
    if not np.isfinite(theta).all() or (theta < 0).any() or not np.allclose(theta.sum(1), 1):
        raise ValueError("Theta must contain finite nonnegative probability rows")
    if not np.isfinite(counts.data).all() or (counts.data < 0).any() or counts.nnz == 0:
        raise ValueError("Descriptor counts must have finite nonnegative mass")
    if max_iter < 1 or batch_size < 1 or tolerance <= 0:
        raise ValueError("EM iteration count, batch size and tolerance must be positive")
    n_topics, vocabulary_size = theta.shape[1], counts.shape[1]
    eta = 1.0 / n_topics + np.diff(counts.indptr).mean() / vocabulary_size
    initial = np.asarray((counts.T @ theta).T) + eta
    beta = initial / initial.sum(1, keepdims=True)
    rows = np.repeat(np.arange(counts.shape[0]), np.diff(counts.indptr))
    columns, values = counts.indices, counts.data
    history = []
    for iteration in range(max_iter):
        expected = np.zeros_like(beta)
        for start in range(0, len(values), batch_size):
            end = start + batch_size
            cols = columns[start:end]
            mass = theta[rows[start:end]] * beta[:, cols].T
            denom = np.maximum(mass.sum(1), 1e-300)
            weighted = mass * (values[start:end] / denom)[:, None]
            for topic in range(n_topics):
                np.add.at(expected[topic], cols, weighted[:, topic])
        updated = expected + eta
        beta = updated / updated.sum(1, keepdims=True)
        likelihood = 0.0
        for start in range(0, len(values), batch_size):
            end = start + batch_size
            probability = np.sum(theta[rows[start:end]] * beta[:, columns[start:end]].T, axis=1)
            likelihood += float(values[start:end] @ np.log(np.maximum(probability, 1e-300)))
        objective = likelihood + float(eta * np.log(beta).sum())
        history.append({"iteration": iteration + 1, "log_likelihood": likelihood, "map_objective": objective})
        if len(history) > 1:
            previous = history[-2]["map_objective"]
            if abs(objective - previous) / max(1.0, abs(previous)) < tolerance:
                break
    return beta, {"eta": float(eta), "iterations": len(history), "history": history}
