"""CSV input and stable ID alignment. No gold labels are required."""
from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile
import os
import numpy as np


@dataclass
class Corpus:
    ids: list[str]
    texts: list[str]

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(list(zip(self.ids, self.texts)), ensure_ascii=False).encode()).hexdigest()


def load_corpus(path, text_column="text", id_column=None):
    with Path(path).expanduser().open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        if text_column not in columns:
            raise ValueError(f"CSV needs a {text_column!r} column; found {columns}")
        if id_column and id_column not in columns:
            raise ValueError(f"ID column {id_column!r} not found in CSV")
        ids, texts = [], []
        for index, row in enumerate(reader):
            text = (row.get(text_column) or "").strip()
            if not text:
                raise ValueError(f"Empty text at CSV record {index + 1}; remove or fill it before running")
            identifier = row.get(id_column, "") if id_column else str(index)
            if identifier is None or not identifier.strip():
                raise ValueError(f"Empty document ID at CSV record {index + 1}")
            ids.append(identifier)
            texts.append(text)
    if len(ids) < 4:
        raise ValueError("At least four nonempty documents are required")
    if len(set(ids)) != len(ids):
        raise ValueError("Document IDs must be unique")
    return Corpus(ids, texts)


def validate_corpus_size(n_documents, config):
    if config.n_topics > n_documents:
        raise ValueError("n_topics cannot exceed the number of documents")
    minimum = max(config.min_support, config.min_validation_examples + 1) + config.min_validation_examples + 1
    if n_documents < minimum:
        raise ValueError(f"Too few documents for disjoint feature validation: need at least {minimum} with these support/example settings. Add documents or lower the settings.")


def embedding_layout(path, corpus, ids_path=None):
    """Check a memory-mapped .npy header and ID alignment without scanning values."""
    x = np.load(Path(path).expanduser(), allow_pickle=False, mmap_mode="r")
    if not isinstance(x, np.ndarray) or x.ndim != 2 or x.shape[0] != len(corpus.ids) or x.shape[1] < 2:
        raise ValueError("Embeddings must be a numeric (number_of_documents, dimension >= 2) array")
    if not np.issubdtype(x.dtype, np.number) or np.iscomplexobj(x):
        raise ValueError("Embeddings must contain finite real numbers")
    order = None
    if ids_path is not None:
        ids = json.loads(Path(ids_path).expanduser().read_text(encoding="utf-8"))
        if not isinstance(ids, list) or any(not isinstance(i, str) for i in ids):
            raise ValueError("Embedding IDs must be a JSON array of strings")
        if len(ids) != len(x) or len(set(ids)) != len(ids) or set(ids) != set(corpus.ids):
            raise ValueError("Embedding IDs must match CSV IDs exactly, with no duplicates")
        lookup = {identifier: i for i, identifier in enumerate(ids)}
        order = [lookup[identifier] for identifier in corpus.ids]
    return x, order


def load_embeddings(path, corpus, ids_path=None):
    x, order = embedding_layout(path, corpus, ids_path)
    if order is not None:
        x = x[order]
    if not np.isfinite(x).all():
        raise ValueError("Embeddings must contain finite real numbers")
    return x.astype(np.float32)


def write_json(path, value):
    """Atomically replace a JSON file; interrupted writes cannot corrupt caches."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_csv(path, rows, fieldnames):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
