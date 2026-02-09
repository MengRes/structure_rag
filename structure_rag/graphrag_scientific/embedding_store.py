"""
Vector store: embed multi-granularity chunks, with Faiss or in-memory index.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional, Tuple

_HAS_FAISS = False
try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    pass

_HAS_EMBED = False
try:
    from sentence_transformers import SentenceTransformer
    _HAS_EMBED = True
except ImportError:
    pass


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    if not _HAS_EMBED:
        return None
    return SentenceTransformer(model_name)


def embed_chunks(
    chunks: List[dict[str, Any]],
    model: Any,
    batch_size: int = 32,
) -> List[List[float]]:
    if model is None:
        return [[0.0] * 384 for _ in chunks]  # dummy
    texts = [c.get("text", "")[:8000] for c in chunks]
    embs = model.encode(texts, batch_size=batch_size, normalize_embeddings=True)
    return embs.tolist()


class EmbeddingStore:
    """In-memory or Faiss index over chunk embeddings."""

    def __init__(self, dim: int = 384, use_faiss: bool = True):
        self.dim = dim
        self.chunks: List[dict[str, Any]] = []
        self.embeddings: List[List[float]] = []
        self.use_faiss = use_faiss and _HAS_FAISS
        self.index: Any = None

    def add(self, chunks: List[dict[str, Any]], embeddings: List[List[float]]) -> None:
        assert len(chunks) == len(embeddings)
        self.chunks.extend(chunks)
        self.embeddings.extend(embeddings)
        if self.use_faiss and self.embeddings:
            import numpy as np
            vecs = np.array(self.embeddings, dtype="float32")
            if self.index is None:
                self.index = faiss.IndexFlatIP(self.dim)
            self.index.add(vecs)

    def search(self, query_embedding: List[float], top_k: int = 10) -> List[Tuple[dict, float]]:
        if not self.chunks:
            return []
        if self.use_faiss and self.index is not None:
            import numpy as np
            q = np.array([query_embedding], dtype="float32")
            scores, indices = self.index.search(q, min(top_k, len(self.chunks)))
            out = []
            for i, sc in zip(indices[0], scores[0]):
                if i < 0:
                    break
                out.append((self.chunks[i], float(sc)))
            return out
        # brute force
        import math
        q = query_embedding
        scored = []
        for c, e in zip(self.chunks, self.embeddings):
            sim = sum(a * b for a, b in zip(q, e))
            scored.append((c, sim))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"chunks": self.chunks, "embeddings": self.embeddings, "dim": self.dim}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        if self.use_faiss and self.index is not None:
            faiss.write_index(self.index, str(path.with_suffix(".faiss")))

    def load(self, path: Path) -> None:
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.chunks = data.get("chunks", [])
        self.embeddings = data.get("embeddings", [])
        self.dim = data.get("dim", 384)
        faiss_path = path.with_suffix(".faiss")
        if faiss_path.exists() and _HAS_FAISS:
            self.index = faiss.read_index(str(faiss_path))
            self.use_faiss = True


def build_embedding_store(
    chunks: List[dict[str, Any]],
    model_name: str = "all-MiniLM-L6-v2",
    use_faiss: bool = True,
) -> EmbeddingStore:
    model = get_embedding_model(model_name)
    embs = embed_chunks(chunks, model)
    dim = len(embs[0]) if embs else 384
    store = EmbeddingStore(dim=dim, use_faiss=use_faiss)
    store.add(chunks, embs)
    return store
