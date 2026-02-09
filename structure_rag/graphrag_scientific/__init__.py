"""
Document-Structure-Aware GraphRAG for Scientific PDFs.

Pipeline: MinerU JSON → Schema → KG → Multi-granularity chunks → Embedding + Graph →
         Hybrid retrieval (semantic + graph expand + rerank) → LLM generation → Visualization.
"""

from pathlib import Path

__all__ = [
    "schema",
    "kg_builder",
    "chunking",
    "embedding_store",
    "retriever",
    "pipeline",
    "visualize",
]
