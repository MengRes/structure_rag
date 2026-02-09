"""
Unified loader for MinerU parse output: content_list.json.

content_list is the structured list produced by MinerU from PDFs; each item is one block
(section, paragraph, figure, table, formula, reference, etc.).
Supports content_list and content_list_v2; v2 is per-page nested and is flattened to list[dict] here.

Implementation lives in graphrag_scientific.schema; this module re-exports for structure_rag scripts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List


def load_content_list(path: Path) -> List[dict[str, Any]]:
    """
    Load content_list.json or content_list_v2.json into a unified list[dict].
    Each item has type, text/img_path/table_body, page_idx, etc.
    """
    from structure_rag.graphrag_scientific.schema import load_content_list as _load
    return _load(Path(path))
