"""
PDF to text/images parser entry point.

This module wraps MinerU PDF parsing to produce content_list.json, images, markdown, etc.,
for use across structure_rag. Actual parsing is in mineru and demo.demo.

After parsing, output layout:
  <output_dir>/<doc_name>/<backend>/
    ├── <doc_name>_content_list.json   # Structured content (sections, paragraphs, figures, tables, formulas, references)
    ├── <doc_name>.md
    ├── <doc_name>_middle.json
    └── images/
"""

from __future__ import annotations

import sys
from pathlib import Path

from typing import List, Optional


def parse_pdfs(
    pdf_paths: List[Path],
    output_dir: Path,
    *,
    lang: str = "en",
    backend: str = "hybrid-auto-engine",
    method: str = "auto",
    server_url: Optional[str] = None,
    start_page_id: int = 0,
    end_page_id: Optional[int] = None,
) -> None:
    """
    Parse PDFs with MinerU and write per-doc content_list.json, md, images under output_dir.

    :param pdf_paths: List of PDF file paths
    :param output_dir: Output root; each doc yields <output_dir>/<stem>/<backend>/...
    :param lang: Language, e.g. "en", "ch", for OCR
    :param backend: Parser backend, e.g. "hybrid-auto-engine", "pipeline"
    :param method: "auto", "txt", "ocr"
    :param server_url: Service URL when backend is *-http-client
    :param start_page_id: Start page (0-based)
    :param end_page_id: End page (exclusive); None means to end of document
    """
    if not pdf_paths:
        return
    # Ensure project root is on path for demo.demo
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from demo.demo import parse_doc
    except ImportError as e:
        raise ImportError(
            "PDF parsing requires MinerU. Install mineru and ensure demo.demo is available. "
            f"Error: {e}"
        ) from e
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parse_doc(
        list(pdf_paths),
        str(output_dir),
        lang=lang,
        backend=backend,
        method=method,
        server_url=server_url,
        start_page_id=start_page_id,
        end_page_id=end_page_id,
    )
