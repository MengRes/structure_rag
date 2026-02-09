#!/usr/bin/env python3
"""
Document-structure-aware GraphRAG for scientific PDFs — entry script.

Build index (from MinerU content_list.json):
  python structure_rag/run_graphrag_scientific.py build structure_rag/output/.../2510.06592v1_content_list.json -o structure_rag/output/.../graphrag_out

Query:
  python structure_rag/run_graphrag_scientific.py query structure_rag/output/.../graphrag_out --query "What is stain normalization?"
  python structure_rag/run_graphrag_scientific.py query structure_rag/output/.../graphrag_out -i
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on path for structure_rag.graphrag_scientific
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from structure_rag.graphrag_scientific.pipeline import (
    run_pipeline,
    load_state,
    query,
    answer_with_llm,
    main as pipeline_main,
)


def main() -> None:
    # build <content_list.json> [-o out_dir]  => build index to out_dir (default: same dir as input / graphrag_out)
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        sys.argv.pop(1)
        rest = sys.argv[1:]
        if rest and not rest[0].startswith("-") and not any(x in rest for x in ("-o", "--output-dir")):
            inp = Path(rest[0])
            if inp.is_file():
                out = inp.parent / "graphrag_out"
                sys.argv = [sys.argv[0], str(inp), "-o", str(out)] + rest[1:]
        pipeline_main()
        return
    # query <index_dir> [--query "..." | -i]
    if len(sys.argv) > 1 and sys.argv[1] == "query":
        sys.argv.pop(1)  # => [script, index_dir, --query ...]
        pipeline_main()
        return
    pipeline_main()


if __name__ == "__main__":
    main()
