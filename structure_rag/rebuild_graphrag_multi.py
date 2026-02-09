#!/usr/bin/env python3
"""
Parse new PDFs under structure_rag/input (MinerU) and build knowledge graphs for docs that already have content_list, into structure_rag/output/graphrag_all.
Supports incremental: only parse new PDFs and build graphs for docs without schema; document relations from reference_links.

Usage (from MinerU project root):
  python structure_rag/rebuild_graphrag_multi.py
      # Parse PDFs that don't have content_list yet, build graphs for all (full)
  python structure_rag/rebuild_graphrag_multi.py --incremental
      # Parse new PDFs, build graph only for docs without schema, refresh index/relations (recommended for new docs)
  python structure_rag/rebuild_graphrag_multi.py --no-parse
      # Skip MinerU; build graphs from existing content_list only

Document relations (auto + manual):
  - Auto: match document title with reference list titles to generate {paper_id}_reference_links.json,
    then merge into document_relations.json (merged with manual config).
  - Manual: place document_relations.json or {paper_id}_reference_links.json under graphrag_all;
    script merges with auto results.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Project root (for sys.path and demo.demo)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Input/output paths under structure_rag; editable.
# INPUT: PDFs; OUTPUT: MinerU output + GraphRAG output. See structure_rag/README.md.
_STRUCTURE_RAG_DIR = Path(__file__).resolve().parent
INPUT_DIR = _STRUCTURE_RAG_DIR / "input"
OUTPUT_DIR = _STRUCTURE_RAG_DIR / "output"
GRAPHAG_UNIFIED = OUTPUT_DIR / "graphrag_all"


def get_pdfs_to_parse() -> list[Path]:
    """All PDFs under structure_rag/input."""
    if not INPUT_DIR.exists():
        return []
    return sorted(INPUT_DIR.glob("*.pdf"))


def get_content_list_paths() -> list[tuple[Path, str]]:
    """
    Scan structure_rag/output for .../hybrid_auto/*_content_list.json; return [(content_list path, paper_id)].
    paper_id = content_list filename with _content_list.json / _content_list_v2.json stripped.
    """
    out: list[tuple[Path, str]] = []
    if not OUTPUT_DIR.exists():
        return out
    for cl in OUTPUT_DIR.rglob("*_content_list.json"):
        if "_content_list_v2.json" in cl.name:
            stem = cl.stem.replace("_content_list_v2", "")
        else:
            stem = cl.stem.replace("_content_list", "")
        if stem:
            out.append((cl, stem))
    # Dedup: same paper_id prefer flat _content_list.json
    by_id: dict[str, Path] = {}
    for p, pid in out:
        if pid not in by_id:
            by_id[pid] = p
        elif "_content_list_v2" in by_id[pid].name and "_content_list_v2" not in p.name:
            by_id[pid] = p
    return [(by_id[pid], pid) for pid in sorted(by_id)]


def run_mineru_on_new_pdfs(pdf_paths: list[Path]) -> None:
    """Run MinerU parse on PDFs that don't have content_list yet."""
    if not pdf_paths:
        return
    existing_stems = {pid for _, pid in get_content_list_paths()}
    to_parse = [p for p in pdf_paths if p.stem not in existing_stems]
    if not to_parse:
        print("All PDFs already have content_list; skipping MinerU.")
        return
    try:
        from structure_rag.pdf_to_content import parse_pdfs
    except ImportError as e:
        print(f"MinerU not available ({e}); skipping parse.")
        print("Parse the following PDFs with MinerU so structure_rag/output/<doc>/hybrid_auto/ has _content_list.json:")
        for p in to_parse:
            print(f"  - {p}")
        print("E.g. put PDFs in structure_rag/input/, then run this script again, or use MinerU CLI with -o structure_rag/output.")
        return
    print(f"Parsing {len(to_parse)} PDF(s) with MinerU (may take a few minutes)...")
    parse_pdfs(to_parse, OUTPUT_DIR, lang="en", backend="hybrid-auto-engine")


def run_graphrag_for_all(incremental: bool = False) -> None:
    """Build knowledge graphs from content_list, output to GRAPHAG_UNIFIED. incremental=True: only build for docs without schema."""
    content_lists = get_content_list_paths()
    if not content_lists:
        print("No content_list.json found; run MinerU to parse PDFs first.")
        return
    GRAPHAG_UNIFIED.mkdir(parents=True, exist_ok=True)
    try:
        from structure_rag.graphrag_scientific.pipeline import run_pipeline
        from structure_rag.graphrag_scientific.reference_matching import build_reference_links_from_title_matching
        from structure_rag.graphrag_scientific.visualize import (
            build_document_relations_from_reference_links,
            export_document_index,
        )
    except ImportError as e:
        print(f"Run from project root and ensure graphrag_scientific is importable: {e}")
        return

    for cl_path, paper_id in content_lists:
        schema_path = GRAPHAG_UNIFIED / f"{paper_id}_schema.json"
        if incremental and schema_path.exists():
            print(f"Skip (graph exists): {paper_id}")
            continue
        print(f"Building graph: {paper_id} <- {cl_path.name}")
        run_pipeline(
            cl_path,
            GRAPHAG_UNIFIED,
            paper_id=paper_id,
        )

    # Auto-match doc title + reference titles to build reference_links, merge into document_relations, refresh index
    built_links = build_reference_links_from_title_matching(GRAPHAG_UNIFIED)
    if built_links:
        print(f"Built reference links by title match: {len(built_links)} doc(s)")
    build_document_relations_from_reference_links(GRAPHAG_UNIFIED)
    export_document_index(GRAPHAG_UNIFIED)
    print(f"Written: {GRAPHAG_UNIFIED}")
    print("Open index.html to view document relations or a single-document knowledge graph.")


def main() -> None:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ap = argparse.ArgumentParser(description="Parse new PDFs and rebuild knowledge graphs (supports incremental)")
    ap.add_argument("--no-parse", action="store_true", help="Do not run MinerU; build graphs from existing content_list only")
    ap.add_argument("--incremental", action="store_true", help="Build graph only for docs without schema; refresh index/relations (recommended when adding new docs)")
    args = ap.parse_args()
    if not args.no_parse:
        pdfs = get_pdfs_to_parse()
        run_mineru_on_new_pdfs(pdfs)
    run_graphrag_for_all(incremental=args.incremental)


if __name__ == "__main__":
    main()
