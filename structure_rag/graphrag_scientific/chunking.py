"""
Multi-granularity chunking: Section / Paragraph / Figure-caption / Formula.

chunk = { type, text, section_id, chunk_id, metadata } for embedding and retrieval.
"""

from __future__ import annotations

from typing import Any

from .schema import load_schema


def _norm(s: str) -> str:
    return " ".join(s.split()) if s else ""


def build_multigranularity_chunks(schema: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Produce multi-granularity chunks with structural context.
    Granularity: section (title + paragraph summary), paragraph, figure_caption, formula.
    """
    chunks: list[dict[str, Any]] = []
    paper_id = schema.get("paper_id", "paper")

    # Abstract as single chunk for retrieval (section_order=-1 means document start)
    abstract = _norm(schema.get("abstract") or "")
    if abstract:
        chunks.append({
            "chunk_id": f"{paper_id}::abstract",
            "type": "abstract",
            "text": abstract,
            "section_id": "",
            "section_title": "Abstract",
            "section_order": -1,
            "section_level": 0,
            "parent_section_id": "",
            "metadata": {},
        })

    for sec in schema.get("sections", []):
        sid = sec["id"]
        title = sec.get("title", "")
        sec_order = sec.get("order", len(chunks))
        sec_level = sec.get("level", 1)
        parent_sid = sec.get("parent") or ""

        # Section-level: title + first N chars of all paragraphs in this section (for high-level retrieval)
        para_texts = [p.get("text", "") for p in sec.get("paragraphs", [])]
        section_body = _norm(" ".join(para_texts))[:1500]
        chunks.append({
            "chunk_id": f"{sid}::section",
            "type": "section",
            "text": f"Section: {title}\n\n{section_body}",
            "section_id": sid,
            "section_title": title,
            "section_order": sec_order,
            "section_level": sec_level,
            "parent_section_id": parent_sid,
            "metadata": {"level": sec_level},
        })

        # Paragraph-level
        for i, p in enumerate(sec.get("paragraphs", [])):
            text = _norm(p.get("text", ""))
            if not text:
                continue
            chunks.append({
                "chunk_id": f"{sid}::p_{i}",
                "type": "paragraph",
                "text": text,
                "section_id": sid,
                "section_title": title,
                "section_order": sec_order,
                "section_level": sec_level,
                "parent_section_id": parent_sid,
                "metadata": {"index": i, "page_idx": p.get("page_idx", 0)},
            })

        # Figure-caption
        for fig in sec.get("figures", []):
            cap = _norm(fig.get("caption", ""))
            fid = fig.get("id", "")
            chunks.append({
                "chunk_id": f"{sid}::{fid}",
                "type": "figure_caption",
                "text": f"Figure {fid}: {cap}",
                "section_id": sid,
                "section_title": title,
                "section_order": sec_order,
                "section_level": sec_level,
                "parent_section_id": parent_sid,
                "metadata": {"figure_id": fid, "caption": cap},
            })

        # Formula
        for eq in sec.get("formulas", []):
            latex = _norm(eq.get("latex", ""))
            eid = eq.get("id", "")
            if not latex:
                continue
            chunks.append({
                "chunk_id": f"{sid}::{eid}",
                "type": "formula",
                "text": f"Formula {eid}: {latex[:500]}",
                "section_id": sid,
                "section_title": title,
                "section_order": sec_order,
                "section_level": sec_level,
                "parent_section_id": parent_sid,
                "metadata": {"formula_id": eid},
            })

        # Table: caption + body preview for retrieval
        for tbl in sec.get("tables", []):
            tid = tbl.get("id", "")
            cap = _norm(tbl.get("caption", ""))
            body = _norm(tbl.get("table_body_preview", ""))[:400]
            if not tid:
                continue
            chunks.append({
                "chunk_id": f"{sid}::{tid}",
                "type": "table_caption",
                "text": f"Table {tid}: {cap}\n{body}".strip(),
                "section_id": sid,
                "section_title": title,
                "section_order": sec_order,
                "section_level": sec_level,
                "parent_section_id": parent_sid,
                "metadata": {"table_id": tid, "caption": cap},
            })

    # Reference: use title only for embedding; authors/venue/year in metadata
    for ref in schema.get("references", []):
        title = _norm(ref.get("title") or "")
        ref_id = ref.get("id", "")
        if not ref_id:
            continue
        # When no title, use first segment of raw text as retrievable content to avoid empty embedding
        text_for_embed = title or _norm(ref.get("text", ""))[:400]
        if not text_for_embed:
            continue
        chunks.append({
            "chunk_id": ref_id,
            "type": "reference",
            "text": text_for_embed,
            "section_id": "",
            "section_title": "",
            "section_order": 99999,
            "section_level": 0,
            "parent_section_id": "",
            "metadata": {
                "number": ref.get("number"),
                "title": title,
                "authors": ref.get("authors", ""),
                "venue": ref.get("venue", ""),
                "year": ref.get("year", ""),
                "full_text": ref.get("text", ""),
            },
        })

    return chunks
