"""
Auto-match document paper title with reference list titles: which ref in this doc maps to which PDF in this corpus.
Used to generate {paper_id}_reference_links.json and document_relations for document-level citation display.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional


def normalize_title(s: str) -> str:
    """Normalize to lowercase, strip punctuation, collapse whitespace for comparison."""
    if not s or not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = re.sub(r"[^\w\s\-]", " ", s)
    s = " ".join(s.split())
    return s


def get_document_title_candidates(schema: dict[str, Any], paper_id: str) -> list[str]:
    """
    Get title candidates for this document from schema (deduped, non-empty).
    - paper_id itself (often from PDF filename / paper title)
    - First section title if not generic (Abstract/Introduction etc.)
    """
    candidates: list[str] = []
    seen: set[str] = set()
    n = normalize_title(paper_id)
    if n and n not in seen:
        seen.add(n)
        candidates.append(n)

    sections = schema.get("sections") or []
    generic = {"abstract", "introduction", "1 introduction", "references"}
    for sec in sections:
        t = (sec.get("title") or "").strip()
        if not t:
            continue
        nt = normalize_title(t)
        if not nt or nt in seen:
            continue
        # Skip generic or too short
        if nt in generic or len(nt) < 10:
            continue
        # Skip purely numeric (e.g. "2 related work" kept, "1" alone dropped)
        if re.match(r"^\d+\s*$", nt):
            continue
        seen.add(nt)
        candidates.append(nt)
    return candidates


def _ref_title_for_matching(ref: dict[str, Any]) -> Optional[str]:
    """Get title from one ref for matching: prefer ref['title'], else slice from ref['text']."""
    title = (ref.get("title") or "").strip()
    # Filter out obvious venue names (e.g. "In: Proceedings of ...", "IEEE ...")
    if title and len(normalize_title(title)) >= 10:
        nt = normalize_title(title)
        if not nt.startswith("in ") and "proceedings" not in nt[:30] and "journal" not in nt[:20]:
            return title
    text = (ref.get("text") or "").strip()
    if not text:
        return None
    # Common ref format "Authors: Paper title. Venue. Year" or "Authors. Paper title. ..."
    for sep in (". ", ".: ", ": "):
        if sep in text:
            part = text.split(sep, 1)[-1].strip()
            part = part.split(". ")[0].strip()[:200]
            if len(normalize_title(part)) >= 10:
                return part
    return text[:200].strip() if len(normalize_title(text)) >= 10 else None


def _word_set(s: str) -> set[str]:
    """Normalize and return word set (filter very short words)."""
    n = normalize_title(s)
    return {w for w in n.split() if len(w) >= 2}


def match_ref_to_paper(
    ref_title: str,
    paper_id_to_candidates: dict[str, list[str]],
    exclude_paper_id: str,
    min_ref_len: int = 10,
    word_overlap_ratio: float = 0.6,
) -> Optional[str]:
    """
    Match one ref title to document title candidates; return paper_id in corpus (excluding self).
    Rules: after normalize (1) exact match (2) one contains the other (3) word overlap ratio >= word_overlap_ratio.
    """
    ref_n = normalize_title(ref_title)
    if len(ref_n) < min_ref_len:
        return None
    ref_words = _word_set(ref_title)
    best_pid: Optional[str] = None
    best_score = 0.0

    for pid, candidates in paper_id_to_candidates.items():
        if pid == exclude_paper_id:
            continue
        for doc_n in candidates:
            if not doc_n:
                continue
            if ref_n == doc_n:
                return pid
            if ref_n in doc_n:
                return pid
            if doc_n in ref_n and len(doc_n) >= min_ref_len:
                return pid
            # Word overlap: fraction of ref words that appear in doc title
            doc_words = _word_set(doc_n)
            if ref_words and doc_words:
                overlap = len(ref_words & doc_words) / len(ref_words)
                if overlap >= word_overlap_ratio and overlap > best_score:
                    best_score = overlap
                    best_pid = pid
    return best_pid


def load_schema(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_reference_links_from_title_matching(
    output_dir: Path,
    min_ref_title_len: int = 10,
    merge_with_existing: bool = True,
    word_overlap_ratio: float = 0.6,
) -> dict[str, dict[str, str]]:
    """
    Match each doc's references[].title to in-corpus doc titles (paper_id + first section title);
    generate/merge {paper_id}_reference_links.json (ref_id -> paper_id in corpus) per document.
    Returns { paper_id: { ref_id: target_paper_id } }.
    """
    output_dir = Path(output_dir)
    from .visualize import get_paper_ids_from_output_dir

    paper_ids = get_paper_ids_from_output_dir(output_dir)
    if not paper_ids:
        return {}

    # Title candidates per document
    paper_id_to_candidates: dict[str, list[str]] = {}
    for pid in paper_ids:
        schema_path = output_dir / f"{pid}_schema.json"
        if not schema_path.exists():
            continue
        try:
            schema = load_schema(schema_path)
            paper_id_to_candidates[pid] = get_document_title_candidates(schema, pid)
        except Exception:
            continue

    result: dict[str, dict[str, str]] = {}
    for pid in paper_ids:
        schema_path = output_dir / f"{pid}_schema.json"
        if not schema_path.exists():
            continue
        try:
            schema = load_schema(schema_path)
        except Exception:
            continue
        refs = schema.get("references") or []
        links: dict[str, str] = {}
        if merge_with_existing:
            existing_path = output_dir / f"{pid}_reference_links.json"
            if existing_path.exists():
                try:
                    links = json.loads(existing_path.read_text(encoding="utf-8"))
                    if not isinstance(links, dict):
                        links = {}
                except Exception:
                    links = {}
        for ref in refs:
            ref_id = ref.get("id")
            if not ref_id or not isinstance(ref_id, str):
                continue
            ref_title = _ref_title_for_matching(ref)
            if not ref_title:
                continue
            matched = match_ref_to_paper(
                ref_title,
                paper_id_to_candidates,
                exclude_paper_id=pid,
                min_ref_len=min_ref_title_len,
                word_overlap_ratio=word_overlap_ratio,
            )
            if matched:
                links[ref_id] = matched
        if links:
            result[pid] = links
            out_path = output_dir / f"{pid}_reference_links.json"
            out_path.write_text(json.dumps(links, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
