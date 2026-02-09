"""
Unified document schema + MinerU content_list adapter.

Target format:
  paper_id, sections[{ id, title, level, parent, children, paragraphs, figures, tables, formulas, citations }], references[]
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

CITATION_PATTERN = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
# Author-year citations: (Author et al., 2017) or (Ouyang et al., 2022; Bai et al., 2022a)
AUTHOR_YEAR_IN_PAREN = re.compile(
    r"\(([^)]*?(?:19|20)\d{2}[a-z]?[^)]*)\)"
)
# Single citation inside parens: Author et al., 2017 or Author, 2022a
AUTHOR_YEAR_PART = re.compile(
    r"\s*([^,;]+?)\s*,\s*((?:19|20)\d{2}[a-z]?)\s*(?:;|$)",
    re.IGNORECASE,
)
URL_PATTERN = re.compile(r"https?://[^\s<>\"')\]]+")


def _norm(s: str) -> str:
    return " ".join(s.split()) if s else ""


def _extract_urls(text: str) -> list[str]:
    """Extract http(s) URLs from text, dedupe and preserve order."""
    if not text:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for m in URL_PATTERN.finditer(text):
        u = m.group(0).rstrip(".,;:")
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _extract_citation_numbers(text: str) -> list[int]:
    seen = set()
    out = []
    for m in CITATION_PATTERN.finditer(text):
        for part in m.group(1).split(","):
            n = int(part.strip())
            if n not in seen:
                seen.add(n)
                out.append(n)
    return out


def _author_surname(author_str: str) -> str:
    """Extract surname for matching (lowercase). E.g. 'Christiano et al.' -> 'christiano'; 'P. F. Christiano, J. Leike' -> 'christiano' (last word before comma)."""
    s = _norm(author_str)
    if not s:
        return ""
    s = re.sub(r"\bet\s+al\.?\s*", " ", s, flags=re.IGNORECASE).strip()
    # If comma present, take last word of segment before first comma (Last, First format)
    if "," in s:
        s = s.split(",")[0].strip()
    words = re.findall(r"[A-Za-z\u00c0-\u024f\u1e00-\u1eff]+", s)
    return (words[-1].lower() if words else "") or ""


def _extract_author_year_citations(text: str) -> list[tuple[str, str]]:
    """Extract author-year citations from text, e.g. (Christiano et al., 2017), (Ouyang et al., 2022; Bai et al., 2022a). Returns [(author_snippet, year), ...]."""
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for paren_m in AUTHOR_YEAR_IN_PAREN.finditer(text):
        inner = paren_m.group(1)
        for part in re.split(r";", inner):
            part = part.strip()
            if not part:
                continue
            m = AUTHOR_YEAR_PART.match(part + ";")
            if not m:
                # Try matching YYYY or YYYYa in the segment
                ym = re.search(r"((?:19|20)\d{2}[a-z]?)\s*$", part)
                if ym:
                    author = part[: ym.start()].strip().rstrip(",")
                    if author:
                        key = (_norm(author), ym.group(1))
                        if key not in seen:
                            seen.add(key)
                            out.append(key)
                continue
            author = _norm(m.group(1))
            year = m.group(2)
            if not author:
                continue
            key = (author, year)
            if key not in seen:
                seen.add(key)
                out.append(key)
    return out


def _build_author_year_to_ref_id(references: list[dict[str, Any]]) -> dict[tuple[str, str], str]:
    """Build (surname_lower, year_4digit) -> ref_id from references list for author-year citation resolution."""
    m: dict[tuple[str, str], str] = {}
    for r in references:
        ref_id = r.get("id") or ""
        authors = _norm(r.get("authors") or "")
        year = (r.get("year") or "")[:4]  # 2022a -> 2022
        if not year or len(year) != 4:
            continue
        # First author surname from authors: usually "Last, First" or "First Last" or "Last et al."
        surname = _author_surname(authors)
        if not surname:
            # When no authors, use first word of title
            title = _norm(r.get("title") or "")[:80]
            if title:
                tw = re.search(r"[A-Za-z\u00c0-\u024f\u1e00-\u1eff]+", title)
                surname = (tw.group(0).lower() if tw else "") or ref_id
            else:
                continue
        key = (surname, year)
        if key not in m:
            m[key] = ref_id
    return m


def _parse_ref_item(item: str) -> tuple[int, str]:
    item = _norm(item)
    m = re.match(r"^(\d+)\.\s*(.+)$", item)
    if m:
        return int(m.group(1)), m.group(2).strip()
    return 0, item


def parse_reference_text(ref_text: str) -> dict[str, str]:
    """
    Heuristically extract from reference paragraph: title (for embedding), authors, venue, year (metadata).
    Returns {"title", "authors", "venue", "year"}; empty string for unrecognized fields.
    """
    ref_text = _norm(ref_text)
    if not ref_text:
        return {"title": "", "authors": "", "venue": "", "year": ""}

    out: dict[str, str] = {"title": "", "authors": "", "venue": "", "year": ""}

    # Year: last 19xx/20xx
    year_m = list(re.finditer(r"\b(19|20)\d{2}\b", ref_text))
    if year_m:
        out["year"] = year_m[-1].group(0)

    # Venue: In Proceedings / In Proc. / Conference on / Journal of / arXiv
    venue_pattern = re.compile(
        r"\b(?:In\s+(?:Proc\.?|Proceedings\s+of)\s+[^.]*?|"
        r"Conference\s+on\s+[^.]*?|"
        r"Journal\s+of\s+[^.]*?|"
        r"arXiv[^.]*?)",
        re.IGNORECASE,
    )
    venue_m = venue_pattern.search(ref_text)
    if venue_m:
        start = venue_m.start()
        # Up to year or next period/comma
        rest = ref_text[start:]
        end = len(rest)
        for sep in (f", {out['year']}", f". {out['year']}", "."):
            idx = rest.find(sep)
            if idx != -1:
                end = min(end, idx + len(sep) if sep != "." else idx + 1)
        out["venue"] = _norm(rest[:end].rstrip(".,"))

    # After removing venue and year, split by ". ": first segment often authors, rest title
    remainder = ref_text
    if out["venue"]:
        remainder = remainder.replace(out["venue"], " ", 1)
    if out["year"]:
        remainder = re.sub(r"\b" + re.escape(out["year"]) + r"\b", " ", remainder)
    remainder = _norm(remainder)

    parts = [p.strip() for p in re.split(r"\.\s+", remainder) if p.strip()]
    if not parts:
        out["title"] = ref_text[:500]  # fallback: use full paragraph as title
        return out
    if len(parts) == 1:
        out["title"] = parts[0][:500]
        return out
    # If first segment has commas or et al, treat as authors
    first = parts[0]
    if "," in first or "et al" in first.lower():
        out["authors"] = first[:300]
        title_parts = parts[1:]
    else:
        title_parts = parts
    title_str = _norm(". ".join(title_parts))[:500]
    # Drop trailing segment that looks like venue (e.g. starts with "In ...")
    if re.match(r"^\s*In\s+", title_str, re.IGNORECASE):
        title_str = ""
    out["title"] = title_str or ref_text[:500]
    return out


def _v2_content_to_text(items: Any) -> str:
    """Concatenate plain text from v2 content list (e.g. title_content / paragraph_content)."""
    if not items or not isinstance(items, list):
        return ""
    parts: list[str] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        typ = it.get("type") or it.get("item_type") or ""
        content = it.get("content") or it.get("item_content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.append(_v2_content_to_text(content))
        elif isinstance(content, dict) and "content" in content:
            parts.append(str(content.get("content", "")))
    return _norm(" ".join(parts))


def _v2_block_to_flat(block: dict[str, Any], page_idx: int) -> dict[str, Any]:
    """Convert one content_list_v2 block (with content substructure) to flat block (same as content_list)."""
    if not isinstance(block, dict):
        return {"type": "text", "text": "", "page_idx": page_idx}
    btype = block.get("type") or "text"
    content = block.get("content")
    if not isinstance(content, dict):
        out = dict(block)
        out.setdefault("page_idx", page_idx)
        return out

    out: dict[str, Any] = {"page_idx": page_idx}

    if btype == "title":
        out["type"] = "text"
        out["text"] = _v2_content_to_text(content.get("title_content") or [])
        out["text_level"] = int(content.get("level", 1))
        return out

    if btype == "paragraph":
        out["type"] = "text"
        out["text"] = _v2_content_to_text(content.get("paragraph_content") or [])
        return out

    if btype == "image":
        out["type"] = "image"
        cap = content.get("image_caption") or content.get("caption")
        out["image_caption"] = _v2_content_to_text(cap) if isinstance(cap, list) else (cap or "")
        out["caption"] = out["image_caption"]
        src = content.get("image_source")
        out["img_path"] = src.get("path", "") if isinstance(src, dict) else (src or "")
        out["image_path"] = out["img_path"]
        return out

    if btype == "table":
        out["type"] = "table"
        cap = content.get("table_caption") or content.get("caption")
        out["table_caption"] = _v2_content_to_text(cap) if isinstance(cap, list) else (cap or "")
        out["caption"] = out["table_caption"]
        body = content.get("table_body") or content.get("html") or ""
        out["table_body"] = body if isinstance(body, str) else ""
        return out

    if btype in ("equation", "equation_interline"):
        out["type"] = "equation"
        latex = content.get("math_content") or content.get("latex") or content.get("text")
        out["text"] = latex if isinstance(latex, str) else ""
        out["latex"] = out["text"]
        return out

    if btype == "list":
        out["type"] = "list"
        list_type = (content.get("list_type") or "").strip().lower()
        out["sub_type"] = "ref_text" if "reference" in list_type else ""
        raw_items = content.get("list_items") or []
        # v2 list_items: [ { item_type, item_content: [ { type, content } ] } ] -> [ "1. xxx", "2. xxx" ]
        flat_items: list[str] = []
        for it in raw_items:
            if not isinstance(it, dict):
                continue
            item_content = it.get("item_content") or it.get("content")
            s = _v2_content_to_text(item_content) if isinstance(item_content, list) else (it.get("content", "") if isinstance(it.get("content"), str) else "")
            if s:
                flat_items.append(s)
        out["list_items"] = flat_items
        return out

    # Other types (page_header, page_number, algorithm, etc.): convert to placeholder or skip; use text to avoid loss
    out["type"] = "text"
    out["text"] = _v2_content_to_text(content.get("content") if isinstance(content.get("content"), list) else []) or ""
    return out


def _detect_abstract_with_llm(
    paragraphs: list[str],
    model_name: str = "llama3.1:8b",
    max_paragraphs: int = 6,
) -> str:
    """
    Use a small LLM to detect abstract from candidate paragraphs: model returns which paragraph indices (0-based) form the abstract.
    Returns concatenated abstract text; returns "" on failure or when LLM is unavailable.
    """
    if not paragraphs or len(paragraphs) > max_paragraphs:
        paragraphs = paragraphs[:max_paragraphs]
    if not paragraphs:
        return ""

    numbered = "\n\n".join(f"[{i}] {_norm(p)[:800]}" for i, p in enumerate(paragraphs))
    prompt = f"""These are the first few paragraphs of an academic paper. Which paragraph(s) form the abstract? Reply with only the 0-based index or indices, e.g. 0 or 0,1 or 0-2. If none is the abstract, reply -1.

{numbered}

Your reply (only numbers, e.g. 0 or 0,1 or -1):"""

    try:
        import ollama
        r = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
        reply = (r.get("message") or {}).get("content", "") or ""
    except Exception:
        return ""

    # Parse reply: 0 / 0,1 / 0-2 / -1
    reply = reply.strip().split("\n")[0].strip()
    indices: list[int] = []
    for part in reply.replace(" ", "").split(","):
        if "-" in part:
            a, b = part.split("-", 1)
            try:
                lo, hi = int(a.strip()), int(b.strip())
                indices.extend(range(lo, hi + 1))
            except ValueError:
                pass
        else:
            try:
                idx = int(part)
                if idx >= 0:
                    indices.append(idx)
            except ValueError:
                pass
    if not indices:
        return ""
    seen: set[int] = set()
    ordered: list[int] = []
    for i in indices:
        if 0 <= i < len(paragraphs) and i not in seen:
            seen.add(i)
            ordered.append(i)
    ordered.sort()
    return _norm(" ".join(paragraphs[i] for i in ordered))


def load_content_list(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    content_list = raw if isinstance(raw, list) else raw.get("content_list", raw.get("items", []))

    # content_list_v2: per-page nested [ [block, ...], [block, ...] ], block has content substructure
    if content_list and isinstance(content_list[0], list):
        flat: list[dict[str, Any]] = []
        for page_idx, page in enumerate(content_list):
            if not isinstance(page, list):
                continue
            for block in page:
                if isinstance(block, dict):
                    flat.append(_v2_block_to_flat(block, page_idx))
        return flat

    return content_list


def mineru_to_schema(
    content_list: list[dict[str, Any]],
    paper_id: str = "paper",
    use_llm_for_abstract: bool = False,
    abstract_llm_model: str = "llama3.1:8b",
) -> dict[str, Any]:
    """
    Convert MinerU content_list to unified schema.
    Output: { paper_id, sections: [...], references: [...] }
    use_llm_for_abstract: when no explicit Abstract section, use small LLM to detect abstract from first paragraphs.
    abstract_llm_model: Ollama model name for abstract detection.
    """
    sections: list[dict[str, Any]] = []
    section_stack: list[dict[str, Any]] = []  # current path of sections (for parent/children)
    section_by_id: dict[str, dict[str, Any]] = {}
    references: list[dict[str, Any]] = []
    ref_map: dict[int, str] = {}  # number -> ref_id
    current_section: Optional[dict[str, Any]] = None
    section_index = 0
    seen_references_section: bool = False  # whether we have seen a References section
    seen_appendix: bool = False  # after References, if Appendix appears, do not treat following lists as references
    fig_global, tab_global, eq_global = 0, 0, 0

    def _make_section_id(title: str) -> str:
        nonlocal section_index
        section_index += 1
        safe = re.sub(r"[^\w\s\-]", "", title)[:40].strip().replace(" ", "_") or f"sec_{section_index}"
        return f"sec_{section_index}_{safe}"

    def _ensure_section(level: int, title: str) -> dict[str, Any]:
        nonlocal current_section
        sid = _make_section_id(title)
        parent = None
        if section_stack:
            for i in range(len(section_stack) - 1, -1, -1):
                if section_stack[i]["level"] < level:
                    parent = section_stack[i]["id"]
                    break
        while section_stack and section_stack[-1]["level"] >= level:
            section_stack.pop()
        sec = {
            "id": sid,
            "title": _norm(title),
            "level": level,
            "parent": parent,
            "children": [],
            "paragraphs": [],
            "figures": [],
            "tables": [],
            "formulas": [],
            "citations": [],
        }
        sections.append(sec)
        section_by_id[sid] = sec
        section_stack.append(sec)
        if parent and parent in section_by_id:
            section_by_id[parent].setdefault("children", []).append(sid)
        current_section = sec
        return sec

    for block in content_list:
        btype = block.get("type") or "text"
        text = _norm(block.get("text") or "")
        page_idx = block.get("page_idx", 0)

        # Section title
        if btype == "text" and block.get("text_level") is not None:
            level = int(block.get("text_level", 1))
            title = text or f"Section_{level}"
            title_lower = title.strip().lower()
            if title_lower == "references":
                _ensure_section(level, "References")  # create References section for graph display
                current_section = None
                seen_references_section = True
                continue
            # Only after References, treat Appendix as appendix so following lists are not parsed as references
            if seen_references_section and (
                title_lower == "appendix" or title_lower == "appendices" or title_lower.startswith("appendix ")
            ):
                seen_appendix = True
            _ensure_section(level, title)
            continue

        # Reference list: parse as title (for embedding) + authors/venue/year (metadata)
        # If Appendix section already seen, do not treat following lists as references
        if btype == "list" and block.get("sub_type") == "ref_text" and not seen_appendix:
            raw_items = block.get("list_items") or []
            for idx, item in enumerate(raw_items):
                item_str = item if isinstance(item, str) else (str(item) if item else "")
                num, ref_text = _parse_ref_item(item_str)
                if num <= 0:
                    num = idx + 1
                    ref_text = _norm(item_str)
                if not ref_text and not item_str:
                    continue
                ref_id = f"ref_{num}"
                ref_map[num] = ref_id
                parsed = parse_reference_text(ref_text) if ref_text else {"title": "", "authors": "", "venue": "", "year": ""}
                # Keep raw string at least so metadata is not all empty
                fallback_text = ref_text or item_str[:500]
                references.append({
                    "id": ref_id,
                    "number": num,
                    "text": fallback_text,
                    "title": parsed["title"] or fallback_text[:300],
                    "authors": parsed["authors"],
                    "venue": parsed["venue"],
                    "year": parsed["year"],
                })
            continue

        # Image: record preceding paragraph index (previous paragraph in document order)
        if btype == "image":
            fig_global += 1
            caption = block.get("image_caption") or block.get("caption") or []
            if isinstance(caption, list):
                caption = _norm(" ".join(caption))
            else:
                caption = _norm(str(caption))
            img_path = block.get("img_path") or block.get("image_path") or ""
            preceding = len(current_section["paragraphs"]) - 1 if current_section else -1
            fig = {
                "id": f"fig_{fig_global}",
                "caption": caption,
                "image_path": img_path,
                "page_idx": page_idx,
                "preceding_paragraph_index": preceding if preceding >= 0 else None,
            }
            if current_section:
                current_section["figures"].append(fig)
            continue

        # Table
        if btype == "table":
            tab_global += 1
            caption = block.get("table_caption") or block.get("caption") or []
            if isinstance(caption, list):
                caption = _norm(" ".join(caption))
            else:
                caption = _norm(str(caption))
            body = block.get("table_body") or ""
            preceding = len(current_section["paragraphs"]) - 1 if current_section else -1
            tbl = {
                "id": f"tab_{tab_global}",
                "caption": caption,
                "table_body_preview": _norm(body)[:500],
                "page_idx": page_idx,
                "preceding_paragraph_index": preceding if preceding >= 0 else None,
            }
            if current_section:
                current_section["tables"].append(tbl)
            continue

        # Formula
        if btype == "equation":
            eq_global += 1
            latex = block.get("text") or block.get("latex") or ""
            preceding = len(current_section["paragraphs"]) - 1 if current_section else -1
            formula = {
                "id": f"eq_{eq_global}",
                "latex": _norm(latex),
                "page_idx": page_idx,
                "preceding_paragraph_index": preceding if preceding >= 0 else None,
            }
            if current_section:
                current_section["formulas"].append(formula)
            continue

        # Body paragraph: record citations and URLs; if no section yet (e.g. leading text), create Abstract section first
        if btype == "text" and text:
            if current_section is None:
                _ensure_section(1, "Abstract")
            para_citations: list[str] = []
            for num in _extract_citation_numbers(text):
                ref_id = ref_map.get(num)
                if not ref_id:
                    ref_id = f"ref_{num}"
                    ref_map[num] = ref_id
                    references.append({
                        "id": ref_id,
                        "number": num,
                        "text": "",
                        "title": "",
                        "authors": "",
                        "venue": "",
                        "year": "",
                    })
                para_citations.append(ref_id)
                if ref_id not in current_section["citations"]:
                    current_section["citations"].append(ref_id)
            # Author-year citations: (Author et al., 2017) etc., match against references authors/year
            author_year_to_ref = _build_author_year_to_ref_id(references)
            for author_snippet, year in _extract_author_year_citations(text):
                surname = _author_surname(author_snippet)
                year_4 = (year or "")[:4]
                if not surname or not year_4:
                    continue
                key = (surname, year_4)
                ref_id = author_year_to_ref.get(key)
                if not ref_id:
                    ref_id = f"ref_ay_{surname}_{year_4}"
                    if not any(r.get("id") == ref_id for r in references):
                        references.append({
                            "id": ref_id,
                            "number": 0,
                            "text": "",
                            "title": "",
                            "authors": author_snippet[:300],
                            "venue": "",
                            "year": year_4,
                        })
                        author_year_to_ref[key] = ref_id
                para_citations.append(ref_id)
                if ref_id not in current_section["citations"]:
                    current_section["citations"].append(ref_id)
            current_section["paragraphs"].append({
                "text": text,
                "page_idx": page_idx,
                "citations": para_citations,
                "urls": _extract_urls(text),
            })

    # Merge author-year placeholders ref_ay_* with numbered refs ref_i (when body comes before References, ref_ay_* created first, then ref_1, ref_2)
    numbered_ay_to_ref: dict[tuple[str, str], str] = {}
    for r in references:
        rid = r.get("id") or ""
        if rid.startswith("ref_") and not rid.startswith("ref_ay_"):
            surname = _author_surname(r.get("authors") or "")
            year_4 = (r.get("year") or "")[:4]
            if surname and year_4 and len(year_4) == 4:
                key = (surname, year_4)
                if key not in numbered_ay_to_ref:
                    numbered_ay_to_ref[key] = rid
    ay_to_numbered: dict[str, str] = {}
    for r in list(references):
        rid = r.get("id") or ""
        if not rid.startswith("ref_ay_"):
            continue
        parts = rid.replace("ref_ay_", "").split("_", 1)
        if len(parts) == 2:
            surname, year_4 = parts[0], (parts[1])[:4]
            if year_4 and len(year_4) == 4:
                canonical = numbered_ay_to_ref.get((surname, year_4))
                if canonical:
                    ay_to_numbered[rid] = canonical
    if ay_to_numbered:
        for sec in sections:
            sec["citations"] = [ay_to_numbered.get(cid, cid) for cid in sec.get("citations", [])]
            for p in sec.get("paragraphs", []):
                p["citations"] = [ay_to_numbered.get(cid, cid) for cid in p.get("citations", [])]
        references[:] = [r for r in references if (r.get("id") or "") not in ay_to_numbered]

    # Section document order: assign order (0-based) by appearance for sorting and subsection hierarchy
    for i, sec in enumerate(sections):
        sec["order"] = i

    # Abstract: use full text of "Abstract" section if present; else optional LLM detection or heuristic first paragraph
    abstract_text = ""
    for sec in sections:
        if _norm(sec.get("title", "")).lower() == "abstract":
            abstract_text = _norm(" ".join(p.get("text", "") for p in sec.get("paragraphs", [])))
            break
    if not abstract_text and sections:
        first_sec = sections[0]
        paras = first_sec.get("paragraphs", [])
        if paras:
            if use_llm_for_abstract:
                candidate_texts = [_norm(p.get("text", "")) for p in paras if _norm(p.get("text", ""))]
                if candidate_texts:
                    abstract_text = _detect_abstract_with_llm(
                        candidate_texts,
                        model_name=abstract_llm_model,
                        max_paragraphs=6,
                    )
            if not abstract_text and len(_norm(paras[0].get("text", ""))) > 150:
                abstract_text = _norm(paras[0].get("text", ""))

    # Extract URLs from figure/table captions (schema already has caption; used when building KG)
    for sec in sections:
        for fig in sec.get("figures", []):
            fig["urls"] = _extract_urls(fig.get("caption", ""))
        for tbl in sec.get("tables", []):
            tbl["urls"] = _extract_urls(tbl.get("caption", "") + " " + tbl.get("table_body_preview", ""))

    return {
        "paper_id": paper_id,
        "abstract": abstract_text,
        "sections": sections,
        "references": references,
    }


def save_schema(schema: dict[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)


def load_schema(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
