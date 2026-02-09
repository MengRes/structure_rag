"""
Test that author-year citations correctly map to the reference list.

Run from project root:
  python -m graphrag_scientific.test_author_year_citation
or
  python graphrag_scientific/test_author_year_citation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure graphrag_scientific can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from .schema import mineru_to_schema


def _content_list_with_refs_first():
    """Build content_list: References section + reference list first, then body paragraph (with author-year citations)."""
    return [
        # 1. References section title
        {"type": "text", "text": "References", "text_level": 1, "page_idx": 0},
        # 2. Reference list (matches body citations below)
        {
            "type": "list",
            "sub_type": "ref_text",
            "page_idx": 0,
            "list_items": [
                "1. Christiano, P. F., Leike, J., et al. Deep reinforcement learning from human preferences. NeurIPS 2017.",
                "2. Ouyang, L., Wu, J., et al. Training language models to follow instructions with human feedback. 2022.",
                "3. Bai, Y., et al. Constitutional AI. 2022.",
            ],
        },
        # 3. Introduction section
        {"type": "text", "text": "Introduction", "text_level": 1, "page_idx": 0},
        # 4. Body paragraph with author-year citations (Christiano et al., 2017), (Ouyang et al., 2022; Bai et al., 2022)
        {
            "type": "text",
            "text": "We use RLHF (Christiano et al., 2017) and instruction tuning (Ouyang et al., 2022; Bai et al., 2022).",
            "page_idx": 1,
        },
    ]


def _content_list_body_first():
    """Build content_list: body first (with author-year citations), then References + list. Simulates real document order."""
    return [
        {"type": "text", "text": "Introduction", "text_level": 1, "page_idx": 0},
        {
            "type": "text",
            "text": "We use RLHF (Christiano et al., 2017) and (Ouyang et al., 2022; Bai et al., 2022).",
            "page_idx": 0,
        },
        {"type": "text", "text": "References", "text_level": 1, "page_idx": 1},
        {
            "type": "list",
            "sub_type": "ref_text",
            "page_idx": 1,
            "list_items": [
                "1. Christiano, P. F., Leike, J., et al. Deep reinforcement learning from human preferences. NeurIPS 2017.",
                "2. Ouyang, L., Wu, J., et al. Training language models to follow instructions with human feedback. 2022.",
                "3. Bai, Y., et al. Constitutional AI. 2022.",
            ],
        },
    ]


def test_refs_first():
    """When references appear first in content_list, paragraph author-year citations should map to ref_1, ref_2, ref_3."""
    content_list = _content_list_with_refs_first()
    schema = mineru_to_schema(content_list, paper_id="test")
    refs = {r["id"]: r for r in schema["references"]}
    # Should have exactly 3 references, no ref_ay_*
    assert len(schema["references"]) == 3, schema["references"]
    intro_paras = []
    for sec in schema["sections"]:
        if (sec.get("title") or "").lower() == "introduction":
            intro_paras = sec.get("paragraphs", [])
            break
    assert len(intro_paras) == 1
    citations = intro_paras[0].get("citations", [])
    # (Christiano et al., 2017) -> ref_1, (Ouyang et al., 2022) -> ref_2, (Bai et al., 2022) -> ref_3
    assert "ref_1" in citations, f"expected ref_1 in {citations}"
    assert "ref_2" in citations, f"expected ref_2 in {citations}"
    assert "ref_3" in citations, f"expected ref_3 in {citations}"
    assert refs["ref_1"].get("year") == "2017"
    assert "Christiano" in (refs["ref_1"].get("authors") or "") or "christiano" in (refs["ref_1"].get("authors") or "").lower()
    print("test_refs_first: OK — author-year citations map to ref_1, ref_2, ref_3")


def test_body_first():
    """When body comes before References, parsing should merge ref_ay_* into ref_1, ref_2, ref_3."""
    content_list = _content_list_body_first()
    schema = mineru_to_schema(content_list, paper_id="test")
    refs = {r["id"]: r for r in schema["references"]}
    # After merge should have exactly 3 references
    assert len(schema["references"]) == 3, [r["id"] for r in schema["references"]]
    assert "ref_ay_" not in str(schema["references"]), "ref_ay_* should be merged away"
    intro_paras = []
    for sec in schema["sections"]:
        if (sec.get("title") or "").lower() == "introduction":
            intro_paras = sec.get("paragraphs", [])
            break
    assert len(intro_paras) == 1
    citations = intro_paras[0].get("citations", [])
    assert "ref_1" in citations
    assert "ref_2" in citations
    assert "ref_3" in citations
    assert refs["ref_1"].get("year") == "2017"
    print("test_body_first: OK — with body first, merge still maps to ref_1, ref_2, ref_3")


if __name__ == "__main__":
    test_refs_first()
    test_body_first()
    print("All tests passed: author-year citations correctly map to references.")
