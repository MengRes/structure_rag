# Structured PDF Knowledge Graph — Improvement Checklist

Based on a review of the current pipeline (schema → KG → chunking → embedding → retriever → generation), the following improvements are suggested, ordered by **priority**.

---

## High priority

### 1. Support content_list_v2 format

**Current**: Only a flat block list is supported (`[{type, text, text_level, ...}, ...]`). MinerU can also produce `_content_list_v2.json` with a **per-page nested list** structure (`[[block, ...], [block, ...], ...]`); passing it directly raises `'list' object has no attribute 'get'`.

**Suggestion**: At the `load_content_list()` or `mineru_to_schema()` entry, detect format: if the first element of `content_list` is a list, flatten with `[block for page in content_list for block in page]` and then run the existing logic. This supports both v2 and flat format.

---

### 2. Table chunks for retrieval

**Current**: The KG has Table nodes (caption + table_body_preview), but **chunking does not produce table chunks**; table content is not semantically retrievable and does not enter LLM context.

**Suggestion**: In `chunking.py`, add a chunk type for each table, similar to figure_caption, e.g.:
- `type: "table_caption"`
- `text: f"Table {tid}: {caption}\n{table_body_preview}"`
- `chunk_id: f"{sid}::{tid}"`
- `retriever._chunk_to_node_ids` already supports `tab_*`; only the chunk output needs to be added.

---

### 3. Stronger Abstract detection

**Current**: When there is no "Abstract" section, the first paragraph of the first section with length > 150 is used as the abstract. Often the first paragraph is author/affiliation and the second is the abstract.

**Suggestion**:
- Within the first section, take the **first paragraph with length > 200** as the abstract; if several exceed 200, take the first or first two concatenated (cap at 500 chars).
- Or: concatenate all paragraphs before a section whose title contains "Introduction" / "1 " (respect section order).

---

## Medium priority

### 4. Fallback when reference parsing fails

**Current**: `parse_reference_text()` is heuristic; for non-English or non-standard formats (e.g. "Author. Title. Venue, Year") it may leave title/authors/venue empty.

**Suggestion**: When parsed `title` is empty, use **the first 400 chars of the full ref_text** for embedding (chunking already has a similar fallback); optionally add a `parsed: bool` flag in the schema for later manual or rule-based handling of "unparsed" entries.

---

### 5. Explicit citation constraint in generation prompt

**Current**: Context includes `[Reference [n]] Title. Authors. Venue. Year`, but the system instruction does not state "only cite from these references; do not invent references."

**Suggestion**: Add to the prompt in `answer_with_llm()`:  
`"Only cite from the references listed above (e.g. [1], [2]); do not invent references."`

---

### 6. Section title and number

**Current**: Section level comes from MinerU's `text_level`; the raw title may be "1 Introduction" or "Introduction", with no separate "number" field.

**Suggestion**: If "Section 2.1" style retrieval or display is needed, add `section_number` to the schema: extract a leading number with a regex (e.g. `^\s*(\d+(\.\d+)*)\s*`) for display and ordering (alongside existing `order`).

---

## Low priority / optional

### 7. Formula context description

**Current**: Formula chunks are LaTeX only; the semantic gap with natural-language queries is large; retrieval often relies on the surrounding paragraph.

**Suggestion**: Do not require separate embedding for formulas; if more readable context is desired, in `build_context_from_chunks` add a short line like "See Section X / formula above" for formula chunks when metadata has section_title or preceding_paragraph.

---

### 8. Multi-PDF / multi-document

**Current**: One graph per paper; state is per paper_id.

**Suggestion**: For multiple papers, keep separate schema/kg/embeddings per paper, or when merging graphs prefix node ids (e.g. `paper1::sec_1_...`) and filter by paper_id at retrieval and generation. Clarify product requirements first.

---

### 9. Configurable rerank weights

**Current**: α=0.6, β=0.3, γ=0.1 are hardcoded.

**Suggestion**: Already exposed via `query(..., alpha=, beta=, gamma=)`; add CLI flags `--alpha`, `--beta`, `--gamma` for tuning by query type.

---

### 10. In-document links in context

**Current**: URLs are graph nodes with edges to paragraph/figure/table, but the context string does not explicitly list "links appearing in the text."

**Suggestion**: If the LLM should cite code/project links, append a "Links in text: https://..." paragraph in `build_context_from_chunks` (from chunk metadata or KG), or inject only when the query mentions "link/code/GitHub".

---

## Already well covered

- Section order (order) and subsection hierarchy (level, parent, children)
- Abstract as a dedicated segment and chunk
- References: only title embedded; authors/venue/year in metadata
- URLs as graph nodes and mentions_link edges
- Retrieval results ordered by section_order + chunk_id before being sent to the LLM
- Visualization shows section order, level, and parent

---

## Suggested implementation order

1. **content_list_v2 flatten** (small change, immediate compatibility with more MinerU output)
2. **Table chunks** (complete retrieval and context)
3. **Abstract heuristic improvements**
4. Others as needed (ref fallback, prompt citation constraint, section_number, etc.)

For concrete code changes on a specific item, refer to the number or file name above.
