# Knowledge Graph Visualization

After running the pipeline to build the index, the following files are generated under `output_dir`. **Start from `index.html`.**

## Entry Page (recommended)

| File | Description |
|------|-------------|
| **index.html** | **Unified entry**. When you select "Document relations" in the dropdown, only PDF-to-PDF links are shown (nodes = documents). When you select a PDF, that document's knowledge graph is shown (iframe loads the corresponding `{paper_id}_knowledge_graph.html`). |
| **document_network.html** | **Document-level relation graph**: nodes are PDFs (paper_id), edges come from `document_relations.json`. When multiple documents share the same output_dir, use this to view relations between documents. |

### Document Relations (optional)

**Auto-matching**: When you run `structure_rag/rebuild_graphrag_multi.py`, document paper titles are matched against reference list titles; for each document the script generates/merges `{paper_id}_reference_links.json` and then aggregates `document_relations.json`. Matching rules: exact match after normalization, or one string contains the other, or word overlap ratio ≥ 0.6.

**Manual config**: Place **document_relations.json** under `output_dir` to specify PDF-to-PDF relations shown in the "Document relations" view. Example format:

```json
[
  {"source": "paper1", "target": "paper2", "relation": "cites"},
  {"source": "paper1", "target": "paper3", "relation": "cites"}
]
```

If not configured, the document graph will have only nodes and no edges (or a placeholder message).

### Reference-to-Local-Document Mapping (optional)

Place **`{paper_id}_reference_links.json`** under `output_dir` for **the current document** to indicate which cited reference corresponds to which PDF in this folder:

- **References that exist in this corpus**: shown in **green**; **double-click the node** to open that document's knowledge graph in a new tab.
- **Unconfigured or external references**: shown in **grey** (external).

Example format (keys are ref node ids, values are paper_id in this folder):

```json
{
  "ref_1": "2510.06592v1",
  "ref_3": "another_paper_id"
}
```

Meaning: ref_1 and ref_3 in this PDF map to documents `2510.06592v1` and `another_paper_id` in this folder.

---

## Single-Document Views

The same document's knowledge graph has multiple views (all HTML):

| File | Description |
|------|-------------|
| `{paper_id}_knowledge_graph.html` | **Full heterogeneous graph**: Paper, Section, Paragraph, Figure, Table, Formula, Reference, URL nodes and all edges. |
| `{paper_id}_citation_network.html` | **Citation subgraph**: Sections, References, and cites edges. |
| `{paper_id}_section_graph.html` | **Section structure subgraph**: Paper → Section hierarchy and subsections. |
| `{paper_id}_query_subgraph.html` | **Query subgraph** (exported on demand): subgraph highlighted for a given retrieval. |

Underlying graph data: `{paper_id}_kg.json`. The HTML files above are different filters/views of the same graph.
