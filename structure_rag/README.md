# Structure RAG (Document-Structure-Aware Retrieval and QA)

Scripts and pipelines for processing academic papers from MinerU parse output, kept separate from core `mineru/` for maintenance and extension.

---

## 1. Required Input (MinerU Parse Output)

All tools depend on **MinerU PDF parse output**; the main input is **`*_content_list.json`**.

### 1.1 How to Obtain content_list.json

After parsing a PDF with MinerU, the output directory will contain (example):

```
<output_dir>/
└── <paper_name_or_pdf_stem>/
    └── hybrid_auto/   # or pipeline, vlm_transformers, etc.
        ├── <name>.md
        ├── <name>_content_list.json    ← required by this toolchain
        ├── <name>_middle.json
        ├── <name>_model.json
        └── images/
```

From the project root:

```bash
# Parse a single PDF, output to structure_rag/output
mineru -p path/to/paper.pdf -o structure_rag/output -b hybrid-auto-engine
# Yields structure_rag/output/paper/hybrid_auto/paper_content_list.json
```

Or using the demo:

```bash
python demo/demo.py   # output under demo/output/
```

### 1.2 content_list.json Format (Brief)

A JSON array produced by MinerU; each element is one content block. Common `type` values: `text` (titles/paragraphs), `image`, `table`, `equation`, `list` (e.g. references `ref_text`). See MinerU docs for field details.

---

## 2. Tools and I/O

### 2.1 GraphRAG Scientific Pipeline (Recommended: Single / Multi-Doc)

| Item | Description |
|------|-------------|
| **Entry** | `run_graphrag_scientific.py` |
| **Input** | Single `*_content_list.json` or existing index directory |
| **Output** | schema, kg, embeddings, HTML visualizations, QA under the given directory |
| **Deps** | `networkx`; optional `sentence-transformers`, `faiss-cpu`, `pyvis`, Ollama or OpenAI |

**Build index:**

```bash
# Run from project root
python structure_rag/run_graphrag_scientific.py build \
  structure_rag/output/2510.06592v1/hybrid_auto/2510.06592v1_content_list.json \
  -o structure_rag/output/2510.06592v1/graphrag_out
```

**Query / interactive:**

```bash
python structure_rag/run_graphrag_scientific.py query structure_rag/output/2510.06592v1/graphrag_out --query "What is stain normalization?"
python structure_rag/run_graphrag_scientific.py query structure_rag/output/2510.06592v1/graphrag_out -i
```

**Structure → Embedding → GraphRAG flow (implemented in this pipeline):**

1. **Schema and KG structure**: `schema` (from content_list via `mineru_to_schema`) and `build_heterogeneous_graph(schema)` yield graph G with node types Paper / Section / Paragraph / Figure / Table / Formula / Reference.
2. **Chunk by structure**: `graphrag_scientific/chunking.py`’s `build_multigranularity_chunks(schema)` produces multi-granularity chunks (abstract, section summary, paragraph, figure_caption, formula, reference), each with `section_id`, `chunk_id`, etc., aligned to graph nodes.
3. **Embedding**: `graphrag_scientific/embedding_store.py`’s `build_embedding_store(chunks, ...)` embeds these chunks (default `sentence-transformers`), writing `*_embeddings.json` (Faiss or in-memory index).
4. **GraphRAG retrieval**: On query, `retriever.hybrid_retrieve()` does semantic recall (embedding similarity) → BFS expansion from chunk nodes → rerank by `α*semantic + β*graph + γ*citation`, then passes context to the LLM.

These steps run once when building the index with `run_graphrag_scientific.py build` or `rebuild_graphrag_multi.py`; use the `query` subcommand to run GraphRAG.

See: [graphrag_scientific/README.md](graphrag_scientific/README.md)

---

### 2.2 Batch Parse + GraphRAG (Multi-PDF)

| Item | Description |
|------|-------------|
| **Entry** | `rebuild_graphrag_multi.py` |
| **Input** | `structure_rag/input/*.pdf` (to parse) + `structure_rag/output/**/hybrid_auto/*_content_list.json` (parsed) |
| **Output** | `structure_rag/output/graphrag_all/` (per-doc index + document relations + index.html) |
| **Deps** | MinerU (`demo.demo.parse_doc`), `structure_rag.graphrag_scientific` |

**Paths (editable at top of `rebuild_graphrag_multi.py`):**

- `INPUT_DIR = structure_rag/input`: place PDFs to parse
- `OUTPUT_DIR = structure_rag/output`: MinerU output and graphrag output
- `GRAPHAG_UNIFIED = structure_rag/output/graphrag_all`

**Usage:**

```bash
# Parse PDFs in structure_rag/input that lack content_list, build GraphRAG for all with content_list
python structure_rag/rebuild_graphrag_multi.py

# Build graph only for docs without schema; skip existing (recommended for incremental)
python structure_rag/rebuild_graphrag_multi.py --incremental

# Skip MinerU; build graphs from existing content_list only
python structure_rag/rebuild_graphrag_multi.py --no-parse
```

---

### 2.3 Legacy KG (build_knowledge_graph + graphrag_query)

For section-level KG with triples and section graph: use `build_knowledge_graph.py` to build and optionally `--visualize`; use `graphrag_query.py` to query the resulting index. See root README “structure_rag scripts” table for when to use which.

---

### 2.4 Citation Linking (Markdown Citations → Reference Hyperlinks)

| Item | Description |
|------|-------------|
| **Entry** | `link_citations.py` |
| **Input** | MinerU-generated `.md` with `# References` and in-text `[1]` etc. |
| **Output** | New Markdown with citations turned into hyperlinks to reference entries |

**Usage:**

```bash
python structure_rag/link_citations.py input.md
# Output default: input_linked.md

python structure_rag/link_citations.py input.md output.md
```

---

## 3. Dependencies Summary

| Tool | Required | Optional |
|------|----------|----------|
| **run_graphrag_scientific** | networkx | sentence-transformers, faiss-cpu, pyvis, Ollama / OpenAI |
| **rebuild_graphrag_multi** | MinerU + graphrag_scientific | same as above |
| **build_knowledge_graph** | networkx | pyvis / matplotlib, Ollama / OpenAI / llama-index |
| **graphrag_query** | none (keyword retrieval) | sentence-transformers, Ollama / OpenAI |
| **link_citations** | none | none |

Install (as needed):

```bash
pip install networkx pyvis
pip install sentence-transformers faiss-cpu   # vector retrieval
# Local LLM: install Ollama and ollama pull llama3.1:8b
```

---

## 4. Recommended Workflows

**Single paper:**

1. MinerU parse PDF → get `*_content_list.json`
2. Build GraphRAG index: `run_graphrag_scientific.py build ... -o <out>`
3. QA: `run_graphrag_scientific.py query <out> -i`
4. Optional: `link_citations.py paper.md paper_linked.md`

**Multiple papers (batch):**

1. Put PDFs in `structure_rag/input/`
2. Run `python structure_rag/rebuild_graphrag_multi.py --incremental`
3. Open `structure_rag/output/graphrag_all/index.html` for document relations, per-doc graphs, and QA

**Graph + keyword/vector retrieval + LLM QA only:**

1. `build_knowledge_graph.py <content_list.json> -o <dir> --visualize`
2. `graphrag_query.py <dir>/*_graphrag_index.json "question"` or `-i`

---

## 5. PDF → Text/Images Code Layout

“Read txt, images, etc. from PDF” is organized in two layers under structure_rag:

| Layer | Module | Role |
|-------|--------|------|
| **Parse entry** | `pdf_to_content.py` | Wraps MinerU: `parse_pdfs(pdf_paths, output_dir, ...)` to produce `content_list.json`, md, images under output_dir. Actual parsing is in mineru / demo.demo. |
| **Content load** | `content_loader.py` | Unified load: `load_content_list(path)` reads `*_content_list.json` (v1/v2), returns `list[dict]` (each item one block: section, paragraph, figure, table, formula, reference). Implementation in `graphrag_scientific.schema`, re-exported here. |

- Batch flow `rebuild_graphrag_multi.py` calls MinerU via `pdf_to_content.parse_pdfs()`, then scans `structure_rag/output/` for `*_content_list.json` to build graphs.
- `build_knowledge_graph.py`, `graphrag_scientific.pipeline`, etc. read content_list via `content_loader.load_content_list()` or `graphrag_scientific.schema.load_content_list()`.

Single-doc layout after parsing (MinerU output):

```
<output_dir>/<doc_name>/<backend>/
├── <doc_name>_content_list.json   # structured content list
├── <doc_name>.md
├── <doc_name>_middle.json
└── images/
```

---

## 6. Directory Layout

```
structure_rag/
├── README.md                    # this file
├── pdf_to_content.py            # PDF parse entry (wraps MinerU)
├── content_loader.py            # unified content_list.json loader
├── graphrag_scientific/        # scientific GraphRAG pipeline (schema/kg/chunk/retrieval/viz)
├── run_graphrag_scientific.py   # GraphRAG entry: build / query
├── input/                       # PDFs to parse (optional, see .gitignore)
├── output/                      # MinerU output + graphrag_all (optional, see .gitignore)
├── rebuild_graphrag_multi.py   # batch: input PDFs → MinerU → output/graphrag_all
├── build_knowledge_graph.py    # knowledge graph build + optional GraphRAG index
├── graphrag_query.py           # retrieval and QA from build_knowledge_graph index
└── link_citations.py           # Markdown citations → reference hyperlinks
```

Run all commands from the **project root** (or ensure project root is on `sys.path`).
