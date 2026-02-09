# structure_rag

This repository bundles **MinerU** (PDF → structured content) and **structure_rag** (document-structure RAG and GraphRAG) for parsing scientific PDFs and running retrieval/QA over them.

---

## What's in this repo

| Path | Description |
|------|-------------|
| **mineru/** | PDF parsing core: layout detection, OCR, table/formula recognition, VLM/hybrid backends. Produces `*_content_list.json`, markdown, and images. |
| **structure_rag/** | Document-structure RAG: builds a knowledge graph from content_list, chunks by structure, embeds, and runs hybrid GraphRAG retrieval + optional LLM QA. |

**Typical flow:** Parse PDFs with MinerU → get `content_list.json` → build schema + KG + chunks + embeddings with structure_rag → query with GraphRAG or run batch indexing for multiple papers.

---

## Project structure

```
.
├── README.md              # this file
├── mineru/                # PDF parsing
│   ├── backend/           # hybrid, pipeline, vlm
│   ├── cli/               # CLI, FastAPI, Gradio, mineru-models-download
│   ├── data/              # reader/writer, io
│   ├── model/             # layout, mfd, table, vlm, pytorchocr
│   ├── resources/         # fasttext, header
│   └── utils/             # pdf, ocr, block, config
└── structure_rag/         # RAG & GraphRAG
    ├── docs/              # example KG image (kg_example.png)
    ├── scripts/           # e.g. gen_kg_example.py
    ├── pdf_to_content.py  # PDF parse entry (calls mineru)
    ├── content_loader.py  # load content_list
    ├── graphrag_scientific/  # schema, kg, chunk, embed, retrieve, visualize
    ├── run_graphrag_scientific.py  # build / query
    ├── rebuild_graphrag_multi.py   # batch: input PDFs → graphrag_all
    ├── build_knowledge_graph.py   # legacy KG + triples
    ├── graphrag_query.py  # query legacy index
    └── link_citations.py # Markdown citation links
```

---

## Requirements

- **Python**: 3.10–3.13 (see `pyproject.toml`).
- **OS**: Linux, macOS, Windows.
- GPU (CUDA) recommended for VLM/hybrid backends; extra deps for sentence-transformers, faiss, pyvis, Ollama/OpenAI for RAG and visualization.

---

## Configuration and setup

### 1. Clone and install

```bash
git clone git@github.com:MengRes/structure_rag.git
cd structure_rag
```

Install the project in editable mode (from repo root):

```bash
pip install -e .
```

Or install only core MinerU (no optional RAG/visualization deps):

```bash
pip install -e ".[core]"
```

Full install including optional RAG and GPU-related extras (see `pyproject.toml` for `[project.optional-dependencies]`):

```bash
pip install -e ".[all]"
```

### 2. MinerU backends (for PDF parsing)

- **hybrid-auto-engine** (default): best quality; uses VLM when available. Needs GPU for reasonable speed.
- **pipeline**: CPU-friendly, no VLM.
- **vlm-transformers / vlm-vllm**: VLM-based; configure via env or MinerU docs.

Set backend when running the parser (see “Run: Parse PDF” below). No extra config file is required for basic use.

### 3. structure_rag (GraphRAG)

- **Input/output paths**: Edit the top of `structure_rag/rebuild_graphrag_multi.py` to set `INPUT_DIR`, `OUTPUT_DIR`, `GRAPHAG_UNIFIED` (defaults under `structure_rag/input`, `structure_rag/output`, `structure_rag/output/graphrag_all`).
- **Embedding model**: Default `all-MiniLM-L6-v2`. Override with `--embed-model` where supported (e.g. `run_graphrag_scientific.py`).
- **LLM for QA**: Ollama (e.g. `ollama pull llama3.1:8b`) or OpenAI; set `OPENAI_API_KEY` if using OpenAI. Used by `run_graphrag_scientific.py query` and `graphrag_query.py`.

### 4. Environment variables

- `OPENAI_API_KEY`: for OpenAI-based LLM in RAG/QA.
- MinerU/VLM: see MinerU docs for any backend-specific env (e.g. CUDA_VISIBLE_DEVICES, vLLM config).

### 5. Download MinerU models

MinerU needs model files for layout detection, OCR, table/formula recognition, and (optionally) VLM. Download them **before** parsing PDFs.

From **Hugging Face** or **ModelScope**:

```bash
mineru-models-download -s huggingface -m all
```

- **`-s`** (source): `huggingface` or `modelscope`. Use `modelscope` if Hugging Face is slow or blocked.
- **`-m`** (model type):
  - **`pipeline`**: layout, MFD, MFR, OCR, table, reading order (required for `pipeline` and `hybrid-auto-engine` backends).
  - **`vlm`**: VLM model used by `vlm-transformers` / `vlm-vllm` and by hybrid when using VLM.
  - **`all`**: both pipeline and VLM (default if you omit `-m`).

Examples:

```bash
# Pipeline only (enough for pipeline backend; hybrid will use pipeline part)
mineru-models-download -s huggingface -m pipeline

# VLM only (if you already have pipeline models)
mineru-models-download -s modelscope -m vlm

# All models from ModelScope
mineru-models-download -s modelscope -m all
```

Models are cached under the Hugging Face / ModelScope cache dir. You can set `MINERU_MODEL_SOURCE=modelscope` or `huggingface` (default) before running the parser if you use both and want to prefer one.

---

## structure_rag scripts: what to run when

| Script | Use when |
|--------|----------|
| **`run_graphrag_scientific.py`** | Build GraphRAG index from one `*_content_list.json`, or query a built index (single-doc RAG/QA). Main entry for schema → KG → chunks → embeddings → hybrid retrieval + LLM. |
| **`rebuild_graphrag_multi.py`** | Batch: parse PDFs in `structure_rag/input/`, build GraphRAG for all docs, write to `graphrag_all/`. Use `--no-parse` to only rebuild from existing content_list. |
| **`pdf_to_content.py`** | Programmatic PDF → content_list (calls MinerU). Used by `rebuild_graphrag_multi.py`; you can also call `mineru` CLI directly. |
| **`content_loader.py`** | Load and normalize `*_content_list.json`. Used internally by graphrag_scientific and other scripts. |
| **`link_citations.py`** | Add hyperlinks from in-text citations to the reference list in a Markdown file. |
| **`build_knowledge_graph.py`** | Legacy: build section-level KG (triples + section graph) and visualize. |
| **`graphrag_query.py`** | Query the legacy KG index produced by `build_knowledge_graph.py`. |

The **`graphrag_scientific/`** package contains schema, KG build, chunking, embedding, and retrieval; you use it via `run_graphrag_scientific.py` and `rebuild_graphrag_multi.py`. For more detail and I/O paths see [structure_rag/README.md](structure_rag/README.md) and [structure_rag/INPUT_OUTPUT.md](structure_rag/INPUT_OUTPUT.md).

### Example: generated knowledge graph

After building a GraphRAG index, you get a heterogeneous graph: sections (green), paragraphs (teal), figures/tables/equations (orange), references (gray), and citation/section links. Below is an example from a real run (e.g. a paper on stain normalization); interactive HTML is also produced (`*_knowledge_graph.html`, `index.html` in `structure_rag/output/graphrag_all/`).

![Example knowledge graph](https://github.com/MengRes/structure_rag/raw/main/structure_rag/docs/kg_example.png)

---

## How to run

All commands below are from the **repository root** (the directory that contains `mineru/` and `structure_rag/`).

### 1. Parse PDFs (MinerU)

Single PDF, output under `structure_rag/output`:

```bash
mineru -p path/to/paper.pdf -o structure_rag/output -b hybrid-auto-engine
```

Or use the demo script (output under `demo/output`):

```bash
python demo/demo.py
```

Result: under `<output_dir>/<doc_name>/hybrid_auto/` you get `*_content_list.json`, `*.md`, and `images/`. structure_rag uses `*_content_list.json`.

### 2. Build GraphRAG index (single document)

```bash
python structure_rag/run_graphrag_scientific.py build \
  structure_rag/output/<doc_name>/hybrid_auto/<doc_name>_content_list.json \
  -o structure_rag/output/<doc_name>/graphrag_out
```

This creates schema, KG, chunks, embeddings, and HTML under the given `-o` directory.

### 3. Query (single-doc GraphRAG)

One-off question:

```bash
python structure_rag/run_graphrag_scientific.py query structure_rag/output/<doc_name>/graphrag_out --query "Your question?"
```

Interactive QA:

```bash
python structure_rag/run_graphrag_scientific.py query structure_rag/output/<doc_name>/graphrag_out -i
```

### 4. Batch: multiple PDFs → GraphRAG

1. Put PDFs in `structure_rag/input/`.
2. Run:

```bash
python structure_rag/rebuild_graphrag_multi.py
```

This parses new PDFs (if any), builds graphs for all docs with `content_list`, and writes to `structure_rag/output/graphrag_all/`. Useful flags: `--no-parse` (skip parsing; rebuild from existing content_list only), `--incremental` (only build for docs without an index).

```bash
python structure_rag/rebuild_graphrag_multi.py --no-parse   # no MinerU, graph only
```

3. Open `structure_rag/output/graphrag_all/index.html` for document relations and per-doc knowledge graphs.

### 5. Citation linking (Markdown)

Turn in-text citations into hyperlinks to the reference list:

```bash
python structure_rag/link_citations.py path/to/paper.md
# Output: path/to/paper_linked.md
```

---

## Docs and references

- **structure_rag**: [structure_rag/README.md](structure_rag/README.md), [structure_rag/INPUT_OUTPUT.md](structure_rag/INPUT_OUTPUT.md), [structure_rag/graphrag_scientific/VIZ_README.md](structure_rag/graphrag_scientific/VIZ_README.md) (visualization).
- **MinerU**: upstream [MinerU](https://github.com/opendatalab/MinerU) and its docs for backends, CLI options, and troubleshooting.

---

## License

See project license (e.g. AGPL-3.0 for MinerU). This repo may aggregate code from different sources; respect each component’s license.
