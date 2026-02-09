# Document-Structure-Aware GraphRAG for Scientific PDFs

Structured graph RAG for academic papers: MinerU parsing → unified schema → heterogeneous graph → multi-granularity chunks → vector + graph dual index → hybrid retrieval → LLM generation → visualization.

## Architecture

```
PDFs → MinerU structured parsing → content_list.json
         ↓
    Unified schema (sections / paragraphs / figures / tables / formulas / citations)
         ↓
    Heterogeneous graph (Paper, Section, Paragraph, Figure, Table, Formula, Reference + structure/citation edges)
         ↓
    Multi-granularity chunks (section / paragraph / figure_caption / formula)
         ↓
    Embedding store (Faiss / in-memory)  +  Graph (NetworkX)
         ↓
    Hybrid retriever: semantic recall → graph expansion → rerank (α·sim + β·graph + γ·citation)
         ↓
    LLM generation  +  visualization (citation / section / query subgraph)
```

## Dependencies

- Python 3.10+
- `networkx` (graph)
- `sentence-transformers` (embedding, optional; falls back to keyword search if missing)
- `faiss-cpu` (optional; in-memory index if missing)
- `pyvis` (visualization)
- `llama-index-llms-ollama` or OpenAI (generation)

## Usage

### 1. Build index (from MinerU content_list.json)

```bash
# Output defaults to <content_list directory>/graphrag_out
python structure_rag/run_graphrag_scientific.py build structure_rag/output/2510.06592v1/hybrid_auto/2510.06592v1_content_list.json

# Specify output directory
python structure_rag/run_graphrag_scientific.py build structure_rag/output/.../2510.06592v1_content_list.json -o structure_rag/output/2510.06592v1/graphrag_out
```

This generates:

- `{paper_id}_schema.json`: unified structure
- `{paper_id}_kg.json`: heterogeneous graph
- `{paper_id}_embeddings.json` (and `.faiss`): multi-granularity chunk vectors

### 2. Query

```bash
# Single query (index must already be built)
python structure_rag/run_graphrag_scientific.py query structure_rag/output/2510.06592v1/graphrag_out --query "What is stain normalization?"

# Interactive
python structure_rag/run_graphrag_scientific.py query structure_rag/output/2510.06592v1/graphrag_out -i
```

Or call the submodule directly:

```bash
python -m structure_rag.graphrag_scientific.pipeline <content_list.json> -o <out_dir>           # build index
python -m structure_rag.graphrag_scientific.pipeline <out_dir> --query "your question"          # query
python -m structure_rag.graphrag_scientific.pipeline <out_dir> --interactive
```

### 3. Visualization

After building the index with the pipeline, use `graphrag_scientific.visualize` to export:

- Citation network (Section → Reference)
- Section graph (Paper → Section hierarchy)
- Query subgraph (retrieval path highlighted)

```python
from pathlib import Path
from structure_rag.graphrag_scientific.pipeline import load_state
from structure_rag.graphrag_scientific.visualize import export_visualizations

state = load_state(Path("structure_rag/output/2510.06592v1/graphrag_out"), "2510.06592v1")
export_visualizations(state["G"], Path("out"), state["paper_id"])
```

## Modules

| Module | Role |
|--------|------|
| `schema` | MinerU content_list → unified JSON schema (sections/paragraphs/figures/tables/formulas/citations) |
| `kg_builder` | Heterogeneous graph: Paper, Section, Paragraph, Figure, Table, Formula, Reference; edges has_section, has_paragraph, cites, etc. |
| `chunking` | Multi-granularity chunks: section summary, paragraph, figure_caption, formula |
| `embedding_store` | Chunk embedding + Faiss / in-memory retrieval |
| `retriever` | Hybrid: semantic top-k → graph expansion → rerank |
| `pipeline` | End-to-end index build + load + query + LLM generation |
| `visualize` | Citation / Section / Query subgraph HTML |

## Retrieval formula

`score = α * embedding_sim + β * graph_score + γ * citation_weight`

Default α=0.6, β=0.3, γ=0.1; adjustable via `query()`.
