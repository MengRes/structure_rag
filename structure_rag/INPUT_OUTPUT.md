# structure_rag I/O Quick Reference

## Prerequisite: MinerU Parse Output

- **Required file**: `*_content_list.json` (produced by MinerU from PDFs; under `<out>/<doc>/hybrid_auto/` or similar backend).
- **How to obtain**:
  - From structure_rag: `structure_rag.pdf_to_content.parse_pdfs(pdf_paths, output_dir)` (wraps MinerU);
  - Or CLI: `mineru -p <pdf> -o <out> -b hybrid-auto-engine` / `python demo/demo.py`.
- **Load content_list**: `structure_rag.content_loader.load_content_list(path)` (supports v1/v2).

---

## Tool Inputs/Outputs

| Tool | Input | Output |
|------|-------|--------|
| **run_graphrag_scientific.py build** | Single `*_content_list.json` | `-o` dir: `*_schema.json`, `*_kg.json`, `*_embeddings.json`, `*_knowledge_graph.html`, etc. |
| **run_graphrag_scientific.py query** | Index dir (same as `-o` above) | Terminal QA; interactive Q&A |
| **rebuild_graphrag_multi.py** | `structure_rag/input/*.pdf` + `structure_rag/output/**/*_content_list.json` | `structure_rag/output/graphrag_all/`: per-doc index + `index.html` + `document_network.html` |
| **build_knowledge_graph.py** | Single `*_content_list.json` or existing `*_kg.json` | `-o` dir: `*_kg.json`, `*_graphrag_index.json`; with `--visualize`: HTML/PNG |
| **graphrag_query.py** | `*_graphrag_index.json` (from build_knowledge_graph) | Terminal retrieval and QA |
| **link_citations.py** | MinerU-generated `.md` (References + `[1]` etc.) | New `.md` (default `*_linked.md`), citations as hyperlinks |

---

## Default Paths (rebuild_graphrag_multi)

- **INPUT_DIR**: `structure_rag/input/` (PDFs to parse)
- **OUTPUT_DIR**: `structure_rag/output/` (MinerU output + graphrag output)
- **GRAPHAG_UNIFIED**: `structure_rag/output/graphrag_all/`

To change: edit `INPUT_DIR` / `OUTPUT_DIR` / `GRAPHAG_UNIFIED` at the top of `structure_rag/rebuild_graphrag_multi.py`.
