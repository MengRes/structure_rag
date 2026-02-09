#!/bin/bash
# Push structure_rag to repo with ONE commit PER logical group (by function).
# Run from MinerU project root. Remote: git@github.com:MengRes/structure_rag.git

set -e
REPO_DIR="/home/mengliang/cihi_lab/structure_rag_repo"
SRC_DIR="/home/mengliang/cihi_lab/MinerU/structure_rag"
R="structure_rag"  # subfolder name in repo

echo "Creating repo at $REPO_DIR with $R/ subfolder..."
rm -rf "$REPO_DIR"
mkdir -p "$REPO_DIR/$R"
cp -r "$SRC_DIR"/* "$REPO_DIR/$R/"
cp "$SRC_DIR"/.gitignore "$REPO_DIR/$R/" 2>/dev/null || true
if [ -f "$SRC_DIR/../REPO_ROOT_README.md" ]; then
  cp "$SRC_DIR/../REPO_ROOT_README.md" "$REPO_DIR/README.md"
else
  echo "# structure_rag

Document-structure RAG and GraphRAG for scientific PDFs (MinerU).

Code is under [structure_rag/](structure_rag/). See [structure_rag/README.md](structure_rag/README.md).
" > "$REPO_DIR/README.md"
fi

cd "$REPO_DIR"
git init
git branch -M main

# 1) Docs (root README, structure_rag README/INPUT_OUTPUT, example KG image)
git add README.md "$R/README.md" "$R/INPUT_OUTPUT.md" "$R/docs/"
git commit -m "docs: add README and I/O overview, example KG image"

# 2) Chore: ignore rules
git add "$R/.gitignore"
git commit -m "chore: add .gitignore for input, output, __pycache__"

# 3) PDF parse + content load entry
git add "$R/__init__.py" "$R/pdf_to_content.py" "$R/content_loader.py"
git commit -m "feat(parse): add PDF parse entry and content_list loader"

# 4) Schema (content_list -> unified schema)
git add "$R/graphrag_scientific/schema.py"
git commit -m "feat(schema): add unified schema and mineru_to_schema"

# 5) Knowledge graph build from schema
git add "$R/graphrag_scientific/kg_builder.py"
git commit -m "feat(kg): add heterogeneous KG builder from schema"

# 6) Chunking
git add "$R/graphrag_scientific/chunking.py"
git commit -m "feat(chunk): add multi-granularity chunking from schema"

# 7) Embedding store
git add "$R/graphrag_scientific/embedding_store.py"
git commit -m "feat(embed): add embedding store for chunks"

# 8) Retriever (hybrid GraphRAG)
git add "$R/graphrag_scientific/retriever.py"
git commit -m "feat(retrieve): add hybrid GraphRAG retriever"

# 9) Visualize + reference/figure helpers
git add "$R/graphrag_scientific/visualize.py" "$R/graphrag_scientific/reference_matching.py" "$R/graphrag_scientific/figure_merge.py"
git commit -m "feat(viz): add KG visualization and reference/figure helpers"

# 10) Pipeline + package docs and test
git add "$R/graphrag_scientific/pipeline.py" "$R/graphrag_scientific/__init__.py" \
  "$R/graphrag_scientific/README.md" "$R/graphrag_scientific/IMPROVEMENTS.md" \
  "$R/graphrag_scientific/VIZ_README.md" "$R/graphrag_scientific/test_author_year_citation.py"
git commit -m "feat(pipeline): add end-to-end pipeline and package docs"

# 11) Legacy KG build (triples + section graph)
git add "$R/build_knowledge_graph.py"
git commit -m "feat(kg-legacy): add build_knowledge_graph (triples + section graph)"

# 12) GraphRAG query (section-level index)
git add "$R/graphrag_query.py"
git commit -m "feat(query): add graphrag_query for section-level index"

# 13) Batch rebuild + run entry
git add "$R/rebuild_graphrag_multi.py" "$R/run_graphrag_scientific.py"
git commit -m "feat(batch): add rebuild_graphrag_multi and run_graphrag_scientific"

# 14) Citation linking
git add "$R/link_citations.py"
git commit -m "feat(citation): add link_citations for Markdown references"

# 15) Push script (optional to track)
git add "$R/push_clean_repo.sh"
git commit -m "chore: add push_clean_repo.sh for repo sync"

git remote add origin git@github.com:MengRes/structure_rag.git
echo "Pushing to origin main (multiple commits)..."
git push -u origin main

echo "Done. Repo at $REPO_DIR."
