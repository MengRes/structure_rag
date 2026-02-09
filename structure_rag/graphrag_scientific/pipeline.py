"""
End-to-end pipeline: MinerU JSON -> Schema -> KG -> Chunks -> Embedding + Graph -> Hybrid retrieval on query -> LLM generation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional, Tuple

from .schema import load_content_list, load_schema, mineru_to_schema, save_schema
from .kg_builder import build_heterogeneous_graph, export_kg_json
from .chunking import build_multigranularity_chunks
from .embedding_store import EmbeddingStore, build_embedding_store, get_embedding_model
from .retriever import hybrid_retrieve, build_context_from_chunks
from .visualize import export_visualizations

try:
    import networkx as nx
except ImportError:
    nx = None


def run_pipeline(
    content_list_path: Path,
    output_dir: Path,
    paper_id: Optional[str] = None,
    embed_model: str = "all-MiniLM-L6-v2",
    use_faiss: bool = True,
    use_abstract_llm: bool = False,
    abstract_llm_model: str = "llama3.1:8b",
) -> dict[str, Any]:
    """
    Full flow: parse -> schema -> KG -> chunks -> embedding store; results written to output_dir.
    use_abstract_llm: when no Abstract section, use small LLM to detect abstract paragraphs.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paper_id = paper_id or Path(content_list_path).stem.replace("_content_list_v2", "").replace("_content_list", "")

    # 1) Schema
    content_list = load_content_list(content_list_path)
    schema = mineru_to_schema(
        content_list,
        paper_id=paper_id,
        use_llm_for_abstract=use_abstract_llm,
        abstract_llm_model=abstract_llm_model,
    )
    # 1.1) Merge same Figure subfigures into one (e.g. Fig. 1a/1b/1c -> single fig_1)
    try:
        from .figure_merge import merge_figure_groups
        content_base = Path(content_list_path).parent
        schema = merge_figure_groups(schema, content_base, output_dir, paper_id)
    except Exception:
        pass
    schema_path = output_dir / f"{paper_id}_schema.json"
    save_schema(schema, schema_path)

    # 2) KG
    G = build_heterogeneous_graph(schema)
    kg_path = output_dir / f"{paper_id}_kg.json"
    export_kg_json(G, kg_path)

    # 3) Chunks
    chunks = build_multigranularity_chunks(schema)

    # 4) Embedding store
    embedding_store = build_embedding_store(chunks, model_name=embed_model, use_faiss=use_faiss)
    emb_path = output_dir / f"{paper_id}_embeddings.json"
    embedding_store.save(emb_path)

    # 5) Visualize knowledge graph (PyVis, same style as build_knowledge_graph)
    viz_paths = export_visualizations(G, output_dir, paper_id, export_full_kg=True)
    if viz_paths:
        print("Exported visualizations: " + ", ".join(viz_paths[:3]) + (" ..." if len(viz_paths) > 3 else ""))
    # 6) Document-level index: index.html (Document relations or a PDF) + document_network.html
    try:
        from .visualize import export_document_index
        doc_index_paths = export_document_index(output_dir)
        if doc_index_paths:
            print("Exported document index: " + ", ".join(doc_index_paths))
    except Exception:
        pass

    return {
        "schema": schema,
        "G": G,
        "chunks": chunks,
        "embedding_store": embedding_store,
        "paths": {
            "schema": schema_path,
            "kg": kg_path,
            "embeddings": emb_path,
        },
        "paper_id": paper_id,
    }


def query(
    state: dict[str, Any],
    question: str,
    top_k_semantic: int = 15,
    top_k_final: int = 5,
    alpha: float = 0.6,
    beta: float = 0.3,
    gamma: float = 0.1,
    max_context_chars: int = 12000,
) -> Tuple[str, List[Tuple[dict, float]]]:
    """
    Hybrid GraphRAG retrieval from state; returns (context_string, ranked_chunks).
    """
    G = state["G"]
    embedding_store = state["embedding_store"]
    model = get_embedding_model()
    chunk_scores = hybrid_retrieve(
        question,
        embedding_store,
        G,
        top_k_semantic=top_k_semantic,
        top_k_final=top_k_final,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        model=model,
    )
    context = build_context_from_chunks(chunk_scores, max_chars=max_context_chars)
    return context, chunk_scores


def load_state(output_dir: Path, paper_id: str) -> dict[str, Any]:
    """Load state (schema + KG + embedding_store) from existing output_dir."""
    output_dir = Path(output_dir)
    schema_path = output_dir / f"{paper_id}_schema.json"
    kg_path = output_dir / f"{paper_id}_kg.json"
    emb_path = output_dir / f"{paper_id}_embeddings.json"
    if not schema_path.exists() or not kg_path.exists() or not emb_path.exists():
        raise FileNotFoundError(f"Missing index files in {output_dir} for {paper_id}")

    schema = load_schema(schema_path)
    G = nx.DiGraph()
    with open(kg_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for n in data.get("nodes", []):
        nid = n.pop("id", None)
        if nid is not None:
            G.add_node(nid, **n)
    for e in data.get("edges", []):
        u, v = e.get("source"), e.get("target")
        if u is not None and v is not None:
            G.add_edge(u, v, **{k: v for k, v in e.items() if k not in ("source", "target")})

    embedding_store = EmbeddingStore()
    embedding_store.load(emb_path)

    chunks = embedding_store.chunks
    return {
        "schema": schema,
        "G": G,
        "chunks": chunks,
        "embedding_store": embedding_store,
        "paths": {"schema": schema_path, "kg": kg_path, "embeddings": emb_path},
        "paper_id": paper_id,
    }


def answer_with_llm(
    question: str,
    context: str,
    use_ollama: bool = True,
    ollama_model: str = "llama3.1:8b",
) -> str:
    """Use LLM to answer question from context. Prefer Ollama downloaded models."""
    llm = None
    # 1) llama_index Ollama
    for mod in ("llama_index.llms.ollama", "llama_index.core.llms.ollama"):
        try:
            from importlib import import_module
            ollama_mod = import_module(mod)
            Ollama = getattr(ollama_mod, "Ollama")
            llm = Ollama(model=ollama_model, request_timeout=120.0)
            break
        except Exception:
            continue
    if llm is None:
        try:
            from llama_index.core import Settings
            llm = getattr(Settings, "llm", None)
        except Exception:
            pass
    # 2) Use ollama Python library directly (pip install ollama)
    if llm is None and use_ollama:
        try:
            import ollama
            response = ollama.chat(
                model=ollama_model,
                messages=[
                    {
                        "role": "user",
                        "content": f"""Based on the following context from a scientific document (sections, paragraphs, figures), answer the question concisely. If the context does not contain enough information, say so.

Context:
{context[:12000]}

Question: {question}

Answer:""",
                    }
                ],
            )
            return (response.get("message") or {}).get("content", "") or f"[Ollama no reply]\n\nContext:\n{context[:2000]}"
        except Exception as e:
            return f"[Ollama call failed: {e}]\n\nConfirm: 1) ollama serve is running  2) model is pulled, e.g. ollama pull {ollama_model}\n\nContext:\n{context[:2000]}"

    if llm is None:
        return f"[LLM not configured]\n\nInstall: pip install ollama or pip install llama-index-llms-ollama\n\nContext:\n{context[:2000]}"

    prompt = f"""Based on the following context from a scientific document (sections, paragraphs, figures), answer the question concisely. If the context does not contain enough information, say so.

Context:
{context[:12000]}

Question: {question}

Answer:"""
    try:
        r = llm.complete(prompt)
        return r.text if hasattr(r, "text") else str(r)
    except Exception as e:
        return f"[LLM error: {e}]\n\n{context[:1500]}"


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Document-Structure-Aware GraphRAG pipeline")
    p.add_argument("input", type=Path, help="content_list.json or existing indexed output_dir")
    p.add_argument("-o", "--output-dir", type=Path, default=None, help="Output directory (default: same as input or input itself)")
    p.add_argument("--paper-id", type=str, default=None, help="paper_id")
    p.add_argument("--embed-model", type=str, default="all-MiniLM-L6-v2", help="sentence-transformers model")
    p.add_argument("--no-faiss", action="store_true", help="Do not use Faiss; use in-memory index")
    p.add_argument("--query", type=str, default="", help="Ask a question directly (index must exist)")
    p.add_argument("--interactive", "-i", action="store_true", help="Interactive Q&A")
    p.add_argument("--ollama-model", type=str, default="llama3.1:8b", help="Ollama model name (same as ollama list, e.g. llama3.1:8b, qwen2.5:14b)")
    p.add_argument("--max-context", type=int, default=12000, help="Max context chars sent to LLM (default 12000)")
    p.add_argument("--abstract-llm", action="store_true", help="When no Abstract section, use small LLM to detect abstract (requires ollama)")
    p.add_argument("--abstract-llm-model", type=str, default="llama3.1:8b", help="Small model for abstract detection (default llama3.1:8b)")
    args = p.parse_args()

    if args.query or args.interactive:
        # Query mode: input = directory with index (or content_list directory)
        inp = Path(args.input)
        out = Path(args.output_dir) if args.output_dir else (inp if inp.is_dir() else inp.parent)
        paper_id = args.paper_id
        if not paper_id:
            for f in out.glob("*_schema.json"):
                paper_id = f.stem.replace("_schema", "")
                break
        if not paper_id:
            paper_id = out.stem
        if not (out / f"{paper_id}_schema.json").exists():
            print("Index not found; run pipeline to build index first. Example: python -m graphrag_scientific.pipeline <content_list.json> -o <out_dir>")
            return
        state = load_state(out, paper_id)
        if args.query:
            ctx, chunks = query(state, args.query, max_context_chars=args.max_context)
            print(answer_with_llm(args.query, ctx, ollama_model=args.ollama_model))
            return
        if args.interactive:
            print("GraphRAG interactive Q&A (empty line or quit to exit)")
            while True:
                try:
                    q = input("\nQuestion> ").strip()
                except EOFError:
                    break
                if not q or q.lower() in ("quit", "exit", "q"):
                    break
                ctx, _ = query(state, q, max_context_chars=args.max_context)
                print(answer_with_llm(q, ctx, ollama_model=args.ollama_model))
        return

    # Build index
    inp = Path(args.input)
    if not inp.exists():
        print(f"Not found: {inp}")
        return
    out = args.output_dir or inp.parent
    if inp.is_dir():
        print("input must be a content_list.json file, not a directory.")
        return
    state = run_pipeline(
        inp,
        out,
        paper_id=args.paper_id,
        embed_model=args.embed_model,
        use_faiss=not args.no_faiss,
        use_abstract_llm=args.abstract_llm,
        abstract_llm_model=args.abstract_llm_model,
    )
    print(f"Written: {state['paths']}")
    print("Use --query or --interactive to ask questions.")


if __name__ == "__main__":
    main()
