#!/usr/bin/env python3
"""
Section-level GraphRAG query: retrieval + QA from knowledge-graph index.

Usage:
  python graphrag_query.py <graphrag_index.json> "your question"
  python graphrag_query.py <graphrag_index.json> --interactive   # interactive

Dependencies: same as build_knowledge_graph; retrieval can use keyword or embedding (pip install sentence-transformers optional).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Optional

# Optional embedding
_HAS_EMBED = False
try:
    from sentence_transformers import SentenceTransformer
    _HAS_EMBED = True
except ImportError:
    pass

# Optional LlamaIndex
_HAS_LLAMA = False
try:
    from llama_index.core import Settings
    _HAS_LLAMA = True
except ImportError:
    pass


def load_graphrag_index(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize(s: str) -> str:
    return " ".join(s.split()) if s else ""


def _section_text_for_retrieval(sec: dict[str, Any]) -> str:
    """Flatten section to one text for retrieval (title + chunk previews + triples)."""
    parts = [sec.get("title", "")]
    for c in sec.get("chunks", [])[:5]:
        parts.append(_normalize(c.get("text", ""))[:400])
    for t in sec.get("triples", [])[:20]:
        s, r, o = t.get("subject", ""), t.get("relation", ""), t.get("object", "")
        parts.append(f" {s} {r} {o} ")
    return _normalize(" ".join(parts))


def _chunk_text(c: dict[str, Any]) -> str:
    return _normalize(c.get("text", ""))


def retrieve_by_keyword(
    index: dict[str, Any],
    query: str,
    top_k_sections: int = 3,
    top_k_chunks_per_section: int = 3,
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    """
    Keyword retrieval: rank by how many query terms appear in section/chunk.
    Returns (sections list, [(section_id, chunk_text), ...]).
    """
    query_n = _normalize(query).lower()
    words = set(re.findall(r"\w+", query_n))
    if not words:
        return [], []

    sections = index.get("sections", [])
    scored: list[tuple[float, dict[str, Any]]] = []
    for sec in sections:
        text = _section_text_for_retrieval(sec).lower()
        score = sum(1 for w in words if w in text)
        if score > 0:
            scored.append((score, sec))
    scored.sort(key=lambda x: -x[0])
    top_sections = [s for _, s in scored[:top_k_sections]]

    # Per section, take top_k_chunks_per_section most relevant chunks
    out_chunks: list[tuple[str, str]] = []
    for sec in top_sections:
        sid = sec.get("section_id", "")
        title = sec.get("title", "")
        chunks = sec.get("chunks", [])
        chunk_scores: list[tuple[float, str]] = []
        for c in chunks:
            t = _chunk_text(c).lower()
            sc = sum(1 for w in words if w in t)
            chunk_scores.append((sc, _chunk_text(c)))
        chunk_scores.sort(key=lambda x: -x[0])
        for _, ct in chunk_scores[:top_k_chunks_per_section]:
            if ct:
                out_chunks.append((f"{title} ({sid})", ct))
        # Append triple summary for this section
        triples = sec.get("triples", [])[:10]
        if triples:
            triple_str = " ; ".join(
                f"{t.get('subject', '')} {t.get('relation', '')} {t.get('object', '')}"
                for t in triples
            )
            out_chunks.append((f"{title} [triples]", triple_str))

    return top_sections, out_chunks


def retrieve_by_embedding(
    index: dict[str, Any],
    query: str,
    model: Any,
    top_k_sections: int = 3,
    top_k_chunks_per_section: int = 3,
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    """Retrieve by embedding similarity."""
    if not _HAS_EMBED or model is None:
        return retrieve_by_keyword(index, query, top_k_sections, top_k_chunks_per_section)
    q_emb = model.encode([query], normalize_embeddings=True)[0]
    sections = index.get("sections", [])
    scored_sec: list[tuple[float, dict[str, Any]]] = []
    for sec in sections:
        text = _section_text_for_retrieval(sec)
        if not text:
            continue
        emb = model.encode([text[:4000]], normalize_embeddings=True)[0]
        sim = float(q_emb @ emb)
        scored_sec.append((sim, sec))
    scored_sec.sort(key=lambda x: -x[0])
    top_sections = [s for _, s in scored_sec[:top_k_sections]]

    out_chunks: list[tuple[str, str]] = []
    for sec in top_sections:
        sid = sec.get("section_id", "")
        title = sec.get("title", "")
        chunks = sec.get("chunks", [])
        if not chunks:
            continue
        texts = [_chunk_text(c) for c in chunks]
        embs = model.encode(texts, normalize_embeddings=True)
        sims = [float(q_emb @ e) for e in embs]
        indexed = list(zip(sims, texts))
        indexed.sort(key=lambda x: -x[0])
        for _, ct in indexed[:top_k_chunks_per_section]:
            if ct:
                out_chunks.append((f"{title} ({sid})", ct))
        triples = sec.get("triples", [])[:10]
        if triples:
            triple_str = " ; ".join(
                f"{t.get('subject', '')} {t.get('relation', '')} {t.get('object', '')}"
                for t in triples
            )
            out_chunks.append((f"{title} [triples]", triple_str))

    return top_sections, out_chunks


def build_context(chunks: list[tuple[str, str]], max_chars: int = 4000) -> str:
    parts = []
    n = 0
    for loc, text in chunks:
        block = f"[{loc}]\n{text[:1500]}"
        if n + len(block) > max_chars:
            break
        parts.append(block)
        n += len(block)
    return "\n\n---\n\n".join(parts)


def answer_with_llm(
    query: str,
    context: str,
    *,
    llm: Any = None,
    use_ollama: bool = False,
    ollama_model: str = "llama3.2",
) -> str:
    """Use LLM to answer query from retrieved context."""
    if _HAS_LLAMA and llm is None:
        if use_ollama:
            try:
                from llama_index.llms.ollama import Ollama
                llm = Ollama(model=ollama_model, request_timeout=120.0)
            except Exception:
                llm = getattr(Settings, "llm", None)
        else:
            llm = getattr(Settings, "llm", None)
    if llm is None and _HAS_LLAMA:
        try:
            from llama_index.llms.ollama import Ollama
            llm = Ollama(model=ollama_model, request_timeout=120.0)
        except Exception:
            pass
    if llm is None:
        return "[LLM not configured; returning retrieved snippets only]\n\n" + context[:2000]

    prompt = f"""Based on the following context extracted from a document knowledge graph, answer the question concisely. If the context does not contain enough information, say so.

Context:
{context[:3500]}

Question: {query}

Answer:"""
    try:
        if hasattr(llm, "complete"):
            r = llm.complete(prompt)
            return r.text if hasattr(r, "text") else str(r)
        return str(llm(prompt))
    except Exception as e:
        return f"[LLM call failed: {e}]\n\nRetrieved context:\n{context[:1500]}"


def query(
    index_path: Path,
    question: str,
    *,
    use_embedding: bool = False,
    embed_model_name: str = "all-MiniLM-L6-v2",
    top_k_sections: int = 3,
    top_k_chunks: int = 3,
    use_ollama: bool = True,
    ollama_model: str = "llama3.2",
) -> str:
    index = load_graphrag_index(index_path)
    model = None
    if use_embedding and _HAS_EMBED:
        model = SentenceTransformer(embed_model_name)

    if model is not None:
        sections, chunks = retrieve_by_embedding(
            index, question, model,
            top_k_sections=top_k_sections,
            top_k_chunks_per_section=top_k_chunks,
        )
    else:
        sections, chunks = retrieve_by_keyword(
            index, question,
            top_k_sections=top_k_sections,
            top_k_chunks_per_section=top_k_chunks,
        )

    context = build_context(chunks)
    if not context.strip():
        return "No relevant passages retrieved; try rephrasing or check the index."

    return answer_with_llm(
        question, context,
        use_ollama=use_ollama,
        ollama_model=ollama_model,
    )


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Section-level GraphRAG: retrieve from KG index and answer")
    p.add_argument("index", type=Path, help="Path to graphrag_index.json (from build_knowledge_graph)")
    p.add_argument("question", type=str, nargs="?", default="", help="Question to ask")
    p.add_argument("--interactive", "-i", action="store_true", help="Interactive Q&A")
    p.add_argument("--embed", action="store_true", help="Use sentence-transformers for retrieval (else keyword)")
    p.add_argument("--embed-model", type=str, default="all-MiniLM-L6-v2", help="Embedding model name")
    p.add_argument("--top-sections", type=int, default=3, help="Max number of sections")
    p.add_argument("--top-chunks", type=int, default=3, help="Max chunks per section")
    p.add_argument("--no-ollama", action="store_true", help="Do not use Ollama; use Settings.llm or OpenAI")
    p.add_argument("--ollama-model", type=str, default="llama3.2", help="Ollama model name")
    args = p.parse_args()

    if not args.index.exists():
        print(f"Index file not found: {args.index}")
        sys.exit(1)

    if args.interactive:
        print("Section-level GraphRAG interactive Q&A (empty line or quit to exit)")
        while True:
            try:
                q = input("\nQuestion> ").strip()
            except EOFError:
                break
            if not q or q.lower() in ("quit", "exit", "q"):
                break
            out = query(
                args.index,
                q,
                use_embedding=args.embed,
                embed_model_name=args.embed_model,
                top_k_sections=args.top_sections,
                top_k_chunks=args.top_chunks,
                use_ollama=not args.no_ollama,
                ollama_model=args.ollama_model,
            )
            print(out)
        return

    if not args.question:
        p.print_help()
        print("\nProvide a question or use --interactive")
        sys.exit(1)

    out = query(
        args.index,
        args.question,
        use_embedding=args.embed,
        embed_model_name=args.embed_model,
        top_k_sections=args.top_sections,
        top_k_chunks=args.top_chunks,
        use_ollama=not args.no_ollama,
        ollama_model=args.ollama_model,
    )
    print(out)


if __name__ == "__main__":
    main()
