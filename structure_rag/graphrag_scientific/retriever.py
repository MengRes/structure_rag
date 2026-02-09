"""
Hybrid GraphRAG retrieval: semantic recall → graph expansion → subgraph → rerank.

score = α * embedding_sim + β * graph_score + γ * citation_weight
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

try:
    import networkx as nx
except ImportError:
    nx = None

from .embedding_store import EmbeddingStore, get_embedding_model


def _chunk_to_node_ids(chunk: dict[str, Any]) -> List[str]:
    """Get graph node id(s) for a chunk (for graph expansion)."""
    cid = chunk.get("chunk_id", "")
    sid = chunk.get("section_id", "")
    ctype = chunk.get("type", "")
    node_ids = []
    if ctype == "reference":
        if cid:
            node_ids.append(cid)  # ref_1, ref_2 etc. are graph node ids
        return node_ids
    if ctype == "abstract":
        if cid and "::" in cid:
            node_ids.append(cid.split("::", 1)[0])  # paper_id
        return node_ids
    if sid:
        node_ids.append(sid)
    if cid and "::" in cid:
        # section_id::p_0 -> paragraph node; section_id::section -> section; section_id::fig_1 -> fig_1
        rest = cid.split("::", 1)[-1]
        if rest == "section":
            node_ids.append(sid)
        elif rest.startswith("p_"):
            node_ids.append(cid)  # paragraph node id = chunk_id
        elif rest.startswith("fig_"):
            node_ids.append(rest)  # fig_1
        elif rest.startswith("tab_"):
            node_ids.append(rest)
        elif rest.startswith("eq_"):
            node_ids.append(rest)
    return node_ids


def expand_subgraph(
    G: "nx.DiGraph",
    seed_node_ids: List[str],
    hops: int = 1,
    max_nodes: int = 50,
) -> List[str]:
    """BFS expansion from seed nodes; returns list of subgraph node ids."""
    if not nx or not seed_node_ids:
        return list(seed_node_ids)
    seen = set()
    frontier = [n for n in seed_node_ids if G.has_node(n)]
    for _ in range(hops + 1):
        next_frontier = []
        for n in frontier:
            if n in seen or len(seen) >= max_nodes:
                continue
            seen.add(n)
            for _, v in G.out_edges(n):
                if v not in seen:
                    next_frontier.append(v)
            for u, _ in G.in_edges(n):
                if u not in seen:
                    next_frontier.append(u)
        frontier = next_frontier
        if not frontier:
            break
    return list(seen)


def graph_score(chunk: dict[str, Any], subgraph_node_ids: List[str]) -> float:
    """Relevance to subgraph: whether chunk's nodes are in subgraph (section/paragraph)."""
    nodes = _chunk_to_node_ids(chunk)
    if not nodes:
        return 0.0
    hit = sum(1 for n in nodes if n in subgraph_node_ids)
    return hit / max(len(nodes), 1)


def citation_weight(chunk: dict[str, Any], G: "nx.DiGraph", section_ids: List[str]) -> float:
    """Give a small bonus if the chunk's section has many citations."""
    sid = chunk.get("section_id")
    if not sid or sid not in section_ids:
        return 0.0
    if not G.has_node(sid):
        return 0.0
    cites = list(G.successors(sid))
    ref_count = sum(1 for c in cites if G.nodes[c].get("type") == "reference")
    return min(ref_count * 0.1, 0.5)


def hybrid_retrieve(
    query: str,
    embedding_store: EmbeddingStore,
    G: "nx.DiGraph",
    top_k_semantic: int = 15,
    hops: int = 1,
    top_k_final: int = 5,
    alpha: float = 0.6,
    beta: float = 0.3,
    gamma: float = 0.1,
    model: Any = None,
) -> List[Tuple[dict, float]]:
    """
    Step A semantic recall → Step B graph expansion → Step C subgraph → Step D rerank.
    Returns [(chunk, score), ...].
    """
    if model is None:
        model = get_embedding_model()
    if model is None:
        # No embedding: simple keyword match scoring in chunk text
        q_lower = query.lower()
        all_chunks = embedding_store.chunks
        scored = []
        for c in all_chunks:
            t = (c.get("text") or "").lower()
            sc = sum(1 for w in q_lower.split() if w in t)
            scored.append((c, float(sc)))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k_final]

    q_emb = model.encode([query], normalize_embeddings=True)[0].tolist()
    semantic_hits = embedding_store.search(q_emb, top_k=top_k_semantic)
    if not semantic_hits:
        return []

    seed_nodes = []
    for c, _ in semantic_hits:
        seed_nodes.extend(_chunk_to_node_ids(c))
    subgraph_nodes = expand_subgraph(G, seed_nodes, hops=hops, max_nodes=50)
    section_ids = [n for n in subgraph_nodes if G.nodes.get(n, {}).get("type") == "section"]

    # Rerank: composite score only for chunks in semantic_hits
    reranked = []
    for chunk, sim in semantic_hits:
        gs = graph_score(chunk, subgraph_nodes)
        cw = citation_weight(chunk, G, section_ids)
        score = alpha * sim + beta * gs + gamma * cw
        reranked.append((chunk, score))
    reranked.sort(key=lambda x: -x[1])
    return reranked[:top_k_final]


def build_context_from_chunks(
    chunk_scores: List[Tuple[dict, float]],
    max_chars: int = 12000,
) -> str:
    """Format retrieved chunks as LLM context, sorted by section order and chunk_id."""
    # Sort by section order, then chunk_id (within section: paragraph < figure < formula)
    sorted_scores = sorted(
        chunk_scores,
        key=lambda x: (x[0].get("section_order", 99999), x[0].get("chunk_id", "")),
    )
    parts = []
    n = 0
    for chunk, _ in sorted_scores:
        sec_title = chunk.get("section_title", "")
        ctype = chunk.get("type", "")
        text = chunk.get("text", "")
        if ctype == "reference":
            meta = chunk.get("metadata", {})
            num = meta.get("number", "")
            title = meta.get("title") or text
            authors = meta.get("authors", "")
            venue = meta.get("venue", "")
            year = meta.get("year", "")
            ref_line = f"[Reference [{num}]] {title}"
            if authors or venue or year:
                ref_line += f". {authors}. {venue}. {year}".rstrip(". ")
            block = f"[{ctype}]\n{ref_line}"
        else:
            # If subsection (parent exists), show level
            sec_order = chunk.get("section_order", "")
            sec_level = chunk.get("section_level", 1)
            head = f"[Section: {sec_title}]"
            if sec_level and sec_level > 1:
                head += f" (subsection, level {sec_level})"
            block = f"{head}\n[{ctype}]\n{text[:1500]}"
        if n + len(block) > max_chars:
            break
        parts.append(block)
        n += len(block)
    return "\n\n---\n\n".join(parts)
