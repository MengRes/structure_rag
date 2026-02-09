"""
Document-level heterogeneous knowledge graph.

Node types: Paper, Section, Paragraph, Figure, Table, Formula, Reference.
Edges: has_section, has_paragraph, has_figure, has_table, has_formula, cites, refers_to_section.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

try:
    import networkx as nx
except ImportError:
    nx = None

from .schema import load_schema, mineru_to_schema

# URL -> graph node id (same url shares one node)
_url_id_map: dict[str, str] = {}
_url_counter = 0


def _url_to_node_id(url: str) -> str:
    global _url_counter
    if url not in _url_id_map:
        _url_id_map[url] = f"link_{_url_counter}"
        _url_counter += 1
    return _url_id_map[url]


def build_heterogeneous_graph(schema: dict[str, Any]) -> "nx.DiGraph":
    """
    Build heterogeneous graph from unified schema.
    Node attributes: type, title/text/caption/latex etc.; edges have relation.
    """
    global _url_id_map, _url_counter
    _url_id_map = {}
    _url_counter = 0
    if nx is None:
        raise ImportError("pip install networkx")
    G = nx.DiGraph()
    paper_id = schema.get("paper_id", "paper")
    G.add_node(
        paper_id,
        type="paper",
        title=paper_id,
        abstract=(schema.get("abstract") or "")[:3000],
    )

    ref_by_id = {r["id"]: r for r in schema.get("references", [])}
    for rid, r in ref_by_id.items():
        G.add_node(
            rid,
            type="reference",
            number=r.get("number"),
            text=r.get("text", "")[:500],
            title=r.get("title", ""),
            authors=r.get("authors", ""),
            venue=r.get("venue", ""),
            year=r.get("year", ""),
        )
    G.add_edge(paper_id, paper_id, relation="self")  # avoid empty graph
    G.remove_edge(paper_id, paper_id)

    for i, sec in enumerate(schema.get("sections", [])):
        sid = sec["id"]
        G.add_node(
            sid,
            type="section",
            title=sec.get("title", ""),
            level=sec.get("level", 1),
            parent=sec.get("parent"),
            order=sec.get("order", i),
        )
        G.add_edge(paper_id, sid, relation="has_section")
        # Edge from subsection to parent section (when level>1 parent is set)
        if sec.get("parent"):
            G.add_edge(sid, sec["parent"], relation="parent_section")
            G.add_edge(sec["parent"], sid, relation="refers_to_section")

        for i, p in enumerate(sec.get("paragraphs", [])):
            pid = f"{sid}::p_{i}"
            text = p.get("text", "")
            G.add_node(pid, type="paragraph", text=text[:2000], section_id=sid, index=i, page_idx=p.get("page_idx", 0))
            G.add_edge(sid, pid, relation="has_paragraph")
            # Paragraph-level citation: Paragraph -> cites -> Reference
            for ref_id in p.get("citations", []):
                if not G.has_node(ref_id):
                    G.add_node(
                        ref_id,
                        type="reference",
                        number=0,
                        text="",
                        title="",
                        authors="",
                        venue="",
                        year="",
                    )
                    G.add_edge(paper_id, ref_id, relation="contains")
                G.add_edge(pid, ref_id, relation="cites")
            # URLs in paragraph -> link nodes, edge paragraph --mentions_link--> link
            for url in p.get("urls") or []:
                link_id = _url_to_node_id(url)
                if not G.has_node(link_id):
                    G.add_node(link_id, type="url", url=url[:500])
                    G.add_edge(paper_id, link_id, relation="contains")
                G.add_edge(pid, link_id, relation="mentions_link")

        for fig in sec.get("figures", []):
            fid = fig["id"]
            if not G.has_node(fid):
                G.add_node(
                    fid,
                    type="figure",
                    caption=fig.get("caption", ""),
                    image_path=fig.get("image_path", ""),
                    section_id=sid,
                    page_idx=fig.get("page_idx", 0),
                )
            G.add_edge(sid, fid, relation="has_figure")
            for url in fig.get("urls") or []:
                link_id = _url_to_node_id(url)
                if not G.has_node(link_id):
                    G.add_node(link_id, type="url", url=url[:500])
                    G.add_edge(paper_id, link_id, relation="contains")
                G.add_edge(fid, link_id, relation="mentions_link")
            # Figure -> preceding paragraph (context in document order)
            idx = fig.get("preceding_paragraph_index")
            if idx is not None and 0 <= idx < len(sec.get("paragraphs", [])):
                prev_pid = f"{sid}::p_{idx}"
                if G.has_node(prev_pid):
                    G.add_edge(fid, prev_pid, relation="context_paragraph")

        for tbl in sec.get("tables", []):
            tid = tbl["id"]
            if not G.has_node(tid):
                G.add_node(
                    tid,
                    type="table",
                    caption=tbl.get("caption", ""),
                    table_body_preview=tbl.get("table_body_preview", ""),
                    section_id=sid,
                    page_idx=tbl.get("page_idx", 0),
                )
            G.add_edge(sid, tid, relation="has_table")
            for url in tbl.get("urls") or []:
                link_id = _url_to_node_id(url)
                if not G.has_node(link_id):
                    G.add_node(link_id, type="url", url=url[:500])
                    G.add_edge(paper_id, link_id, relation="contains")
                G.add_edge(tid, link_id, relation="mentions_link")
            idx = tbl.get("preceding_paragraph_index")
            if idx is not None and 0 <= idx < len(sec.get("paragraphs", [])):
                prev_pid = f"{sid}::p_{idx}"
                if G.has_node(prev_pid):
                    G.add_edge(tid, prev_pid, relation="context_paragraph")

        for eq in sec.get("formulas", []):
            eid = eq["id"]
            if not G.has_node(eid):
                G.add_node(
                    eid,
                    type="formula",
                    latex=eq.get("latex", ""),
                    section_id=sid,
                    page_idx=eq.get("page_idx", 0),
                )
            G.add_edge(sid, eid, relation="has_formula")
            idx = eq.get("preceding_paragraph_index")
            if idx is not None and 0 <= idx < len(sec.get("paragraphs", [])):
                prev_pid = f"{sid}::p_{idx}"
                if G.has_node(prev_pid):
                    G.add_edge(eid, prev_pid, relation="context_paragraph")

        # Section-level citation aggregation (which refs this section cites)
        for ref_id in sec.get("citations", []):
            if G.has_node(ref_id):
                G.add_edge(sid, ref_id, relation="cites")
            if not G.has_node(ref_id):
                G.add_node(
                    ref_id,
                    type="reference",
                    number=0,
                    text="",
                    title="",
                    authors="",
                    venue="",
                    year="",
                )
                G.add_edge(paper_id, ref_id, relation="contains")
            G.add_edge(sid, ref_id, relation="cites")

    # Attach schema.references to References section for centralized display in graph
    ref_list = schema.get("references") or []
    if ref_list:
        for sec in schema.get("sections") or []:
            title = (sec.get("title") or "").strip().lower()
            if title in ("references", "reference", "bibliography"):
                sid = sec["id"]
                if not G.has_node(sid):
                    continue
                for r in ref_list:
                    rid = r.get("id")
                    if rid and G.has_node(rid):
                        G.add_edge(sid, rid, relation="lists_reference")
                break

    return G


def export_kg_json(G: "nx.DiGraph", path: Path) -> None:
    nodes = [{"id": n, **{k: v for k, v in G.nodes[n].items()}} for n in G.nodes()]
    edges = [{"source": u, "target": v, **dict(G.edges[u, v])} for u, v in G.edges()]
    for n in nodes:
        for k, v in list(n.items()):
            if k == "id":
                continue
            try:
                json.dumps(v)
            except (TypeError, ValueError):
                n[k] = str(v)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"nodes": nodes, "edges": edges}, f, ensure_ascii=False, indent=2)
