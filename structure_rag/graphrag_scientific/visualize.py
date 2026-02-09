"""
Visualization: Citation network, Section graph, Query subgraph highlighting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Set

try:
    import networkx as nx
except ImportError:
    nx = None

try:
    from pyvis.network import Network
    HAS_PYVIS = True
except ImportError:
    HAS_PYVIS = False


def _truncate(s: str, n: int = 40) -> str:
    s = (s or "").strip()
    return s[:n] + "..." if len(s) > n else s


def _ref_display_title(title: str, text: str = "") -> str:
    """Normalize reference title for display; use English placeholders (no Chinese)."""
    raw = (title or text or "").strip()
    if raw in ("(no title)", "(no citation)"):
        return "(no title)"
    return raw or "(no title)"


def _inject_node_url_click(html: str) -> str:
    """Inject into PyVis HTML: double-click on a node with url opens it in a new tab."""
    inject = """
    if (typeof network !== 'undefined' && network.body && network.body.data && network.body.data.nodes) {
      network.on('doubleClick', function(params) {
        if (params.nodes.length === 1) {
          var n = network.body.data.nodes.get(params.nodes[0]);
          if (n && n.url) window.open(n.url, '_blank');
        }
      });
    }
"""
    # PyVis output typically has network = new vis.Network(container, data, options);
    if "new vis.Network(" in html and "network" in html:
        import re
        # Insert after network = new vis.Network(...);
        pattern = r"(network\s*=\s*new\s+vis\.Network\s*\([^)]+\)\s*;)"
        match = re.search(pattern, html)
        if match:
            insert_after = match.group(0)
            if inject.strip() not in html:
                html = html.replace(insert_after, insert_after + "\n" + inject.strip())
    return html


# Node type colors (aligned with build_knowledge_graph)
NODE_COLORS = {
    "paper": "#1f77b4",
    "section": "#2ca02c",
    "paragraph": "#17becf",
    "figure": "#ff7f0e",
    "table": "#9467bd",
    "formula": "#8c564b",
    "reference": "#d62728",
    "url": "#bcbd22",
}
# References: in corpus (clickable to open that doc's KG) vs external
REF_IN_CORPUS_COLOR = "#2ca02c"   # green: PDF exists in this folder
REF_EXTERNAL_COLOR = "#888888"    # grey: cited but not in corpus


def _kg_node_edge_data(
    G: "nx.DiGraph",
    paper_id: str,
    max_label: int,
    reference_links: Optional[dict[str, str]] = None,
    existing_paper_ids: Optional[Set[str]] = None,
) -> tuple[list[dict], list[dict]]:
    """Build node/edge data from G for full_kg_html (PyVis or fallback HTML). Returns (nodes, edges)."""
    if not nx or not G:
        return [], []
    ref_links = reference_links or {}
    in_corpus = existing_paper_ids or set()
    nodes: list[dict] = []
    edges: list[dict] = []
    for n in G.nodes():
        attrs = G.nodes[n]
        t = attrs.get("type", "paper")
        color = NODE_COLORS.get(t, "#888")
        node_url = None
        if t == "reference":
            target = ref_links.get(n)
            if target and target in in_corpus:
                color = REF_IN_CORPUS_COLOR
                node_url = f"{target}_knowledge_graph.html"
            else:
                color = REF_EXTERNAL_COLOR
        if t == "paper":
            label = attrs.get("title", paper_id)
        elif t == "section":
            order = attrs.get("order", "")
            sec_title = attrs.get("title", n)
            label = (f"[{order}] " if order != "" else "") + _truncate(sec_title, max_label)
        elif t == "paragraph":
            label = f"P{attrs.get('index', '')}: " + _truncate((attrs.get("text") or ""), 20)
        elif t == "figure":
            label = "Fig: " + _truncate(attrs.get("caption", ""), 25)
        elif t == "table":
            label = "Tab: " + _truncate(attrs.get("caption", ""), 25)
        elif t == "formula":
            label = "Eq: " + _truncate(attrs.get("latex", ""), 20)
        elif t == "reference":
            ref_title = _ref_display_title(attrs.get("title", ""), attrs.get("text", ""))
            if ref_title != "(no title)":
                ref_title = _truncate(ref_title, 25)
            suffix = " (in corpus)" if ref_links.get(n) in in_corpus else " (external)"
            label = f"Ref[{attrs.get('number', '?')}]" + f": {ref_title}" + suffix
        elif t == "url":
            label = "Link: " + _truncate(attrs.get("url", n), 30)
        else:
            label = _truncate(str(n), max_label)
        parts = [f"type: {t}"]
        for k, v in attrs.items():
            if k in ("text", "caption", "latex", "title", "authors", "venue", "year", "url", "abstract", "order", "level", "parent") and v is not None:
                vstr = str(v)
                parts.append(f"{k}: {vstr[:200]}" + ("..." if len(vstr) > 200 else ""))
        if t == "reference" and node_url:
            parts.append("Double-click node to open that document's knowledge graph")
        node_opts: dict[str, Any] = {"id": n, "label": label, "title": "\n".join(parts), "color": color, "shape": "dot"}
        if node_url:
            node_opts["url"] = node_url
        if t == "reference":
            node_opts["size"] = 28
        elif t == "section" and (attrs.get("title") or "").strip().lower() in ("references", "reference", "bibliography"):
            node_opts["size"] = 32
        nodes.append(node_opts)
    for u, v in G.edges():
        rel = G.edges[u, v].get("relation", "")
        edges.append({"from": u, "to": v, "title": rel})
    return nodes, edges


def full_kg_html(
    G: "nx.DiGraph",
    paper_id: str,
    title: str = "Knowledge Graph",
    max_label: int = 35,
    reference_links: Optional[dict[str, str]] = None,
    existing_paper_ids: Optional[Set[str]] = None,
) -> str:
    """Full heterogeneous graph visualization. References: green if in reference_links and existing_paper_ids (clickable), else grey."""
    if not nx:
        return ""
    ref_links = reference_links or {}
    in_corpus = existing_paper_ids or set()
    nodes, edges = _kg_node_edge_data(G, paper_id, max_label, ref_links, in_corpus)
    if not nodes:
        return ""
    if HAS_PYVIS:
        net = Network(directed=True, height="100vh", width="100%", bgcolor="#ffffff", font_color="#333")
        for no in nodes:
            nid = no["id"]
            net.add_node(nid, **{k: v for k, v in no.items() if k != "id"})
        for e in edges:
            net.add_edge(e["from"], e["to"], title=e.get("title", ""))
        html = net.generate_html()
        if ref_links and in_corpus:
            html = _inject_node_url_click(html)
        html = _inject_fit_and_fullheight(html)
        return html
    # When PyVis is missing, use vis-network CDN to generate HTML so reference nodes still show
    import json
    nodes_js = json.dumps(nodes, ensure_ascii=False)
    edges_js = json.dumps(edges, ensure_ascii=False)
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<link href="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/dist/vis-network.min.css" rel="stylesheet" />
<script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/vis-network.min.js"></script>
<style>#mynetwork {{ width: 100%; height: 100vh; }} body {{ margin: 0; }}</style>
</head><body>
<div id="mynetwork"></div>
<script>
var nodes = new vis.DataSet({nodes_js});
var edges = new vis.DataSet({edges_js});
var container = document.getElementById("mynetwork");
var data = {{ nodes: nodes, edges: edges }};
var options = {{ physics: {{ stabilization: {{ iterations: 200 }} }}, nodes: {{ font: {{ color: "#333" }} }} }};
var network = new vis.Network(container, data, options);
network.on("doubleClick", function(params) {{
  if (params.nodes.length === 1) {{
    var n = nodes.get(params.nodes[0]);
    if (n && n.url) window.open(n.url, "_blank");
  }}
}});
setTimeout(function() {{ try {{ network.fit(); }} catch(e) {{}} }}, 300);
</script>
</body></html>"""
    return _inject_fit_and_fullheight(html)


def _inject_fit_and_fullheight(html: str) -> str:
    """Inject: canvas full height (#mynetwork height to 100vh) and network.fit() after load to fit viewport."""
    import re
    # Set height to viewport height, keep other styles
    html = re.sub(r"(#mynetwork\s*\{[^}]*)(height:\s*)\d+px", r"\1\2 100vh; min-height: 400px;", html, count=1)
    if "network.fit(" in html:
        return html
    inject = """
    if (typeof network !== 'undefined') {
      setTimeout(function() { try { network.fit(); } catch(e) {} }, 300);
    }
"""
    if "new vis.Network(" in html and "network" in html:
        pattern = r"(network\s*=\s*new\s+vis\.Network\s*\([^)]+\)\s*;)"
        match = re.search(pattern, html)
        if match:
            insert_after = match.group(0)
            html = html.replace(insert_after, insert_after + "\n" + inject.strip())
    return html


def citation_network_html(
    G: "nx.DiGraph",
    paper_id: str,
    highlight_refs: Optional[Set[str]] = None,
    reference_links: Optional[dict[str, str]] = None,
    existing_paper_ids: Optional[Set[str]] = None,
) -> str:
    """Generate citation subgraph. References: green if in corpus (double-click to open KG), else grey."""
    if not nx or not HAS_PYVIS:
        return ""
    ref_links = reference_links or {}
    in_corpus = existing_paper_ids or set()
    net = Network(directed=True, height="100vh", width="100%", bgcolor="#f8f9fa")
    added = set()
    for u, v in G.edges():
        rel = G.edges[u, v].get("relation", "")
        if rel not in ("cites", "has_section", "lists_reference"):
            continue
        if u not in added:
            t = G.nodes[u].get("type", "")
            label = G.nodes[u].get("title") or G.nodes[u].get("caption") or u
            net.add_node(u, label=_truncate(str(label), 30), title=str(G.nodes[u]), color="#2ca02c" if t == "section" else "#1f77b4", shape="dot")
            added.add(u)
        if v not in added:
            t = G.nodes[v].get("type", "")
            label = G.nodes[v].get("title") or G.nodes[v].get("text", "")[:30] or v
            color = "#ff7f0e" if (highlight_refs and v in highlight_refs) else "#9467bd"
            node_url = None
            if t == "reference":
                target = ref_links.get(v)
                if target and target in in_corpus:
                    color = REF_IN_CORPUS_COLOR
                    node_url = f"{target}_knowledge_graph.html"
                else:
                    color = REF_EXTERNAL_COLOR
                display_title = _ref_display_title(G.nodes[v].get("title", ""), G.nodes[v].get("text", ""))
                label = _truncate(display_title, 28) + (" (in corpus)" if node_url else " (external)")
            opts = {"label": label, "title": str(G.nodes[v]), "color": color, "shape": "dot"}
            if node_url:
                opts["url"] = node_url
            net.add_node(v, **opts)
            added.add(v)
        net.add_edge(u, v, title=rel)
    html = net.generate_html()
    if ref_links and in_corpus:
        html = _inject_node_url_click(html)
    return html


def section_graph_html(G: "nx.DiGraph", paper_id: str) -> str:
    """Section hierarchy graph: Paper -> Section, Section -> parent/children."""
    if not nx or not HAS_PYVIS:
        return ""
    net = Network(directed=True, height="100vh", width="100%", bgcolor="#f8f9fa")
    for n in G.nodes():
        if G.nodes[n].get("type") not in ("paper", "section"):
            continue
        label = G.nodes[n].get("title") or n
        color = "#1f77b4" if G.nodes[n].get("type") == "paper" else "#2ca02c"
        net.add_node(n, label=_truncate(str(label), 35), title=str(G.nodes[n]), color=color, shape="dot")
    for u, v in G.edges():
        if G.nodes[u].get("type") not in ("paper", "section") or G.nodes[v].get("type") not in ("paper", "section"):
            continue
        rel = G.edges[u, v].get("relation", "")
        if rel in ("has_section", "parent_section", "refers_to_section"):
            net.add_edge(u, v, title=rel)
    return net.generate_html()


def query_subgraph_html(
    G: "nx.DiGraph",
    seed_node_ids: List[str],
    hops: int = 1,
    max_nodes: int = 80,
) -> str:
    """Highlight retrieved subgraph: seed nodes + spread neighbors."""
    if not nx or not HAS_PYVIS:
        return ""
    seeds = set(seed_node_ids)
    sub = set(seeds)
    frontier = list(seeds)
    for _ in range(hops):
        next_f = []
        for n in frontier:
            if len(sub) >= max_nodes:
                break
            for _, v in G.out_edges(n):
                if v not in sub and G.has_node(v):
                    sub.add(v)
                    next_f.append(v)
            for u, _ in G.in_edges(n):
                if u not in sub and G.has_node(u):
                    sub.add(u)
                    next_f.append(u)
        frontier = next_f
        if not frontier:
            break

    net = Network(directed=True, height="100vh", width="100%", bgcolor="#f8f9fa")
    for n in sub:
        attrs = G.nodes[n]
        t = attrs.get("type", "")
        label = attrs.get("title") or attrs.get("caption") or attrs.get("text", "")[:25] or n
        color = "#e74c3c" if n in seeds else "#3498db"
        net.add_node(n, label=_truncate(str(label), 28), title=str(attrs), color=color, shape="dot")
    for u, v in G.edges():
        if u in sub and v in sub:
            net.add_edge(u, v, title=G.edges[u, v].get("relation", ""))
    return net.generate_html()


def export_visualizations(
    G: "nx.DiGraph",
    output_dir: Path,
    paper_id: str,
    query_seed_nodes: Optional[List[str]] = None,
    export_full_kg: bool = True,
) -> List[str]:
    """Export full KG HTML + citation, section, query subgraphs. Refs: green if in corpus (double-click to open KG), else grey. When PyVis is missing, still generate full KG HTML (CDN fallback)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    ref_links = load_reference_links(output_dir, paper_id)
    existing_paper_ids = set(get_paper_ids_from_output_dir(output_dir))
    if export_full_kg:
        html_full = full_kg_html(
            G, paper_id,
            reference_links=ref_links,
            existing_paper_ids=existing_paper_ids,
        )
        if html_full:
            p = output_dir / f"{paper_id}_knowledge_graph.html"
            p.write_text(html_full, encoding="utf-8")
            written.append(str(p))
    if not HAS_PYVIS:
        return written
    html_cite = citation_network_html(
        G, paper_id,
        reference_links=ref_links,
        existing_paper_ids=existing_paper_ids,
    )
    if html_cite:
        html_cite = _inject_fit_and_fullheight(html_cite)
        p = output_dir / f"{paper_id}_citation_network.html"
        p.write_text(html_cite, encoding="utf-8")
        written.append(str(p))
    html_sec = section_graph_html(G, paper_id)
    if html_sec:
        html_sec = _inject_fit_and_fullheight(html_sec)
        p = output_dir / f"{paper_id}_section_graph.html"
        p.write_text(html_sec, encoding="utf-8")
        written.append(str(p))
    if query_seed_nodes:
        html_q = query_subgraph_html(G, query_seed_nodes)
        if html_q:
            p = output_dir / f"{paper_id}_query_subgraph.html"
            p.write_text(html_q, encoding="utf-8")
            written.append(str(p))
    return written


# ---------- Document-level view: PDF relations + select single doc for KG ----------

def load_reference_links(output_dir: Path, paper_id: str) -> dict[str, str]:
    """
    Read from output_dir / {paper_id}_reference_links.json: which ref in this doc maps to which document in this folder.
    Format: { "ref_1": "target_paper_id", "ref_2": "another_paper_id" }.
    Returns {} if file missing or invalid.
    """
    p = Path(output_dir) / f"{paper_id}_reference_links.json"
    if not p.exists():
        return {}
    try:
        import json
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items() if v}
        return {}
    except Exception:
        return {}


def get_paper_ids_from_output_dir(output_dir: Path) -> List[str]:
    """Scan output_dir for all *_kg.json or *_schema.json and return list of paper_id."""
    output_dir = Path(output_dir)
    ids: set[str] = set()
    for suf in ("_kg.json", "_schema.json"):
        for f in output_dir.glob("*" + suf):
            pid = f.name[: -len(suf)]
            if pid:
                ids.add(pid)
    return sorted(ids)


def load_document_relations(output_dir: Path) -> List[dict[str, str]]:
    """
    Read document relations from output_dir/document_relations.json.
    Format: [ {"source": "paper1", "target": "paper2", "relation": "cites"}, ... ]
    Returns [] if file missing or empty.
    """
    p = Path(output_dir) / "document_relations.json"
    if not p.exists():
        return []
    try:
        import json
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [e for e in data if isinstance(e, dict) and e.get("source") and e.get("target")]
        return []
    except Exception:
        return []


def build_document_relations_from_reference_links(output_dir: Path) -> List[dict[str, str]]:
    """
    Build document citation edges from each doc's {paper_id}_reference_links.json, merge with
    existing document_relations.json and write back. Each ref -> paper_id in reference_links
    becomes edge source=this doc, target=that paper_id, relation=cites.
    Returns merged relations list.
    """
    import json
    output_dir = Path(output_dir)
    paper_ids = set(get_paper_ids_from_output_dir(output_dir))
    relations: List[dict[str, str]] = load_document_relations(output_dir)
    seen = {(r["source"], r["target"]) for r in relations}

    for pid in paper_ids:
        p = output_dir / f"{pid}_reference_links.json"
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                continue
            for _ref_key, target_id in data.items():
                if not isinstance(target_id, str) or target_id not in paper_ids:
                    continue
                if (pid, target_id) in seen:
                    continue
                seen.add((pid, target_id))
                relations.append({"source": pid, "target": target_id, "relation": "cites"})
        except Exception:
            continue

    if relations:
        out_path = output_dir / "document_relations.json"
        out_path.write_text(json.dumps(relations, ensure_ascii=False, indent=2), encoding="utf-8")
    return relations


def document_network_html(
    paper_ids: List[str],
    relations: List[dict[str, str]],
    output_dir: Path,
) -> str:
    """
    Generate document-relations graph: nodes = PDFs (paper_id), edges from document_relations.json.
    Shows only PDF-to-PDF links, not single-document internal structure.
    Uses PyVis if available; otherwise vis-network CDN so the view is always an interactive graph.
    """
    if not paper_ids:
        return ""
    output_dir = Path(output_dir)
    paper_set = set(paper_ids)
    nodes = [
        {
            "id": pid,
            "label": _truncate(pid, 50),
            "title": f"Double-click to open this document's knowledge graph in a new tab\n{pid}",
            "color": "#1f77b4",
            "url": f"{pid}_knowledge_graph.html",
            "shape": "dot",
        }
        for pid in paper_ids
    ]
    edges = [
        {"from": r["source"], "to": r["target"], "title": r.get("relation", "relates_to")}
        for r in relations
        if r.get("source") in paper_set and r.get("target") in paper_set
    ]
    if HAS_PYVIS:
        net = Network(directed=True, height="100vh", width="100%", bgcolor="#f8f9fa", font_color="#333")
        for no in nodes:
            nid = no["id"]
            net.add_node(nid, **{k: v for k, v in no.items() if k != "id"})
        for e in edges:
            net.add_edge(e["from"], e["to"], title=e.get("title", ""), arrows="to")
        html = net.generate_html()
        html = _inject_node_url_click(html)
        return _inject_fit_and_fullheight(html)
    # When PyVis is missing: use vis-network CDN so we still get an interactive graph
    import json
    nodes_js = json.dumps(nodes, ensure_ascii=False)
    edges_js = json.dumps(edges, ensure_ascii=False)
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Document relations</title>
<link href="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/dist/vis-network.min.css" rel="stylesheet" />
<script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/vis-network.min.js"></script>
<style>#mynetwork {{ width: 100%; height: 100vh; min-height: 400px; }} body {{ margin: 0; }}</style>
</head><body>
<div id="mynetwork"></div>
<script>
var nodes = new vis.DataSet({nodes_js});
var edges = new vis.DataSet({edges_js});
var container = document.getElementById("mynetwork");
var data = {{ nodes: nodes, edges: edges }};
var options = {{ physics: {{ stabilization: {{ iterations: 200 }} }}, nodes: {{ font: {{ color: "#333" }} }}, edges: {{ arrows: {{ to: {{ enabled: true }} }} }} }};
var network = new vis.Network(container, data, options);
network.on("doubleClick", function(params) {{
  if (params.nodes.length === 1) {{
    var n = nodes.get(params.nodes[0]);
    if (n && n.url) window.open(n.url, "_blank");
  }}
}});
setTimeout(function() {{ try {{ network.fit(); }} catch(e) {{}} }}, 300);
</script>
</body></html>"""
    return _inject_fit_and_fullheight(html)


def export_document_index(output_dir: Path) -> List[str]:
    """
    Generate document-level index and document-relation graph:
    - index.html: choose "Document relations" or a PDF to show relation graph or that PDF's knowledge graph.
    - document_network.html: graph of PDF-to-PDF relations only.
    Returns list of written file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []

    paper_ids = get_paper_ids_from_output_dir(output_dir)
    if not paper_ids:
        return written

    relations = load_document_relations(output_dir)
    html_net = document_network_html(paper_ids, relations, output_dir)
    if html_net:
        html_net = _inject_fit_and_fullheight(html_net)
        p = output_dir / "document_network.html"
        p.write_text(html_net, encoding="utf-8")
        written.append(str(p))

    # index.html: dropdown for "Document relations" or a PDF; iframe shows corresponding content
    first = '<option value="document_network.html">Document relations</option>\n'
    rest = "\n".join(f'<option value="{pid}_knowledge_graph.html">{pid}</option>' for pid in paper_ids)
    index_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Knowledge Graph</title>
  <style>
    body {{ font-family: sans-serif; margin: 12px; }}
    .toolbar {{ margin-bottom: 8px; display: flex; align-items: center; gap: 8px; }}
    label {{ font-weight: bold; }}
    select {{ min-width: 280px; padding: 6px; }}
    iframe {{ border: 1px solid #ccc; width: 100%; height: calc(100vh - 140px); min-height: 500px; }}
  </style>
</head>
<body>
  <h2>Knowledge Graph</h2>
  <div class="toolbar">
    <label for="view">View:</label>
    <select id="view">
{first}{rest}
    </select>
    <span style="color:#666;">Choose "Document relations" for PDF-to-PDF links; choose a document for its knowledge graph.</span>
  </div>
  <iframe id="kgframe" src="document_network.html"></iframe>
  <script>
    document.getElementById("view").addEventListener("change", function() {{
      document.getElementById("kgframe").src = this.value;
    }});
  </script>
</body>
</html>
"""
    index_path = output_dir / "index.html"
    index_path.write_text(index_content, encoding="utf-8")
    written.append(str(index_path))
    return written
