#!/usr/bin/env python3
"""
Build knowledge graph from MinerU parse output.

Extracts from content_list.json: sections, figures/tables, references, in-text citations;
builds nodes (Document, Section, Figure, Table, Reference) and relations
(contains, cites, next_section). Optionally keeps per-section paragraph summaries for internal subgraph/embedding.
Deps: pip install networkx
Viz: pip install pyvis (interactive HTML) or matplotlib (static)
Triples: default LLM; use --ollama for local Ollama or --no-llm for regex. Ollama: pip install llama-index-llms-ollama; OpenAI: pip install llama-index-llms-openai
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Optional

# Triple (subject, relation, object)
Triple = tuple[str, str, str]

try:
    import networkx as nx
except ImportError:
    nx = None

try:
    from pyvis.network import Network
    HAS_PYVIS = True
except ImportError:
    HAS_PYVIS = False

# Optional: LlamaIndex for LLM triple extraction (pip install llama-index)
_HAS_LLAMA = False
try:
    from llama_index.core import Settings
    from llama_index.core.llms import LLM
    _HAS_LLAMA = True
except ImportError:
    pass

CITATION_PATTERN = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def _normalize_whitespace(s: str) -> str:
    return " ".join(s.split()) if s else ""


def _truncate_at_word(text: str, max_chars: int, suffix: str = "...") -> str:
    """
    Truncate at word boundary within max_chars to avoid mid-word cuts.
    If over limit, cut at last space and append suffix; if no space, still cut at max_chars.
    """
    if not text or max_chars <= 0:
        return ""
    text = _normalize_whitespace(text)
    if len(text) <= max_chars:
        return text
    chunk = text[: max_chars + 1]
    last_space = chunk.rfind(" ")
    if last_space > max_chars // 2:
        return chunk[:last_space].strip() + suffix
    return text[:max_chars].rstrip() + suffix


def _load_content_list(path: Path) -> list[dict[str, Any]]:
    """Load via content_loader; supports content_list_v2."""
    from structure_rag.content_loader import load_content_list
    return load_content_list(Path(path))


def parse_reference_item(item: str) -> tuple[int, str]:
    item = _normalize_whitespace(item)
    m = re.match(r"^(\d+)\.\s*(.+)$", item)
    if m:
        return int(m.group(1)), m.group(2).strip()
    return 0, item


def extract_citation_numbers(text: str) -> list[int]:
    seen = set()
    out = []
    for m in CITATION_PATTERN.finditer(text):
        for part in m.group(1).split(","):
            n = int(part.strip())
            if n not in seen:
                seen.add(n)
                out.append(n)
    return out


def get_figure_caption(block: dict) -> str:
    captions = block.get("image_caption") or block.get("caption") or []
    if isinstance(captions, list):
        return _normalize_whitespace(" ".join(captions))
    return _normalize_whitespace(str(captions))


def get_table_caption(block: dict) -> str:
    captions = block.get("table_caption") or block.get("caption") or []
    if isinstance(captions, list):
        return _normalize_whitespace(" ".join(captions))
    return _normalize_whitespace(str(captions))


def get_image_path(block: dict) -> str:
    return block.get("img_path") or block.get("image_path") or ""


def chunk_text(
    paragraphs: list[str],
    chunk_size: int = 300,
    overlap: int = 50,
) -> list[str]:
    """
    Split paragraph list into chunks by character count, aligned to word boundaries.
    chunk_size: target length per chunk (chars); overlap: overlap between adjacent chunks.
    """
    if not paragraphs or chunk_size <= 0:
        return []
    text = " ".join(_normalize_whitespace(p) for p in paragraphs)
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        # If end is mid-word: prefer previous space; if no space in range, extend to next space
        if end < len(text) and end > start and text[end] not in " \t":
            last_sp = text.rfind(" ", start, end)
            if last_sp >= start:
                end = last_sp + 1
            else:
                next_sp = text.find(" ", end)
                end = (next_sp + 1) if next_sp != -1 else len(text)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap if overlap < chunk_size else end
        # If start is mid-word, advance to next word start
        if start > 0 and start < len(text) and text[start] not in " \t":
            next_sp = text.find(" ", start)
            if next_sp != -1:
                start = next_sp + 1
            else:
                start = len(text)
    return chunks


def extract_triples_simple(text: str, max_triples: int) -> list[Triple]:
    """
    Extract (subject, relation, object) triples from text using rules/regex (no LLM).
    Default when LLM is disabled; can be replaced by external extract_triples.
    """
    if max_triples <= 0:
        return []
    text = _normalize_whitespace(text)
    triples: list[Triple] = []
    seen: set[tuple[str, str, str]] = set()

    def add(s: str, r: str, o: str) -> None:
        s = _truncate_at_word(s.strip(), 80, "")
        o = _truncate_at_word(o.strip(), 80, "")
        if len(s) < 2 or len(o) < 2 or (s, r, o) in seen:
            return
        seen.add((s, r, o))
        triples.append((s, r, o))

    # Patterns: X is/are Y, X of Y, X in Y, X for Y, X - Y, X (Y), X: Y
    patterns = [
        (r"([A-Za-z0-9\s\-]{2,40})\s+is\s+([A-Za-z0-9\s\-]{2,60})", "is"),
        (r"([A-Za-z0-9\s\-]{2,40})\s+are\s+([A-Za-z0-9\s\-]{2,60})", "are"),
        (r"([A-Za-z0-9\s\-]{2,40})\s+of\s+([A-Za-z0-9\s\-]{2,60})", "of"),
        (r"([A-Za-z0-9\s\-]{2,40})\s+in\s+([A-Za-z0-9\s\-]{2,60})", "in"),
        (r"([A-Za-z0-9\s\-]{2,40})\s+for\s+([A-Za-z0-9\s\-]{2,60})", "for"),
        (r"([A-Za-z0-9\s\-]{2,40})\s+from\s+([A-Za-z0-9\s\-]{2,60})", "from"),
        (r"([A-Za-z0-9\s\-]{2,40})\s+to\s+([A-Za-z0-9\s\-]{2,60})", "to"),
        (r"([A-Za-z0-9\s\-]{2,40})\s+-\s+([A-Za-z0-9\s\-]{2,60})", "related_to"),
        (r"([A-Za-z0-9\s\-]{2,40})\s+such as\s+([A-Za-z0-9\s\-]{2,60})", "such_as"),
    ]
    for pat, rel in patterns:
        if len(triples) >= max_triples:
            break
        for m in re.finditer(pat, text, re.IGNORECASE):
            add(m.group(1), rel, m.group(2))
            if len(triples) >= max_triples:
                break
    return triples[:max_triples]


# --------------- LLM triple extraction (LlamaIndex SimpleLLMPathExtractor style) ---------------

LLM_TRIPLE_PROMPT = """Extract knowledge triples (subject, relation, object) from the following text.
Each triple should be a short factual statement. Output exactly one triple per line in this format:
SUBJECT | RELATION | OBJECT
Use only the characters before the newline for each line. No numbering, no extra text.

Text:
"""
# Regex to parse LLM output: SUBJECT | RELATION | OBJECT per line
_LLM_TRIPLE_LINE = re.compile(r"^\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*$")


def _parse_llm_triple_lines(response: str, max_triples: int) -> list[Triple]:
    """Parse (subject, relation, object) list from LLM multi-line output."""
    triples: list[Triple] = []
    seen: set[tuple[str, str, str]] = set()
    for line in response.strip().splitlines():
        line = line.strip()
        if not line or len(triples) >= max_triples:
            continue
        m = _LLM_TRIPLE_LINE.match(line)
        if not m:
            continue
        s, r, o = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
        if len(s) < 2 or len(o) < 2 or len(r) < 1:
            continue
        s = _truncate_at_word(s, 80, "")
        o = _truncate_at_word(o, 80, "")
        r = _truncate_at_word(r, 40, "")
        key = (s, r, o)
        if key in seen:
            continue
        seen.add(key)
        triples.append((s, r, o))
    return triples[:max_triples]


def extract_triples_llm(
    text: str,
    max_triples: int,
    *,
    llm: Optional[Any] = None,
    prompt_template: str = LLM_TRIPLE_PROMPT,
) -> list[Triple]:
    """
    Extract (subject, relation, object) triples from text using LLM.
    LLM only, no regex fallback; returns [] if LLM not configured.
    """
    if max_triples <= 0:
        return []
    text = _normalize_whitespace(text)
    if not text:
        return []
    prompt = prompt_template + text.strip()[:4000]  # Keep single chunk short
    if llm is None and _HAS_LLAMA:
        llm = getattr(Settings, "llm", None)
    if llm is None:
        return []
    try:
        if hasattr(llm, "complete"):
            response = llm.complete(prompt)
            out = response.text if hasattr(response, "text") else str(response)
        else:
            out = str(llm(prompt))
    except Exception:
        return []
    return _parse_llm_triple_lines(out, max_triples)


def make_llm_extract_triples_fn(
    llm: Optional[Any] = None,
    prompt_template: str = LLM_TRIPLE_PROMPT,
) -> Callable[[str, int], list[Triple]]:
    """Return a (text, max_triples) -> list[Triple] extractor using LLM."""
    def fn(text: str, max_triples: int) -> list[Triple]:
        return extract_triples_llm(text, max_triples, llm=llm, prompt_template=prompt_template)
    return fn


def build_graph_from_content_list(
    content_list: list[dict[str, Any]],
    doc_name: str = "document",
    include_internal_subgraph: bool = True,
) -> "nx.DiGraph":
    if nx is None:
        raise ImportError("Install networkx: pip install networkx")

    G = nx.DiGraph()
    doc_id = f"doc:{doc_name}"
    G.add_node(doc_id, type="document", title=doc_name)

    current_section_id: Optional[str] = None
    section_order: list[str] = []
    section_paragraphs: dict[str, list[str]] = {}
    references: dict[int, str] = {}
    ref_texts: dict[int, str] = {}
    figure_ids: list[str] = []
    table_ids: list[str] = []

    for block in content_list:
        btype = block.get("type") or "text"
        text = _normalize_whitespace(block.get("text") or "")
        page_idx = block.get("page_idx", 0)

        if btype == "text" and block.get("text_level") is not None:
            level = int(block.get("text_level", 1))
            title = text or f"Section_{level}"
            if title.strip().lower() == "references":
                current_section_id = None
                continue
            sid = f"section:{len(section_order)+1}:{_truncate_at_word(title, 50, '')}"
            section_order.append(sid)
            G.add_node(sid, type="section", title=title, level=level, page_idx=page_idx, index=len(section_order))
            if current_section_id and current_section_id in G:
                G.add_edge(current_section_id, sid, relation="next_section")
            G.add_edge(doc_id, sid, relation="contains")
            current_section_id = sid
            section_paragraphs.setdefault(sid, [])
            continue

        if btype == "list" and block.get("sub_type") == "ref_text":
            for item in block.get("list_items") or []:
                num, ref_content = parse_reference_item(item)
                if num <= 0:
                    continue
                ref_id = f"ref:{num}"
                references[num] = ref_id
                ref_texts[num] = ref_content
                G.add_node(ref_id, type="reference", number=num, text=_truncate_at_word(ref_content, 500, ""))
                G.add_edge(doc_id, ref_id, relation="contains")
            continue

        if btype == "image":
            caption = get_figure_caption(block)
            img_path = get_image_path(block)
            fid = f"figure:{len(figure_ids)+1}"
            figure_ids.append(fid)
            G.add_node(fid, type="figure", caption=caption, image_path=img_path, page_idx=block.get("page_idx", 0))
            G.add_edge(doc_id, fid, relation="contains")
            if current_section_id:
                G.add_edge(current_section_id, fid, relation="contains")
            continue

        if btype == "table":
            caption = get_table_caption(block)
            tid = f"table:{len(table_ids)+1}"
            table_ids.append(tid)
            table_body = block.get("table_body") or ""
            G.add_node(tid, type="table", caption=caption, page_idx=block.get("page_idx", 0), table_body_preview=_truncate_at_word(table_body, 300, "") if table_body else "")
            G.add_edge(doc_id, tid, relation="contains")
            if current_section_id:
                G.add_edge(current_section_id, tid, relation="contains")
            continue

        if btype == "text" and text and current_section_id:
            section_paragraphs.setdefault(current_section_id, []).append(text)
            for num in extract_citation_numbers(text):
                ref_id = references.get(num)
                if not ref_id:
                    ref_id = f"ref:{num}"
                    references[num] = ref_id
                    ref_texts[num] = ""
                    G.add_node(ref_id, type="reference", number=num, text="")
                    G.add_edge(doc_id, ref_id, relation="contains")
                G.add_edge(current_section_id, ref_id, relation="cites")

    if include_internal_subgraph:
        for sid, paras in section_paragraphs.items():
            full_text = " ".join(paras)
            G.nodes[sid]["paragraph_count"] = len(paras)
            G.nodes[sid]["text_preview"] = _truncate_at_word(full_text, 2000, "")
            G.nodes[sid]["paragraphs"] = paras  # For building section internal knowledge graph

    return G


def _section_filename(section_id: str, main_stem: str) -> str:
    """Build a safe filename from section node id (same dir as main)."""
    # section_id e.g. "section:1:Introduction" -> section_1_Introduction
    safe = re.sub(r"[^\w\-]", "_", section_id).strip("_")
    return f"{main_stem}_internal_{safe}.html"


def build_section_internal_graph(
    G: "nx.DiGraph",
    section_id: str,
    *,
    chunk_size: int = 300,
    overlap: int = 50,
    max_triples_per_chunk: int = 5,
    extract_triples_fn: Optional[Callable[[str, int], list[Triple]]] = None,
) -> Optional["nx.DiGraph"]:
    """
    Build internal knowledge subgraph for a section: chunks, triples, ref/figure/table citations.
    Nodes: section, chunk, triple, reference, figure, table; edges: section->chunk/ref/figure/table etc.
    """
    if nx is None or section_id not in G.nodes():
        return None
    attrs = G.nodes[section_id]
    if attrs.get("type") != "section":
        return None
    paras = attrs.get("paragraphs") or []
    cited_refs = [v for v in G.successors(section_id) if G.nodes[v].get("type") == "reference"]
    figures_in_section = [v for v in G.successors(section_id) if G.nodes[v].get("type") == "figure"]
    tables_in_section = [v for v in G.successors(section_id) if G.nodes[v].get("type") == "table"]
    chunks_list = chunk_text(paras, chunk_size=chunk_size, overlap=overlap)
    if not chunks_list and not cited_refs and not figures_in_section and not tables_in_section:
        return None

    extract_fn = extract_triples_fn or extract_triples_simple
    H = nx.DiGraph()
    H.add_node(section_id, type="section", title=attrs.get("title", ""), label=_truncate_at_word(attrs.get("title") or section_id, 50, ""))

    for i, chunk_text_str in enumerate(chunks_list):
        cid = f"{section_id}::chunk_{i}"
        # Chunk node: label = "Chunk i" + short preview; tooltip = summary; keep text for GraphRAG
        clabel = f"Chunk {i}: " + _truncate_at_word(chunk_text_str, 25, "")
        preview = _truncate_at_word(chunk_text_str, 200, "")
        H.add_node(cid, type="chunk", index=i, text=chunk_text_str, text_preview=f"[Chunk {i} summary]\n{preview}", label=clabel)
        H.add_edge(section_id, cid, relation="contains")
        if i > 0:
            H.add_edge(f"{section_id}::chunk_{i-1}", cid, relation="next")

        if max_triples_per_chunk > 0:
            triples = extract_fn(chunk_text_str, max_triples_per_chunk)
            for j, (s, r, o) in enumerate(triples):
                tid = f"{cid}::triple_{j}"
                H.add_node(
                    tid,
                    type="triple",
                    subject=s,
                    relation=r,
                    object=o,
                    label=f"{_truncate_at_word(s, 14, '')} | {r} | {_truncate_at_word(o, 14, '')}",
                    text_preview=f"subject: {s}\nrelation: {r}\nobject: {o}",
                )
                H.add_edge(cid, tid, relation="contains_triple")

    for ref_id in cited_refs:
        ref_attrs = G.nodes[ref_id]
        if ref_id not in H:
            H.add_node(
                ref_id,
                type="reference",
                number=ref_attrs.get("number"),
                text=_truncate_at_word(ref_attrs.get("text") or "", 200, ""),
                label=f"Ref[{ref_attrs.get('number', '?')}]",
            )
        H.add_edge(section_id, ref_id, relation="cites")

    for fid in figures_in_section:
        f_attrs = G.nodes[fid]
        if fid not in H:
            cap = _truncate_at_word(f_attrs.get("caption") or "Figure", 80, "")
            H.add_node(
                fid,
                type="figure",
                caption=f_attrs.get("caption", ""),
                image_path=f_attrs.get("image_path", ""),
                page_idx=f_attrs.get("page_idx", 0),
                label=f"Fig: {cap}",
            )
        H.add_edge(section_id, fid, relation="contains")

    for tid in tables_in_section:
        t_attrs = G.nodes[tid]
        if tid not in H:
            cap = _truncate_at_word(t_attrs.get("caption") or "Table", 80, "")
            H.add_node(
                tid,
                type="table",
                caption=t_attrs.get("caption", ""),
                page_idx=t_attrs.get("page_idx", 0),
                table_body_preview=_truncate_at_word(t_attrs.get("table_body_preview") or "", 200, ""),
                label=f"Tab: {cap}",
            )
        H.add_edge(section_id, tid, relation="contains")

    return H


def _section_to_graphrag_entry(section_id: str, H: "nx.DiGraph", section_title: str) -> dict[str, Any]:
    """Extract from section internal graph H: chunks (full text) and triples for GraphRAG."""
    chunks: list[dict[str, Any]] = []
    triples: list[dict[str, Any]] = []
    for nid in H.nodes():
        attrs = H.nodes[nid]
        t = attrs.get("type", "")
        if t == "chunk":
            idx = attrs.get("index", len(chunks))
            text = attrs.get("text") or attrs.get("text_preview", "")
            chunks.append({"chunk_id": nid, "index": idx, "text": text})
        elif t == "triple":
            # nid e.g. section:1:Intro::chunk_0::triple_1 -> chunk_index=0
            chunk_index = -1
            if "::chunk_" in nid:
                try:
                    chunk_index = int(nid.split("::chunk_")[1].split("::")[0])
                except (IndexError, ValueError):
                    pass
            triples.append({
                "subject": attrs.get("subject", ""),
                "relation": attrs.get("relation", ""),
                "object": attrs.get("object", ""),
                "chunk_index": chunk_index,
            })
    chunks.sort(key=lambda x: x["index"])
    return {
        "section_id": section_id,
        "title": section_title,
        "chunks": chunks,
        "triples": triples,
    }


def export_graphrag_index(
    sections_data: list[dict[str, Any]],
    out_dir: Path,
    stem: str,
) -> Path:
    """Export GraphRAG index JSON for graphrag_query retrieval and QA."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}_graphrag_index.json"
    payload = {"doc_name": stem, "sections": sections_data}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def export_graph_json(G: "nx.DiGraph", path: Path) -> None:
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
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"nodes": nodes, "edges": edges}, f, ensure_ascii=False, indent=2)


def export_graph_graphml(G: "nx.DiGraph", path: Path) -> None:
    G2 = nx.DiGraph()
    for n in G.nodes():
        attrs = {k: str(v) if v is not None and not isinstance(v, (str, int, float, bool)) else v for k, v in G.nodes[n].items()}
        G2.add_node(n, **attrs)
    for u, v in G.edges():
        G2.add_edge(u, v, **dict(G.edges[u, v]))
    nx.write_graphml(G2, path)


# Node type colors (for visualization)
NODE_COLORS = {
    "document": "#1f77b4",
    "section": "#2ca02c",
    "figure": "#ff7f0e",
    "table": "#9467bd",
    "reference": "#d62728",
    "paragraph": "#17becf",
    "chunk": "#17becf",
    "triple": "#e377c2",
}


def _node_label(nid: str, attrs: dict) -> str:
    """Generate short node label (word-boundary truncation)."""
    t = attrs.get("type", "")
    if t == "document":
        return attrs.get("title", nid)
    if t == "section":
        return _truncate_at_word(attrs.get("title") or nid, 40, "")
    if t == "figure":
        cap = _truncate_at_word(attrs.get("caption") or "Figure", 35, "")
        return f"Fig: {cap}"
    if t == "table":
        cap = _truncate_at_word(attrs.get("caption") or "Table", 35, "")
        return f"Tab: {cap}"
    if t == "reference":
        return f"Ref[{attrs.get('number', '?')}]"
    if t == "paragraph":
        return _truncate_at_word(attrs.get("label", f"P{attrs.get('index', '')}"), 35, "")
    if t == "chunk":
        return _truncate_at_word(attrs.get("label", f"Chunk{attrs.get('index', '')}"), 35, "")
    if t == "triple":
        return _truncate_at_word(attrs.get("label", "s|r|o"), 35, "")
    return _truncate_at_word(nid, 30, "")


def _inject_click_to_open_url(html_path: Path) -> None:
    """Inject into PyVis HTML: open node url in new tab on click."""
    html_path = Path(html_path)
    if not html_path.exists():
        return
    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()
    # Skip if already injected
    if "nodeClickOpenUrl" in content:
        return
    # Bind click after network creation to avoid browser blocking window.open
    inject_after_network = """
                  network.on("click", function(params) {
                    if (params.nodes.length === 1) {
                      var n = nodes.get(params.nodes[0]);
                      if (n && n.url) { window.open(n.url, "_blank"); }
                    }
                  });
                  var nodeClickOpenUrl = true;
    """
    marker = "network = new vis.Network(container, data, options);"
    if marker in content:
        content = content.replace(marker, marker + inject_after_network, 1)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(content)
    else:
        # Legacy: inject after drawGraph() (may be blocked)
        fallback = """
    setTimeout(function() {
      if (typeof network !== "undefined" && network && typeof nodes !== "undefined" && nodes) {
        network.on("click", function(params) {
          if (params.nodes.length === 1) {
            var n = nodes.get(params.nodes[0]);
            if (n && n.url) window.open(n.url, "_blank");
          }
        });
      }
    }, 100);
    """
        if "drawGraph();" in content and "network.on(\"click\"" not in content:
            content = content.replace("drawGraph();", "drawGraph();" + fallback, 1)
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(content)


def _visualize_section_internal(H: "nx.DiGraph", path: Path) -> None:
    """Write section internal subgraph to a separate interactive HTML."""
    if not HAS_PYVIS:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    net = Network(
        directed=True,
        height="600px",
        width="100%",
        bgcolor="#f8f9fa",
        font_color="#333",
    )
    for n in H.nodes():
        attrs = H.nodes[n]
        node_type = attrs.get("type", "paragraph")
        color = NODE_COLORS.get(node_type, "#888")
        label = attrs.get("label") or _node_label(n, attrs)
        title = attrs.get("text_preview") or attrs.get("text") or str(attrs)
        net.add_node(n, label=_truncate_at_word(label, 50, ""), title=_truncate_at_word(str(title), 800, ""), color=color)
    for u, v in H.edges():
        rel = H.edges[u, v].get("relation", "")
        net.add_edge(u, v, title=rel)
    if path.suffix.lower() != ".html":
        path = path.with_suffix(".html")
    net.save_graph(str(path))


def visualize_graph(
    G: "nx.DiGraph",
    path: Path,
    *,
    use_pyvis: bool = True,
    section_internal_html: bool = True,
    main_stem: str = "",
    chunk_size: int = 300,
    chunk_overlap: int = 50,
    max_triples_per_chunk: int = 5,
    extract_triples_fn: Optional[Callable[[str, int], list[Triple]]] = None,
) -> None:
    """
    Visualize graph and write to file.
    - Section internal graphs by chunk; at most max_triples_per_chunk triples per chunk.
    - extract_triples_fn: if provided, use it (e.g. LLM); else use regex extract_triples_simple.
    """
    if nx is None:
        raise ImportError("Install networkx")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out_dir = path.parent
    stem = main_stem or path.stem.replace(".html", "").replace("_kg", "")

    if use_pyvis and HAS_PYVIS:
        section_urls: dict[str, str] = {}
        sections_data_for_graphrag: list[dict[str, Any]] = []
        if section_internal_html:
            for n in list(G.nodes()):
                if G.nodes[n].get("type") != "section":
                    continue
                H = build_section_internal_graph(
                    G,
                    n,
                    chunk_size=chunk_size,
                    overlap=chunk_overlap,
                    max_triples_per_chunk=max_triples_per_chunk,
                    extract_triples_fn=extract_triples_fn,
                )
                if H is None:
                    continue
                fname = _section_filename(n, stem)
                section_path = out_dir / fname
                _visualize_section_internal(H, section_path)
                section_urls[n] = fname  # Relative path for main graph click
                # Collect GraphRAG index: chunks + triples
                title = G.nodes[n].get("title") or n
                sections_data_for_graphrag.append(_section_to_graphrag_entry(n, H, title))
            if sections_data_for_graphrag:
                idx_path = export_graphrag_index(sections_data_for_graphrag, out_dir, stem)
                print(f"Exported GraphRAG index: {idx_path} (use graphrag_query.py for retrieval/QA)")

        net = Network(
            directed=True,
            height="700px",
            width="100%",
            bgcolor="#ffffff",
            font_color="#333",
        )
        for n in G.nodes():
            attrs = G.nodes[n]
            node_type = attrs.get("type", "")
            color = NODE_COLORS.get(node_type, "#888")
            label = _node_label(n, attrs)
            title = _truncate_at_word(str(attrs) if attrs else n, 800, "")
            node_opts = dict(label=label, title=title, color=color)
            if node_type == "section" and n in section_urls:
                node_opts["url"] = section_urls[n]
                node_opts["title"] = (title or "") + "\n\nClick node to open this section's internal knowledge graph"
            net.add_node(n, **node_opts)
        for u, v in G.edges():
            rel = G.edges[u, v].get("relation", "")
            net.add_edge(u, v, title=rel)
        ext = path.suffix.lower()
        if ext != ".html":
            path = path.with_suffix(".html")
        net.save_graph(str(path))
        _inject_click_to_open_url(path)
        print(f"Exported interactive graph: {path} (open in browser)")
        if section_urls:
            print(f"Exported {len(section_urls)} section internal graphs; click section nodes in main graph to open")
        return

    # Fallback: matplotlib static graph
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        if use_pyvis:
            print("pyvis not installed; run: pip install pyvis")
        raise SystemExit("Visualization requires pyvis or matplotlib: pip install pyvis or pip install matplotlib")
    pos = nx.spring_layout(G, k=1.5, seed=42, iterations=50)
    node_colors = [NODE_COLORS.get(G.nodes[n].get("type"), "#888") for n in G.nodes()]
    labels = {n: _node_label(n, G.nodes[n]) for n in G.nodes()}
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=400, alpha=0.9)
    nx.draw_networkx_edges(G, pos, edge_color="#999", arrows=True, arrowsize=12, alpha=0.6)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=6, font_family="sans-serif")
    plt.axis("off")
    plt.tight_layout()
    if path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".pdf", ".svg"):
        path = path.with_suffix(".png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Exported static graph: {path}")


def load_graph_from_json(path: Path) -> "nx.DiGraph":
    """Load _kg.json exported by this script as a NetworkX digraph."""
    if nx is None:
        raise ImportError("Install networkx")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    G = nx.DiGraph()
    for n in data.get("nodes", []):
        nid = n.pop("id", None)
        if nid is not None:
            G.add_node(nid, **n)
    for e in data.get("edges", []):
        u, v = e.get("source"), e.get("target")
        if u is not None and v is not None:
            G.add_edge(u, v, **{k: v for k, v in e.items() if k not in ("source", "target")})
    return G


def _get_extract_triples_fn(
    use_llm: bool,
    use_ollama: bool = False,
    ollama_model: str = "llama3.2",
) -> Optional[Callable[[str, int], list[Triple]]]:
    """Default: use LLM for extraction; use_llm=False returns None (use regex). use_ollama=True for local Ollama."""
    if not use_llm:
        return None
    if not _HAS_LLAMA:
        print("Warning: llama-index not installed; triples will be empty. Install: pip install llama-index llama-index-llms-ollama or llama-index-llms-openai")
        return make_llm_extract_triples_fn(llm=None)
    llm = None
    if use_ollama:
        try:
            from llama_index.llms.ollama import Ollama
            llm = Ollama(model=ollama_model, request_timeout=120.0)
            print(f"Using Ollama model: {ollama_model}")
        except Exception as e:
            print(f"Warning: could not create Ollama LLM ({e}). Ensure Ollama is installed and model pulled, e.g.: ollama pull {ollama_model}")
            return make_llm_extract_triples_fn(llm=None)
    if llm is None:
        try:
            llm = getattr(Settings, "llm", None)
        except Exception:
            llm = None
        if llm is None:
            try:
                from llama_index.llms.openai import OpenAI
                llm = OpenAI(temperature=0.0, model="gpt-3.5-turbo")
            except Exception as e:
                print(f"Warning: could not create OpenAI LLM ({e}); triples will be empty. Set OPENAI_API_KEY, use --ollama, or Settings.llm")
                return make_llm_extract_triples_fn(llm=None)
    return make_llm_extract_triples_fn(llm=llm)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Build knowledge graph from MinerU content_list, or visualize existing _kg.json")
    parser.add_argument("input", type=Path, nargs="?", help="content_list.json or existing *_kg.json")
    parser.add_argument("-o", "--output-dir", type=Path, default=None, help="Output directory (or output file path when visualizing)")
    parser.add_argument("--doc-name", type=str, default=None, help="Document name")
    parser.add_argument("--no-internal", action="store_true", help="Do not build section internal paragraph summaries")
    parser.add_argument("--graphml", action="store_true", help="Also export GraphML")
    parser.add_argument("-v", "--visualize", action="store_true", help="Visualize after build or visualize existing JSON to HTML/PNG")
    parser.add_argument("--static", action="store_true", help="Use matplotlib static graph (default: pyvis interactive HTML)")
    parser.add_argument("--chunk-size", type=int, default=300, help="Section internal chunk size in chars (default 300)")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="Overlap between adjacent chunks (default 50)")
    parser.add_argument("--max-triples-per-chunk", type=int, default=5, help="Max triples per chunk (default 5; 0 = no triples)")
    parser.add_argument("--no-llm", action="store_true", help="Do not use LLM; use regex for triples (default: LLM)")
    parser.add_argument("--ollama", action="store_true", help="Use local Ollama model for triples (run ollama serve and pull model first)")
    parser.add_argument("--ollama-model", type=str, default="llama3.2", help="Ollama model name (default llama3.2), e.g. llama3.1:8b, qwen2.5:7b")
    args = parser.parse_args()

    path = args.input
    if not path or not path.exists():
        parser.print_help()
        raise SystemExit("Provide a valid input: content_list.json or *_kg.json")

    # Visualization only: input is *_kg.json
    if "_kg.json" in path.name and path.suffix.lower() == ".json":
        out_path = Path(args.output_dir) if args.output_dir else path.parent
        if out_path.suffix.lower() in (".html", ".png", ".svg", ".pdf"):
            viz_path = out_path
        else:
            viz_path = (out_path / path.stem.replace("_kg", "")).with_suffix(".html")
        G = load_graph_from_json(path)
        stem = path.stem.replace("_kg", "")
        extract_fn = _get_extract_triples_fn(
            use_llm=not args.no_llm,
            use_ollama=args.ollama,
            ollama_model=args.ollama_model,
        )
        visualize_graph(
            G,
            viz_path,
            use_pyvis=not args.static,
            section_internal_html=not args.no_internal,
            main_stem=stem,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            max_triples_per_chunk=args.max_triples_per_chunk,
            extract_triples_fn=extract_fn,
        )
        return

    # Build from content_list
    out_dir = args.output_dir or path.parent
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    doc_name = args.doc_name or path.stem.replace("_content_list", "").replace("_content_list_v2", "")

    content_list = _load_content_list(path)
    G = build_graph_from_content_list(content_list, doc_name=doc_name, include_internal_subgraph=not args.no_internal)

    base = out_dir / f"{doc_name}_kg"
    export_graph_json(G, base.with_suffix(".json"))
    print(f"Exported: {base}.json (nodes={G.number_of_nodes()}, edges={G.number_of_edges()})")
    if args.graphml:
        export_graph_graphml(G, base.with_suffix(".graphml"))
        print(f"Exported: {base}.graphml")
    if args.visualize:
        extract_fn = _get_extract_triples_fn(
            use_llm=not args.no_llm,
            use_ollama=args.ollama,
            ollama_model=args.ollama_model,
        )
        visualize_graph(
            G,
            base.with_suffix(".html"),
            use_pyvis=not args.static,
            section_internal_html=not args.no_internal,
            main_stem=doc_name,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            max_triples_per_chunk=args.max_triples_per_chunk,
            extract_triples_fn=extract_fn,
        )


if __name__ == "__main__":
    main()
