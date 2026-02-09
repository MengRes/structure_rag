"""
Merge same Figure (e.g. Fig. 1 subfigures 1a/1b/1c) into one image; update schema figures and image_path.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, List, Optional, Tuple


def _norm(s: str) -> str:
    return " ".join((s or "").split()).strip()


def _figure_number_from_caption(caption: str) -> Optional[int]:
    """Parse main figure number from caption: Fig. 1 / Figure 1 / Fig. 1a -> 1."""
    c = _norm(caption)
    if not c:
        return None
    m = re.search(r"(?i)Fig(?:ure)?\.?\s*(\d+)[a-z]?", c)
    if m:
        return int(m.group(1))
    return None


def _is_subfigure_caption(caption: str) -> bool:
    """Whether caption is a subfigure label, e.g. (a) ... (b) ..."""
    c = _norm(caption)
    return bool(re.match(r"^\s*\([a-z]\)\s*", c))


def _group_figures_in_section(figures: List[dict]) -> List[List[dict]]:
    """
    Group figures in a section by same Figure number.
    Rule: caption with Fig. N / Figure N starts new group when N differs; (a)(b)(c) or other unnumbered captions go to current group.
    """
    if not figures:
        return []
    groups: List[List[dict]] = []
    current: List[dict] = []
    current_num: Optional[int] = None

    for fig in figures:
        cap = fig.get("caption") or ""
        num = _figure_number_from_caption(cap)
        is_sub = _is_subfigure_caption(cap)

        if num is not None and not is_sub and num != current_num:
            if current:
                groups.append(current)
            current = [fig]
            current_num = num
        elif current_num is not None or current:
            # Subfigure (a)(b)(c), or same Fig. N as current, or current group already has items -> append to current (e.g. "Middle Ring" sub-labels)
            current.append(fig)
            if num is not None and current_num is None:
                current_num = num
        else:
            current = [fig]
            current_num = num
    if current:
        groups.append(current)
    return groups


def _copy_single_image(
    image_path: str,
    base_path: Path,
    out_path: Path,
) -> bool:
    """Copy single image to output_dir so HTML under graphrag_all can load it."""
    if not image_path:
        return False
    src = base_path / image_path
    if not src.exists():
        return False
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(src, out_path)
        return True
    except Exception:
        return False


def _merge_images(
    image_paths: List[str],
    base_path: Path,
    out_path: Path,
    max_width: int = 1200,
    gap: int = 8,
) -> bool:
    """Stitch multiple images vertically into one and write to out_path."""
    try:
        from PIL import Image
    except ImportError:
        return False
    imgs: List[Image.Image] = []
    for rel in image_paths:
        if not rel:
            continue
        p = base_path / rel
        if not p.exists():
            continue
        try:
            im = Image.open(p).convert("RGB")
            imgs.append(im)
        except Exception:
            continue
    if not imgs:
        return False
    # Uniform width, scale proportionally
    w = min(max_width, max(im.width for im in imgs))
    scaled = []
    for im in imgs:
        if im.width != w:
            r = w / im.width
            h = int(im.height * r)
            im = im.resize((w, h), Image.Resampling.LANCZOS)
        scaled.append(im)
    total_h = sum(im.height for im in scaled) + gap * (len(scaled) - 1)
    out = Image.new("RGB", (w, total_h), (255, 255, 255))
    y = 0
    for im in scaled:
        out.paste(im, (0, y))
        y += im.height + gap
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path, "PNG", quality=95)
    return True


def merge_figure_groups(
    schema: dict[str, Any],
    content_list_base_path: Path,
    output_dir: Path,
    paper_id: str,
) -> dict[str, Any]:
    """
    Merge multiple subfigures of same Figure in schema into one image; update each section's figures and image_path.
    - content_list_base_path: directory of content_list (image paths relative to it).
    - output_dir: directory for merged images; image_path stored as figures/{paper_id}_fig_{N}_merged.png (relative to output_dir).
    Returns modified schema (modified in place too).
    """
    base = Path(content_list_base_path)
    out_dir = Path(output_dir)
    paper_id = (paper_id or schema.get("paper_id", "paper")).strip()
    fig_counter = 0

    for sec in schema.get("sections") or []:
        figures = sec.get("figures") or []
        if not figures:
            continue
        groups = _group_figures_in_section(figures)
        new_figures: List[dict] = []
        for group in groups:
            if len(group) <= 1:
                fig_counter += 1
                one = dict(group[0])
                one["id"] = f"fig_{fig_counter}"
                # Single image also copied to output_dir/figures/ for unified loading in graph page
                old_path = one.get("image_path") or one.get("img_path") or ""
                if old_path and not old_path.startswith("figures/"):
                    suffix = Path(old_path).suffix or ".png"
                    rel_out = f"figures/{paper_id}_fig_{fig_counter}{suffix}"
                    abs_out = out_dir / rel_out
                    if _copy_single_image(old_path, base, abs_out):
                        one["image_path"] = rel_out
                new_figures.append(one)
                continue
            paths = [f.get("image_path") or f.get("img_path") or "" for f in group]
            paths = [p for p in paths if p]
            if not paths:
                fig_counter += 1
                one = dict(group[0])
                one["id"] = f"fig_{fig_counter}"
                new_figures.append(one)
                continue
            fig_num = _figure_number_from_caption((group[0].get("caption") or "")) or fig_counter
            rel_out = f"figures/{paper_id}_fig_{fig_num}_merged.png"
            abs_out = out_dir / rel_out
            if _merge_images(paths, base, abs_out):
                fig_counter += 1
                new_figures.append({
                    "id": f"fig_{fig_counter}",
                    "caption": _norm(group[0].get("caption") or ""),
                    "image_path": rel_out,
                    "page_idx": group[0].get("page_idx", 0),
                    "preceding_paragraph_index": group[0].get("preceding_paragraph_index"),
                })
            else:
                for f in group:
                    fig_counter += 1
                    one = dict(f)
                    one["id"] = f"fig_{fig_counter}"
                    new_figures.append(one)
        sec["figures"] = new_figures
    return schema
