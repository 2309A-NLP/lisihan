# -*- coding: utf-8 -*-
"""工单编号：人工智能 NLP-RAG-图像内容解析及检索优化。

本文件属于 PDF 招股说明书智能问答系统，用于保留工单一到工单四的文本检索、
结构化问答、负向问题处理、图片内容解析和检索优化能力。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

try:
    import fitz
except Exception:  # pragma: no cover - 可选运行依赖
    fitz = None


MAIN_CONTENT_RECT = (50.0, 100.0, 550.0, 700.0)
MIN_TABLE_AREA = 3000.0
TABLE_MERGE_GAP = 28.0
TABLE_CONTINUATION_TOP = 170.0
TABLE_CONTINUATION_BOTTOM_GAP = 150.0


def _resolve_output_path(pdf_path: Path, output: str | None) -> Path:
    default_dir = Path("parsed_output")
    if not output:
        default_dir.mkdir(parents=True, exist_ok=True)
        return default_dir / f"{pdf_path.stem}_chunks.json"

    output_path = Path(output)
    if output_path.suffix.lower() == ".json":
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

    output_path.mkdir(parents=True, exist_ok=True)
    return output_path / f"{pdf_path.stem}_chunks.json"


def _safe_stem(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def _rect_to_list(rect) -> List[float]:
    return [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)]


def _rect_key(rect: "fitz.Rect") -> tuple[int, int, int, int]:
    return (round(rect.x0 * 10), round(rect.y0 * 10), round(rect.x1 * 10), round(rect.y1 * 10))


def _list_to_rect(bbox: Iterable[float]) -> "fitz.Rect":
    values = list(bbox or [0, 0, 0, 0])
    return fitz.Rect(values[0], values[1], values[2], values[3])


def _expand_rect(rect: "fitz.Rect", page_rect: "fitz.Rect", margin: float = 8.0) -> "fitz.Rect":
    expanded = fitz.Rect(rect.x0 - margin, rect.y0 - margin, rect.x1 + margin, rect.y1 + margin)
    return expanded & page_rect


def _rect_area(rect: "fitz.Rect") -> float:
    return max(0.0, float(rect.width)) * max(0.0, float(rect.height))


def _main_content_rect(page_rect: "fitz.Rect") -> "fitz.Rect":
    x0, y0, x1, y1 = MAIN_CONTENT_RECT
    return fitz.Rect(
        max(page_rect.x0, x0),
        max(page_rect.y0, y0),
        min(page_rect.x1, x1),
        min(page_rect.y1, y1),
    )


def _table_content_rect(page_rect: "fitz.Rect") -> "fitz.Rect":
    return fitz.Rect(
        max(page_rect.x0, 35.0),
        max(page_rect.y0, 58.0),
        min(page_rect.x1, page_rect.x1 - 35.0),
        min(page_rect.y1, page_rect.y1 - 48.0),
    )


def _clip_to_main_content(rect: "fitz.Rect", page_rect: "fitz.Rect") -> "fitz.Rect":
    return rect & _main_content_rect(page_rect)


def _clip_to_table_content(rect: "fitz.Rect", page_rect: "fitz.Rect") -> "fitz.Rect":
    return rect & _table_content_rect(page_rect)


def _is_in_main_content(rect: "fitz.Rect", page_rect: "fitz.Rect", min_ratio: float = 0.65) -> bool:
    if _rect_area(rect) <= 0:
        return False
    return _rect_area(rect & _main_content_rect(page_rect)) / _rect_area(rect) >= min_ratio


def _overlap_ratio(a: "fitz.Rect", b: "fitz.Rect") -> float:
    intersection = a & b
    area = min(_rect_area(a), _rect_area(b))
    if area <= 0:
        return 0.0
    return _rect_area(intersection) / area


def _dedupe_rects(rects: List["fitz.Rect"], overlap_threshold: float = 0.85) -> List["fitz.Rect"]:
    unique: List[fitz.Rect] = []
    for rect in sorted(rects, key=lambda item: _rect_area(item), reverse=True):
        if any(_overlap_ratio(rect, existing) >= overlap_threshold for existing in unique):
            continue
        unique.append(rect)
    return sorted(unique, key=lambda item: (item.y0, item.x0))


def _union_rects(rects: List["fitz.Rect"]) -> "fitz.Rect":
    if not rects:
        return fitz.Rect()
    union = fitz.Rect(rects[0])
    for rect in rects[1:]:
        union |= rect
    return union


def _rect_horizontal_overlap_ratio(a: "fitz.Rect", b: "fitz.Rect") -> float:
    overlap = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
    return overlap / max(1.0, min(a.width, b.width))


def _rect_vertical_overlap_ratio(a: "fitz.Rect", b: "fitz.Rect") -> float:
    overlap = max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))
    return overlap / max(1.0, min(a.height, b.height))


def _rect_vertical_gap(a: "fitz.Rect", b: "fitz.Rect") -> float:
    if b.y0 >= a.y1:
        return b.y0 - a.y1
    if a.y0 >= b.y1:
        return a.y0 - b.y1
    return 0.0


def _can_merge_table_rect(a: "fitz.Rect", b: "fitz.Rect") -> bool:
    if _overlap_ratio(a, b) >= 0.25:
        return True
    same_column = _rect_horizontal_overlap_ratio(a, b) >= 0.55
    dynamic_gap = min(TABLE_MERGE_GAP, max(10.0, min(a.height, b.height) * 0.12))
    close_vertically = _rect_vertical_gap(a, b) <= dynamic_gap
    return same_column and close_vertically


def _merge_related_table_rects(rects: List["fitz.Rect"]) -> List["fitz.Rect"]:
    pending = sorted([rect for rect in rects if _rect_area(rect) >= MIN_TABLE_AREA], key=lambda item: (item.y0, item.x0))
    groups: List[List[fitz.Rect]] = []

    for rect in pending:
        merged = False
        for group in groups:
            union = _union_rects(group)
            if _can_merge_table_rect(union, rect):
                group.append(rect)
                merged = True
                break
        if not merged:
            groups.append([rect])

    changed = True
    while changed:
        changed = False
        merged_groups: List[List[fitz.Rect]] = []
        for group in groups:
            union = _union_rects(group)
            target = None
            for existing in merged_groups:
                if _can_merge_table_rect(_union_rects(existing), union):
                    target = existing
                    break
            if target is None:
                merged_groups.append(list(group))
            else:
                target.extend(group)
                changed = True
        groups = merged_groups

    return sorted([_union_rects(group) for group in groups], key=lambda item: (item.y0, item.x0))


def _rect_looks_like_cross_page_start(rect: "fitz.Rect", page_rect: "fitz.Rect") -> bool:
    return (page_rect.y1 - rect.y1) <= TABLE_CONTINUATION_BOTTOM_GAP


def _rect_looks_like_cross_page_continuation(rect: "fitz.Rect", page_rect: "fitz.Rect") -> bool:
    return rect.y0 <= TABLE_CONTINUATION_TOP


def _can_stitch_table_rects(prev_rect: "fitz.Rect", next_rect: "fitz.Rect", prev_page_rect: "fitz.Rect", next_page_rect: "fitz.Rect") -> bool:
    if not _rect_looks_like_cross_page_start(prev_rect, prev_page_rect):
        return False
    if not _rect_looks_like_cross_page_continuation(next_rect, next_page_rect):
        return False
    return _rect_horizontal_overlap_ratio(prev_rect, next_rect) >= 0.45
