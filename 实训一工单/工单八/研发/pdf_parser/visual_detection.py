# -*- coding: utf-8 -*-
"""工单编号：人工智能 NLP-RAG-图像内容解析及检索优化。

本文件属于 PDF 招股说明书智能问答系统，用于保留工单一到工单四的文本检索、
结构化问答、负向问题处理、图片内容解析和检索优化能力。
"""

from __future__ import annotations

import math
from typing import Dict, List

try:
    import fitz
except Exception:  # pragma: no cover - 可选运行依赖
    fitz = None

from .visual_geometry import (
    TABLE_CONTINUATION_BOTTOM_GAP,
    _can_merge_table_rect,
    _can_stitch_table_rects,
    _clip_to_main_content,
    _clip_to_table_content,
    _dedupe_rects,
    _expand_rect,
    _is_in_main_content,
    _list_to_rect,
    _merge_related_table_rects,
    _overlap_ratio,
    _rect_area,
    _rect_horizontal_overlap_ratio,
    _rect_key,
    _rect_looks_like_cross_page_continuation,
    _rect_looks_like_cross_page_start,
    _rect_vertical_overlap_ratio,
    _table_content_rect,
    _union_rects,
)


CHART_TITLE_MARKERS = ("图", "结构图", "增长", "应用结构", "市场结构", "流程图", "示意图")
MIN_IMAGE_AREA = 12000.0
MIN_TABLE_AREA = 3000.0
TABLE_LINE_SEARCH_MARGIN = 42.0
TABLE_RENDER_MARGIN = 14.0
TABLE_LINE_CONNECT_TOLERANCE = 16.0
MAX_DUPLICATE_IMAGE_COUNT = 3


def _drawing_rects_from_page(page: "fitz.Page") -> List["fitz.Rect"]:
    rects: List[fitz.Rect] = []
    try:
        drawings = page.get_drawings()
    except Exception:
        return rects

    for drawing in drawings:
        rect = drawing.get("rect")
        if rect is None:
            continue
        rect = fitz.Rect(rect)
        if rect.width <= 0 and rect.height <= 0:
            continue

        is_line = rect.width <= 2.5 or rect.height <= 2.5
        is_table_sized = rect.width >= 18 and rect.height >= 18
        if is_line or is_table_sized:
            rects.append(rect)
    return rects


def _rect_distance(a: "fitz.Rect", b: "fitz.Rect") -> float:
    if _rect_area(a & b) > 0:
        return 0.0
    dx = max(a.x0 - b.x1, b.x0 - a.x1, 0.0)
    dy = max(a.y0 - b.y1, b.y0 - a.y1, 0.0)
    return math.hypot(dx, dy)


def _is_table_line_candidate(rect: "fitz.Rect", seed: "fitz.Rect", page_rect: "fitz.Rect") -> bool:
    if rect.width <= 0 and rect.height <= 0:
        return False
    table_area = _table_content_rect(page_rect)
    center = fitz.Point((rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2)
    if center not in table_area:
        return False

    thin_line = rect.width <= 2.5 or rect.height <= 2.5
    table_border = rect.width >= 18 and rect.height >= 18
    return thin_line or table_border


def _is_in_table_content(rect: "fitz.Rect", page_rect: "fitz.Rect", min_ratio: float = 0.50) -> bool:
    if _rect_area(rect) <= 0:
        return False
    return _rect_area(rect & _table_content_rect(page_rect)) / _rect_area(rect) >= min_ratio


def _connected_table_line_rects(seed: "fitz.Rect", page: "fitz.Page") -> List["fitz.Rect"]:
    candidates = [
        rect
        for rect in _drawing_rects_from_page(page)
        if _is_table_line_candidate(rect, seed, page.rect)
    ]
    selected = [seed]
    selected_keys = {_rect_key(seed)}
    changed = True

    while changed:
        changed = False
        current_union = _expand_rect(_union_rects(selected), page.rect, margin=TABLE_LINE_CONNECT_TOLERANCE)
        for rect in candidates:
            rect_key = _rect_key(rect)
            if rect_key in selected_keys:
                continue
            touches_current = _rect_area(rect & current_union) > 0 or _rect_distance(rect, current_union) <= TABLE_LINE_CONNECT_TOLERANCE
            shares_table_axis = (
                _rect_horizontal_overlap_ratio(rect, current_union) >= 0.10
                or _rect_horizontal_overlap_ratio(current_union, rect) >= 0.10
                or _rect_vertical_overlap_ratio(rect, current_union) >= 0.10
                or _rect_vertical_overlap_ratio(current_union, rect) >= 0.10
            )
            if touches_current and shares_table_axis:
                selected.append(rect)
                selected_keys.add(rect_key)
                changed = True

    return selected


def _expand_table_rect_to_complete_region(rect: "fitz.Rect", page: "fitz.Page") -> "fitz.Rect":
    seed = _clip_to_table_content(rect, page.rect)
    related = _connected_table_line_rects(seed, page)
    expanded = _union_rects(related)
    expanded = _expand_rect(expanded, page.rect, margin=TABLE_RENDER_MARGIN)
    return _clip_to_table_content(expanded, page.rect)


def _text_near_rect(text_blocks: List[Dict], page: int, rect: "fitz.Rect", *, above_only: bool = True) -> str:
    candidates: List[tuple[float, str]] = []
    for block in text_blocks:
        if int(block.get("page", 0) or 0) != page:
            continue
        content = str(block.get("content", "")).strip()
        if not content:
            continue
        bbox = _list_to_rect(block.get("bbox", [0, 0, 0, 0]))
        horizontal_overlap = max(0.0, min(rect.x1, bbox.x1) - max(rect.x0, bbox.x0))
        overlap_ratio = horizontal_overlap / max(1.0, min(rect.width, bbox.width))
        if above_only:
            distance = rect.y0 - bbox.y1
            if distance < -5 or distance > 90 or overlap_ratio < 0.15:
                continue
        else:
            distance = min(abs(rect.y0 - bbox.y1), abs(bbox.y0 - rect.y1))
            if distance > 90 or overlap_ratio < 0.15:
                continue
        candidates.append((distance, content.replace("\n", " ")))
    if not candidates:
        return ""
    candidates.sort(key=lambda item: (abs(item[0]), len(item[1])))
    return candidates[0][1][:120]


def _guess_title(text_blocks: List[Dict], page: int, rect: "fitz.Rect", fallback: str) -> str:
    nearby = _text_near_rect(text_blocks, page, rect)
    if nearby:
        return nearby
    return fallback


def _repeated_image_xrefs(doc: "fitz.Document", threshold: int = MAX_DUPLICATE_IMAGE_COUNT) -> set[int]:
    counts: Dict[int, int] = {}
    for page in doc:
        seen_on_page = set()
        for image_info in page.get_images(full=True):
            xref = int(image_info[0])
            if xref in seen_on_page:
                continue
            seen_on_page.add(xref)
            counts[xref] = counts.get(xref, 0) + 1
    return {xref for xref, count in counts.items() if count > threshold}


def _image_rects_from_page(page: "fitz.Page", repeated_xrefs: set[int] | None = None) -> List["fitz.Rect"]:
    rects: List[fitz.Rect] = []
    repeated_xrefs = repeated_xrefs or set()

    try:
        for info in page.get_image_info(xrefs=True):
            xref = int(info.get("xref", 0) or 0)
            if xref in repeated_xrefs:
                continue
            bbox = info.get("bbox")
            if not bbox:
                continue
            rect = fitz.Rect(bbox)
            if _rect_area(rect) >= MIN_IMAGE_AREA and _is_in_main_content(rect, page.rect):
                rects.append(rect)
    except Exception:
        pass

    try:
        seen_xrefs = set()
        for image_info in page.get_images(full=True):
            xref = image_info[0]
            if xref in repeated_xrefs:
                continue
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)
            for rect in page.get_image_rects(xref):
                rect = fitz.Rect(rect)
                if _rect_area(rect) >= MIN_IMAGE_AREA and _is_in_main_content(rect, page.rect):
                    rects.append(rect)
    except Exception:
        pass

    return _dedupe_rects(rects)


def _nearby_image_rects(page: "fitz.Page", base_rect: "fitz.Rect", repeated_xrefs: set[int] | None = None) -> List["fitz.Rect"]:
    candidates = []
    search_rect = _expand_rect(base_rect, page.rect, margin=45.0)
    for rect in _image_rects_from_page(page, repeated_xrefs):
        if _rect_area(rect & search_rect) > 0:
            candidates.append(rect)
    return candidates


def _chart_rects_from_text(text_blocks: List[Dict], page_number: int, page_rect: "fitz.Rect") -> List["fitz.Rect"]:
    rects: List[fitz.Rect] = []
    for block in text_blocks:
        if int(block.get("page", 0) or 0) != page_number:
            continue
        content = str(block.get("content", ""))
        if not any(marker in content for marker in CHART_TITLE_MARKERS):
            continue
        bbox = _list_to_rect(block.get("bbox", [0, 0, 0, 0]))
        if "如下图" in content or "结构图" in content or "增长" in content or "应用结构" in content:
            rects.append(fitz.Rect(page_rect.x0 + 35, min(page_rect.y1 - 80, bbox.y1 + 4), page_rect.x1 - 35, page_rect.y1 - 45))
    return _dedupe_rects([rect for rect in rects if _rect_area(rect) >= 10000])


def _table_visual_rects(table_blocks: List[Dict]) -> Dict[int, List[Dict]]:
    by_page: Dict[int, List[Dict]] = {}
    for block in table_blocks:
        if not _is_likely_real_table_block(block):
            continue
        page = int(block.get("page", 0) or 0)
        if page <= 0:
            continue
        by_page.setdefault(page, []).append(block)
    return by_page


def _is_likely_real_table_block(block: Dict) -> bool:
    metadata = block.get("metadata", {}) or {}
    row_count = int(metadata.get("row_count", 0) or 0)
    column_count = int(metadata.get("column_count", 0) or 0)
    if row_count < 2 or column_count < 2:
        return False

    content = block.get("content", "")
    if isinstance(content, list):
        non_empty_cells = sum(1 for row in content for cell in (row or []) if str(cell).strip())
        return non_empty_cells >= 4

    return len(str(content).strip()) >= 8


def _build_page_table_groups(table_blocks: List[Dict], doc: "fitz.Document") -> Dict[int, List[Dict]]:
    by_page = _table_visual_rects(table_blocks)
    result: Dict[int, List[Dict]] = {}

    for page_number, blocks in by_page.items():
        if page_number < 1 or page_number > len(doc):
            continue
        page = doc[page_number - 1]
        source_rects: List[tuple[Dict, fitz.Rect]] = []
        for block in blocks:
            raw_rect = _clip_to_table_content(
                _expand_rect(_list_to_rect(block.get("bbox", [0, 0, 0, 0])), page.rect, margin=TABLE_RENDER_MARGIN),
                page.rect,
            )
            rect = _expand_table_rect_to_complete_region(raw_rect, page)
            if _rect_area(rect) >= MIN_TABLE_AREA:
                source_rects.append((block, rect))

        merged_rects = _merge_related_table_rects([rect for _, rect in source_rects])
        merged_rects = [_expand_table_rect_to_complete_region(rect, page) for rect in merged_rects]
        groups: List[Dict] = []
        for rect in merged_rects:
            related_blocks = [block for block, block_rect in source_rects if _overlap_ratio(rect, block_rect) >= 0.20]
            if not related_blocks:
                related_blocks = [blocks[0]] if blocks else []
            groups.append({"rect": rect, "blocks": related_blocks})
        result[page_number] = groups

    return result


def _stitch_cross_page_table_groups(page_groups: Dict[int, List[Dict]], doc: "fitz.Document") -> List[List[Dict]]:
    stitched: List[List[Dict]] = []
    consumed: set[tuple[int, int]] = set()

    for page_number in sorted(page_groups):
        groups = page_groups.get(page_number, [])
        for group_index, group in enumerate(groups):
            key = (page_number, group_index)
            if key in consumed:
                continue

            chain = [{"page": page_number, **group}]
            consumed.add(key)
            current_page = page_number
            current_rect = group["rect"]

            while current_page + 1 in page_groups:
                next_page = current_page + 1
                current_page_rect = doc[current_page - 1].rect
                next_page_rect = doc[next_page - 1].rect
                best_index = None
                best_group = None
                for next_index, candidate in enumerate(page_groups[next_page]):
                    candidate_key = (next_page, next_index)
                    if candidate_key in consumed:
                        continue
                    candidate_rect = candidate["rect"]
                    if _can_stitch_table_rects(current_rect, candidate_rect, current_page_rect, next_page_rect):
                        best_index = next_index
                        best_group = candidate
                        break

                if best_group is None or best_index is None:
                    break

                chain.append({"page": next_page, **best_group})
                consumed.add((next_page, best_index))
                current_page = next_page
                current_rect = best_group["rect"]

            stitched.append(chain)

    return stitched
