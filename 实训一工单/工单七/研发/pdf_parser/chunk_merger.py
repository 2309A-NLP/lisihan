# -*- coding: utf-8 -*-
"""工单编号：人工智能 NLP-RAG-图像内容解析及检索优化。

本文件属于 PDF 招股说明书智能问答系统，保留工单一到工单四的文本检索、
结构化问答、负向问题处理、图片内容解析和检索优化能力。
"""

from __future__ import annotations

import copy
import re
from typing import Dict, List, Sequence

from .filters import normalize_text

SENTENCE_END_PATTERN = re.compile(r"[。！？；：\.\?\!;:]$")


def is_complete_sentence(text: str) -> bool:
    """Return True when the text ends with a sentence-level terminator."""
    return bool(SENTENCE_END_PATTERN.search(normalize_text(text)))


def _bbox_area(bbox: Sequence[float]) -> float:
    x0, y0, x1, y1 = bbox
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _intersection_area(a: Sequence[float], b: Sequence[float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    x0 = max(ax0, bx0)
    y0 = max(ay0, by0)
    x1 = min(ax1, bx1)
    y1 = min(ay1, by1)
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _intersects(a: Sequence[float], b: Sequence[float], ratio: float = 0.35) -> bool:
    base = min(_bbox_area(a), _bbox_area(b))
    if base <= 0:
        return False
    return _intersection_area(a, b) / base >= ratio


def _bbox_union(a: Sequence[float], b: Sequence[float]) -> List[float]:
    """Return the union bbox of two boxes."""
    return [
        min(float(a[0]), float(b[0])),
        min(float(a[1]), float(b[1])),
        max(float(a[2]), float(b[2])),
        max(float(a[3]), float(b[3])),
    ]


def _char_count(content: object) -> int:
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        return sum(len(str(cell)) for row in content for cell in row)
    return len(str(content))


def _append_text(left: str, right: str) -> str:
    """Join two text fragments without introducing unwanted spaces in Chinese."""
    left = normalize_text(left)
    right = normalize_text(right)
    if not left:
        return right
    if not right:
        return left
    if re.search(r"[\w）)]$", left) and re.search(r"^[A-Za-z0-9(（]", right):
        return f"{left} {right}"
    return f"{left}{right}"


def _can_merge_split_blocks(previous: Dict, current: Dict) -> bool:
    """Check whether two neighboring blocks match the PyMuPDF split-fragment rules."""
    if previous.get("type") != "text" or current.get("type") != "text":
        return False
    if int(previous.get("page", 0)) != int(current.get("page", 0)):
        return False

    previous_text = str(previous.get("content", ""))
    current_text = str(current.get("content", ""))
    if not previous_text or not current_text or is_complete_sentence(previous_text):
        return False

    previous_bbox = previous.get("bbox") or [0, 0, 0, 0]
    current_bbox = current.get("bbox") or [0, 0, 0, 0]
    vertical_distance = abs(float(current_bbox[1]) - float(previous_bbox[3]))
    horizontal_distance = abs(float(current_bbox[0]) - float(previous_bbox[0]))
    return vertical_distance < 18 and horizontal_distance < 30


def merge_split_blocks(blocks: List[Dict]) -> List[Dict]:
    """Merge adjacent text blocks that are fragments of the same natural sentence."""
    sorted_blocks = sorted(
        blocks,
        key=lambda block: (
            int(block.get("page", 0)),
            float((block.get("bbox") or [0, 0, 0, 0])[1]),
            float((block.get("bbox") or [0, 0, 0, 0])[0]),
            0 if block.get("type") == "text" else 1,
        ),
    )

    merged: List[Dict] = []
    for block in sorted_blocks:
        current = copy.deepcopy(block)
        current.setdefault("metadata", {})

        if merged and _can_merge_split_blocks(merged[-1], current):
            previous = merged[-1]
            previous["content"] = _append_text(str(previous.get("content", "")), str(current.get("content", "")))
            previous["bbox"] = _bbox_union(previous.get("bbox") or [0, 0, 0, 0], current.get("bbox") or [0, 0, 0, 0])
            previous_metadata = dict(previous.get("metadata", {}))
            previous_metadata["char_count"] = _char_count(previous["content"])
            previous_metadata["merged_from"] = int(previous_metadata.get("merged_from", 1)) + int(
                current.get("metadata", {}).get("merged_from", 1)
            )
            previous["metadata"] = previous_metadata
            continue

        current_metadata = dict(current.get("metadata", {}))
        current_metadata.setdefault("char_count", _char_count(current.get("content", "")))
        current_metadata.setdefault("merged_from", 1)
        current["metadata"] = current_metadata
        merged.append(current)

    return merged


def _split_text_by_sentence(text: str, max_chars: int) -> List[str]:
    """Split long text on sentence boundaries, falling back to hard slices."""
    text = normalize_text(text)
    if len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    current = ""
    parts = re.split(r"(?<=[。！？；：\.\?\!;:])", text)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if len(current) + len(part) <= max_chars:
            current += part
            continue
        if current:
            chunks.append(current)
        if len(part) > max_chars:
            chunks.extend(part[i : i + max_chars] for i in range(0, len(part), max_chars))
            current = ""
        else:
            current = part
    if current:
        chunks.append(current)
    return [chunk for chunk in chunks if chunk]


def _can_merge_paragraph(previous: Dict, current: Dict) -> bool:
    if previous.get("type") != "text" or current.get("type") != "text":
        return False
    if int(previous.get("page", 0)) != int(current.get("page", 0)):
        return False
    previous_bbox = previous.get("bbox") or [0, 0, 0, 0]
    current_bbox = current.get("bbox") or [0, 0, 0, 0]
    vertical_gap = float(current_bbox[1]) - float(previous_bbox[3])
    horizontal_distance = abs(float(current_bbox[0]) - float(previous_bbox[0]))
    return 0 <= vertical_gap < 28 and horizontal_distance < 45


def merge_into_paragraphs(blocks: List[Dict], max_chars: int = 1500) -> List[Dict]:
    """Aggregate nearby text blocks into paragraph chunks and split oversized text."""
    paragraphs: List[Dict] = []
    for block in merge_split_blocks(blocks):
        current = copy.deepcopy(block)
        current.setdefault("metadata", {})

        if paragraphs and _can_merge_paragraph(paragraphs[-1], current):
            previous = paragraphs[-1]
            previous["content"] = f"{normalize_text(str(previous.get('content', '')))}\n{normalize_text(str(current.get('content', '')))}"
            previous["bbox"] = _bbox_union(previous.get("bbox") or [0, 0, 0, 0], current.get("bbox") or [0, 0, 0, 0])
            previous_metadata = dict(previous.get("metadata", {}))
            previous_metadata["char_count"] = _char_count(previous["content"])
            previous_metadata["merged_from"] = int(previous_metadata.get("merged_from", 1)) + int(
                current.get("metadata", {}).get("merged_from", 1)
            )
            previous["metadata"] = previous_metadata
            continue

        current_metadata = dict(current.get("metadata", {}))
        current_metadata["char_count"] = _char_count(current.get("content", ""))
        current_metadata.setdefault("merged_from", 1)
        current["metadata"] = current_metadata
        paragraphs.append(current)

    output: List[Dict] = []
    for block in paragraphs:
        content = block.get("content", "")
        if block.get("type") != "text" or not isinstance(content, str) or len(content) <= max_chars:
            output.append(block)
            continue

        for index, segment in enumerate(_split_text_by_sentence(content, max_chars), start=1):
            segment_block = copy.deepcopy(block)
            segment_block["content"] = segment
            segment_block["metadata"] = dict(segment_block.get("metadata", {}))
            segment_block["metadata"]["char_count"] = len(segment)
            segment_block["metadata"]["split_part"] = index
            output.append(segment_block)

    return output


def _convert_table_content(content: object, table_format: str) -> object:
    if table_format == "raw":
        return content
    if isinstance(content, str):
        return content

    rows = content if isinstance(content, list) else []
    if table_format == "csv":
        return "\n".join(",".join(cell for cell in row) for row in rows)

    lines = []
    if rows:
        width = max((len(row) for row in rows), default=0)
        normalized = [row + [""] * (width - len(row)) for row in rows]
        header = normalized[0]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * width) + " |")
        for row in normalized[1:]:
            lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def merge_blocks(
    text_blocks: List[Dict],
    table_blocks: List[Dict],
    *,
    table_to_text: bool = False,
    table_format: str = "markdown",
    max_text_chars: int = 1500,
    deduplicate_table_text: bool = True,
) -> List[Dict]:
    tables_by_page: Dict[int, List[Dict]] = {}
    for table in table_blocks:
        tables_by_page.setdefault(int(table["page"]), []).append(table)

    merged: List[Dict] = []
    for block in text_blocks:
        page = int(block["page"])
        bbox = block["bbox"]
        if deduplicate_table_text:
            overlaps = any(_intersects(bbox, table["bbox"]) for table in tables_by_page.get(page, []))
            if overlaps:
                continue
        merged.append(block)

    merged = merge_into_paragraphs(merged, max_chars=max_text_chars)

    for table in table_blocks:
        block = copy.deepcopy(table)
        if table_to_text:
            block["content"] = _convert_table_content(block["content"], table_format=table_format)
            block["metadata"] = dict(block.get("metadata", {}))
            block["metadata"]["table_format"] = table_format
        block.setdefault("metadata", {})
        block["metadata"]["char_count"] = _char_count(block.get("content", ""))
        merged.append(block)

    ordered = sorted(
        merged,
        key=lambda block: (
            int(block["page"]),
            float(block["bbox"][1]) if block.get("bbox") else 0.0,
            float(block["bbox"][0]) if block.get("bbox") else 0.0,
            0 if block["type"] == "text" else 1,
        ),
    )

    page_counters: Dict[int, int] = {}
    for block in ordered:
        page = int(block["page"])
        page_counters[page] = page_counters.get(page, 0) + 1
        block["chunk_id"] = f"p{page}_{page_counters[page]:03d}"

    return ordered
