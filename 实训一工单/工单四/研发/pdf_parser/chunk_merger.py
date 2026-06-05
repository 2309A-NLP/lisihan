# -*- coding: utf-8 -*-
"""工单编号：人工智能 NLP-RAG-图像内容解析及检索优化。

本文件属于 PDF 招股说明书智能问答系统，保留工单一到工单四的文本检索、
结构化问答、负向问题处理、图片内容解析和检索优化能力。
"""

from __future__ import annotations

import copy
import re
from typing import Dict, List, Sequence, Tuple

from .filters import normalize_text

DEFAULT_CHUNK_STRATEGY = "combined"
CHUNK_STRATEGY_ALIASES = {
    "default": "combined",
    "combined": "combined",
    "hybrid": "combined",
    "hybrid_chunking": "combined",
    "paragraph": "paragraph",
    "paragraphs": "paragraph",
    "段落切块": "paragraph",
    "recursive": "recursive",
    "recursive_character": "recursive",
    "递归切块": "recursive",
    "heading": "heading",
    "title": "heading",
    "标题切块": "heading",
    "parent_child": "combined",
    "parent-child": "combined",
    "parentchild": "combined",
    "父子块切块": "combined",
}

RECURSIVE_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", ";", "，", ",", " ", ""]
HEADING_PATTERNS: List[Tuple[re.Pattern, int]] = [
    (re.compile(r"^#{1,6}\s+(.+)$"), 1),
    (re.compile(r"^第[一二三四五六七八九十百千万0-9]+[章节篇部分]\s*.{0,80}$"), 1),
    (re.compile(r"^[一二三四五六七八九十]+[、.．]\s*.{1,80}$"), 2),
    (re.compile(r"^（[一二三四五六七八九十]+）\s*.{1,80}$"), 3),
    (re.compile(r"^\([一二三四五六七八九十]+\)\s*.{1,80}$"), 3),
    (re.compile(r"^[0-9]+(?:\.[0-9]+)*[、.．]\s*.{1,80}$"), 4),
]

SENTENCE_END_PATTERN = re.compile(r"[。！？；：\.\?\!;:]$")


def is_complete_sentence(text: str) -> bool:
    """Return True when the text ends with a sentence-level terminator."""
    return bool(SENTENCE_END_PATTERN.search(normalize_text(text)))


def normalize_chunk_strategy(strategy: str | None) -> str:
    normalized = (strategy or DEFAULT_CHUNK_STRATEGY).strip().lower()
    return CHUNK_STRATEGY_ALIASES.get(normalized, DEFAULT_CHUNK_STRATEGY)


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


def _tail_overlap(text: str, overlap: int) -> str:
    if overlap <= 0 or not text:
        return ""
    return text[-overlap:]


def _hard_split_text(text: str, max_chars: int, overlap: int = 0) -> List[str]:
    if max_chars <= 0:
        return [text] if text else []
    step = max(1, max_chars - max(0, overlap))
    return [text[index : index + max_chars] for index in range(0, len(text), step) if text[index : index + max_chars]]


def _split_keep_separator(text: str, separator: str) -> List[str]:
    if not separator:
        return list(text)
    raw_parts = text.split(separator)
    parts: List[str] = []
    for index, part in enumerate(raw_parts):
        if index < len(raw_parts) - 1:
            part = f"{part}{separator}"
        if part:
            parts.append(part)
    return parts


def _merge_parts_with_limit(parts: List[str], max_chars: int, overlap: int) -> List[str]:
    chunks: List[str] = []
    current = ""
    for part in parts:
        if not part:
            continue
        if not current:
            current = part
            continue
        if len(current) + len(part) <= max_chars:
            current += part
            continue

        chunks.append(current.strip())
        prefix = _tail_overlap(current, overlap)
        current = f"{prefix}{part}" if len(prefix) + len(part) <= max_chars else part

    if current:
        chunks.append(current.strip())
    return [chunk for chunk in chunks if chunk]


def _recursive_split_text(
    text: str,
    max_chars: int,
    overlap: int = 0,
    separators: Sequence[str] | None = None,
    separator_index: int = 0,
) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    separators = list(separators or RECURSIVE_SEPARATORS)
    separator = ""
    next_index = len(separators)
    for index in range(separator_index, len(separators)):
        candidate = separators[index]
        if candidate == "" or candidate in text:
            separator = candidate
            next_index = index + 1
            break

    if separator == "":
        return _hard_split_text(text, max_chars, overlap=overlap)

    pieces: List[str] = []
    for part in _split_keep_separator(text, separator):
        if len(part) > max_chars:
            pieces.extend(
                _recursive_split_text(
                    part,
                    max_chars,
                    overlap=overlap,
                    separators=separators,
                    separator_index=next_index,
                )
            )
        else:
            pieces.append(part)
    return _merge_parts_with_limit(pieces, max_chars, overlap)


def _copy_text_block(block: Dict, content: str, extra_metadata: Dict | None = None) -> Dict:
    copied = copy.deepcopy(block)
    copied["type"] = "text"
    copied["content"] = normalize_text(content)
    metadata = dict(copied.get("metadata", {}))
    metadata["char_count"] = len(copied["content"])
    if extra_metadata:
        metadata.update(extra_metadata)
    copied["metadata"] = metadata
    return copied


def _split_block_recursive(block: Dict, max_chars: int, overlap: int, strategy: str) -> List[Dict]:
    content = str(block.get("content", ""))
    segments = _recursive_split_text(content, max_chars=max_chars, overlap=overlap)
    if len(segments) <= 1:
        return [_copy_text_block(block, segments[0] if segments else content, {"chunk_strategy": strategy})]

    output = []
    for index, segment in enumerate(segments, start=1):
        output.append(
            _copy_text_block(
                block,
                segment,
                {
                    "chunk_strategy": strategy,
                    "split_part": index,
                    "split_total": len(segments),
                },
            )
        )
    return output


def _detect_heading(text: str) -> Tuple[int, str] | None:
    lines = [line.strip() for line in normalize_text(text).splitlines() if line.strip()]
    if not lines:
        return None
    first_line = lines[0]
    if len(first_line) > 100 or first_line.startswith("|"):
        return None
    for pattern, level in HEADING_PATTERNS:
        if pattern.match(first_line):
            return level, first_line.lstrip("#").strip()
    return None


def _append_block_to_section(section: Dict, block: Dict) -> None:
    section["content"] = f"{normalize_text(str(section.get('content', '')))}\n{normalize_text(str(block.get('content', '')))}"
    section["metadata"] = dict(section.get("metadata", {}))
    section["metadata"]["char_count"] = len(str(section.get("content", "")))
    section["metadata"]["page_end"] = max(
        int(section["metadata"].get("page_end", section.get("page", 0))),
        int(block.get("page", 0)),
    )
    if int(section.get("page", 0)) == int(block.get("page", 0)):
        section["bbox"] = _bbox_union(section.get("bbox") or [0, 0, 0, 0], block.get("bbox") or [0, 0, 0, 0])


def _heading_sections(blocks: List[Dict]) -> List[Dict]:
    sections: List[Dict] = []
    current: Dict | None = None

    ordered_blocks = sorted(
        blocks,
        key=lambda block: (
            int(block.get("page", 0)),
            float((block.get("bbox") or [0, 0, 0, 0])[1]),
            float((block.get("bbox") or [0, 0, 0, 0])[0]),
            0 if block.get("type") == "text" else 1,
        ),
    )
    for block in ordered_blocks:
        if block.get("type") != "text":
            continue
        block = copy.deepcopy(block)
        block.setdefault("metadata", {})
        block["metadata"].setdefault("char_count", _char_count(block.get("content", "")))

        heading = _detect_heading(str(block.get("content", "")))
        if heading is not None:
            if current is not None:
                sections.append(current)
            level, title = heading
            current = copy.deepcopy(block)
            metadata = dict(current.get("metadata", {}))
            metadata.update(
                {
                    "heading": title,
                    "heading_level": level,
                    "page_start": int(block.get("page", 0)),
                    "page_end": int(block.get("page", 0)),
                }
            )
            current["metadata"] = metadata
            continue

        if current is None:
            current = copy.deepcopy(block)
            metadata = dict(current.get("metadata", {}))
            metadata.setdefault("page_start", int(block.get("page", 0)))
            metadata.setdefault("page_end", int(block.get("page", 0)))
            current["metadata"] = metadata
        else:
            _append_block_to_section(current, block)

    if current is not None:
        sections.append(current)
    return sections


def chunk_by_recursive(blocks: List[Dict], max_chars: int, overlap: int = 0) -> List[Dict]:
    output: List[Dict] = []
    paragraph_limit = max(max_chars * 4, max_chars)
    for block in merge_into_paragraphs(blocks, max_chars=paragraph_limit):
        if block.get("type") != "text":
            output.append(block)
            continue
        output.extend(_split_block_recursive(block, max_chars=max_chars, overlap=overlap, strategy="recursive"))
    return output


def chunk_by_headings(blocks: List[Dict], max_chars: int, overlap: int = 0) -> List[Dict]:
    output: List[Dict] = []
    for section in _heading_sections(blocks):
        output.extend(_split_block_recursive(section, max_chars=max_chars, overlap=overlap, strategy="heading"))
    return output


def chunk_by_parent_child(
    blocks: List[Dict],
    max_chars: int,
    overlap: int = 0,
    parent_max_chars: int | None = None,
    strategy: str = "combined",
) -> List[Dict]:
    parent_max_chars = parent_max_chars or max(max_chars * 3, max_chars)
    parent_blocks: List[Dict] = []
    for section in _heading_sections(blocks):
        parent_blocks.extend(_split_block_recursive(section, max_chars=parent_max_chars, overlap=0, strategy="parent"))

    output: List[Dict] = []
    for parent_index, parent in enumerate(parent_blocks, start=1):
        parent_content = normalize_text(str(parent.get("content", "")))
        if not parent_content:
            continue
        parent_metadata = dict(parent.get("metadata", {}))
        parent_id = f"parent_{parent_index:04d}"
        heading = parent_metadata.get("heading", "")
        child_segments = _recursive_split_text(parent_content, max_chars=max_chars, overlap=overlap)
        for child_index, segment in enumerate(child_segments, start=1):
            child_content = segment
            if heading and heading not in child_content[: max(120, len(str(heading)) + 10)]:
                child_content = f"{heading}\n{child_content}"
            output.append(
                _copy_text_block(
                    parent,
                    child_content,
                    {
                        "chunk_strategy": strategy,
                        "parent_id": parent_id,
                        "parent_heading": heading,
                        "parent_content": parent_content,
                        "parent_char_count": len(parent_content),
                        "child_index": child_index,
                        "child_total": len(child_segments),
                    },
                )
            )
    return output


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
    chunk_overlap: int = 0,
    chunk_strategy: str = DEFAULT_CHUNK_STRATEGY,
    parent_text_chars: int | None = None,
    deduplicate_table_text: bool = True,
) -> List[Dict]:
    chunk_strategy = normalize_chunk_strategy(chunk_strategy)
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

    if chunk_strategy == "recursive":
        merged = chunk_by_recursive(merged, max_chars=max_text_chars, overlap=chunk_overlap)
    elif chunk_strategy == "heading":
        merged = chunk_by_headings(merged, max_chars=max_text_chars, overlap=chunk_overlap)
    elif chunk_strategy == "combined":
        merged = chunk_by_parent_child(
            merged,
            max_chars=max_text_chars,
            overlap=chunk_overlap,
            parent_max_chars=parent_text_chars,
            strategy=chunk_strategy,
        )
    else:
        merged = merge_into_paragraphs(merged, max_chars=max_text_chars)
        for block in merged:
            block.setdefault("metadata", {})
            block["metadata"]["chunk_strategy"] = chunk_strategy

    for table in table_blocks:
        block = copy.deepcopy(table)
        if table_to_text:
            block["content"] = _convert_table_content(block["content"], table_format=table_format)
            block["metadata"] = dict(block.get("metadata", {}))
            block["metadata"]["table_format"] = table_format
        block.setdefault("metadata", {})
        block["metadata"]["char_count"] = _char_count(block.get("content", ""))
        block["metadata"]["chunk_strategy"] = chunk_strategy
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
