# -*- coding: utf-8 -*-
"""Configurable text chunking strategies for parsed PDF blocks."""

from __future__ import annotations

import copy
import re
from typing import Dict, Iterable, List, Tuple

from .filters import normalize_text

CHUNK_STRATEGIES = ("paragraph", "recursive", "title", "parent_child")

_SENTENCE_BOUNDARY = r"(?<=[。！？；：\.\?\!;:])"
_HEADING_PATTERNS: Tuple[Tuple[re.Pattern, int], ...] = (
    (re.compile(r"^\s*(#{1,6})\s+(.+?)\s*$"), 0),
    (re.compile(r"^\s*第[一二三四五六七八九十百千万\d]+章\s+(.+?)\s*$"), 1),
    (re.compile(r"^\s*第[一二三四五六七八九十百千万\d]+节\s+(.+?)\s*$"), 2),
    (re.compile(r"^\s*[一二三四五六七八九十]+[、．.]\s*(.+?)\s*$"), 2),
    (re.compile(r"^\s*[（(][一二三四五六七八九十\d]+[）)]\s*(.+?)\s*$"), 3),
    (re.compile(r"^\s*\d+(?:\.\d+){0,3}[、．.]?\s+(.+?)\s*$"), 3),
)


def normalize_chunk_strategy(strategy: str | None) -> str:
    value = (strategy or "paragraph").strip().lower().replace("-", "_")
    aliases = {
        "default": "paragraph",
        "sentence": "paragraph",
        "heading": "title",
        "headers": "title",
        "header": "title",
        "parent-child": "parent_child",
        "parent_child_chunk": "parent_child",
    }
    value = aliases.get(value, value)
    if value not in CHUNK_STRATEGIES:
        return "paragraph"
    return value


def _ordered(blocks: Iterable[Dict]) -> List[Dict]:
    return sorted(
        (copy.deepcopy(block) for block in blocks),
        key=lambda block: (
            int(block.get("page", 0)),
            float((block.get("bbox") or [0, 0, 0, 0])[1]),
            float((block.get("bbox") or [0, 0, 0, 0])[0]),
        ),
    )


def _first_line(text: str) -> str:
    for line in (text or "").splitlines():
        line = normalize_text(line)
        if line:
            return line
    return ""


def detect_heading(text: str) -> Tuple[int, str] | None:
    first = _first_line(text)
    if not first or len(first) > 120:
        return None

    markdown = _HEADING_PATTERNS[0][0].match(first)
    if markdown:
        return len(markdown.group(1)), markdown.group(2).strip()

    for pattern, level in _HEADING_PATTERNS[1:]:
        if pattern.match(first):
            return level, first.strip()

    if len(first) <= 36 and not re.search(r"[。！？；;]$", first) and re.search(r"[\u4e00-\u9fff]", first):
        if any(marker in first for marker in ("概况", "情况", "业务", "风险", "发行", "财务", "募集", "关联", "管理")):
            return 3, first
    return None


def split_text_recursive(text: str, max_chars: int, overlap: int = 0) -> List[str]:
    text = normalize_text(text)
    max_chars = max(80, int(max_chars or 1500))
    overlap = max(0, min(int(overlap or 0), max_chars // 3))
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    parts = _recursive_parts(text, max_chars, ["\n\n", "\n", _SENTENCE_BOUNDARY, " ", ""])
    chunks = _pack_parts(parts, max_chars)
    if overlap:
        chunks = _with_overlap(chunks, max_chars, overlap)
    return [chunk for chunk in chunks if chunk]


def _recursive_parts(text: str, max_chars: int, separators: List[str]) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    if not separators:
        return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]

    separator = separators[0]
    if separator == "":
        return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]
    if separator == _SENTENCE_BOUNDARY:
        raw_parts = re.split(separator, text)
    else:
        raw_parts = text.split(separator)

    if len(raw_parts) <= 1:
        return _recursive_parts(text, max_chars, separators[1:])

    parts: List[str] = []
    joiner = "" if separator == _SENTENCE_BOUNDARY else separator
    for raw in raw_parts:
        piece = normalize_text(raw)
        if not piece:
            continue
        if len(piece) > max_chars:
            parts.extend(_recursive_parts(piece, max_chars, separators[1:]))
        else:
            parts.append(piece + (joiner if separator in {"\n\n", "\n"} else ""))
    return parts


def _pack_parts(parts: List[str], max_chars: int) -> List[str]:
    chunks: List[str] = []
    current = ""
    for part in parts:
        part = normalize_text(part)
        if not part:
            continue
        candidate = f"{current}\n{part}" if current else part
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(normalize_text(current))
        current = part
        while len(current) > max_chars:
            chunks.append(current[:max_chars])
            current = current[max_chars:]
    if current:
        chunks.append(normalize_text(current))
    return chunks


def _with_overlap(chunks: List[str], max_chars: int, overlap: int) -> List[str]:
    if not chunks or overlap <= 0:
        return chunks
    output = [chunks[0]]
    for previous, current in zip(chunks, chunks[1:]):
        prefix = previous[-overlap:]
        merged = normalize_text(f"{prefix}\n{current}")
        output.append(merged[-max_chars:] if len(merged) > max_chars else merged)
    return output


def _new_block(template: Dict, content: str, *, index: int, strategy: str, extra_metadata: Dict | None = None) -> Dict:
    block = copy.deepcopy(template)
    block["content"] = normalize_text(content)
    block["type"] = "text"
    block["metadata"] = dict(block.get("metadata", {}))
    block["metadata"].update(extra_metadata or {})
    block["metadata"]["chunk_strategy"] = strategy
    block["metadata"]["char_count"] = len(block["content"])
    if index > 1:
        block["metadata"]["split_part"] = index
    return block


def recursive_chunk_blocks(blocks: List[Dict], max_chars: int, overlap: int = 0) -> List[Dict]:
    output: List[Dict] = []
    for block in _ordered(blocks):
        content = block.get("content", "")
        if block.get("type") != "text" or not isinstance(content, str):
            output.append(block)
            continue
        segments = split_text_recursive(content, max_chars=max_chars, overlap=overlap)
        for index, segment in enumerate(segments, start=1):
            output.append(_new_block(block, segment, index=index, strategy="recursive"))
    return output


def _sectionize(blocks: List[Dict]) -> List[Dict]:
    sections: List[Dict] = []
    stack: List[Tuple[int, str]] = []
    current: Dict | None = None

    for block in _ordered(blocks):
        content = normalize_text(str(block.get("content", "")))
        heading = detect_heading(content)
        if heading:
            level, title = heading
            stack = [(item_level, item_title) for item_level, item_title in stack if item_level < level]
            stack.append((level, title))
            current = {
                "title": title,
                "level": level,
                "heading_path": [item_title for _, item_title in stack],
                "blocks": [],
                "start_page": int(block.get("page", 0)),
                "template": block,
            }
            sections.append(current)
        elif current is None:
            current = {
                "title": "正文",
                "level": 0,
                "heading_path": [],
                "blocks": [],
                "start_page": int(block.get("page", 0)),
                "template": block,
            }
            sections.append(current)

        current["blocks"].append(block)

    return sections


def _section_text(section: Dict) -> str:
    body = "\n".join(normalize_text(str(block.get("content", ""))) for block in section["blocks"])
    heading_path = " > ".join(section.get("heading_path") or [])
    if heading_path and not body.startswith(heading_path):
        return normalize_text(f"{heading_path}\n{body}")
    return normalize_text(body)


def title_chunk_blocks(blocks: List[Dict], max_chars: int, overlap: int = 0) -> List[Dict]:
    output: List[Dict] = []
    for section_index, section in enumerate(_sectionize(blocks), start=1):
        text = _section_text(section)
        if not text:
            continue
        segments = split_text_recursive(text, max_chars=max_chars, overlap=overlap)
        extra = {
            "section_title": section.get("title", ""),
            "heading_path": " > ".join(section.get("heading_path") or []),
            "section_level": section.get("level", 0),
            "section_index": section_index,
        }
        for split_index, segment in enumerate(segments, start=1):
            output.append(
                _new_block(
                    section["template"],
                    segment,
                    index=split_index,
                    strategy="title",
                    extra_metadata=extra,
                )
            )
    return output


def parent_child_chunk_blocks(blocks: List[Dict], max_chars: int, overlap: int = 0) -> List[Dict]:
    output: List[Dict] = []
    parent_max_chars = max(max_chars * 3, max_chars + 1)
    for section_index, section in enumerate(_sectionize(blocks), start=1):
        parent_id = f"section_{section_index:04d}"
        parent_text = _section_text(section)
        if not parent_text:
            continue
        heading_path = " > ".join(section.get("heading_path") or [])
        parent_preview = parent_text[: min(500, len(parent_text))]
        segments = split_text_recursive(parent_text, max_chars=max_chars, overlap=overlap)
        if len(parent_text) <= parent_max_chars:
            parent_extra = {
                "chunk_level": "parent",
                "parent_id": parent_id,
                "section_title": section.get("title", ""),
                "heading_path": heading_path,
                "section_level": section.get("level", 0),
                "section_index": section_index,
            }
            output.append(
                _new_block(
                    section["template"],
                    parent_text,
                    index=1,
                    strategy="parent_child",
                    extra_metadata=parent_extra,
                )
            )
        for split_index, segment in enumerate(segments, start=1):
            child_extra = {
                "chunk_level": "child",
                "parent_id": parent_id,
                "parent_title": section.get("title", ""),
                "heading_path": heading_path,
                "section_level": section.get("level", 0),
                "section_index": section_index,
                "parent_preview": parent_preview,
            }
            child_content = normalize_text(f"{heading_path}\n{segment}" if heading_path and heading_path not in segment[:120] else segment)
            output.append(
                _new_block(
                    section["template"],
                    child_content,
                    index=split_index,
                    strategy="parent_child",
                    extra_metadata=child_extra,
                )
            )
    return output


def apply_text_chunk_strategy(
    blocks: List[Dict],
    *,
    strategy: str,
    max_chars: int,
    overlap: int = 0,
) -> List[Dict]:
    normalized_strategy = normalize_chunk_strategy(strategy)
    if normalized_strategy == "recursive":
        return recursive_chunk_blocks(blocks, max_chars=max_chars, overlap=overlap)
    if normalized_strategy == "title":
        return title_chunk_blocks(blocks, max_chars=max_chars, overlap=overlap)
    if normalized_strategy == "parent_child":
        return parent_child_chunk_blocks(blocks, max_chars=max_chars, overlap=overlap)
    output = []
    for block in blocks:
        current = copy.deepcopy(block)
        current.setdefault("metadata", {})
        current["metadata"]["chunk_strategy"] = "paragraph"
        output.append(current)
    return output
