# -*- coding: utf-8 -*-
"""MinerU parsing adapter.

This module keeps MinerU as the parsing backend while preserving the
project's original ``*_chunks.json`` shape for downstream RAG code.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

from pdf_parser.filters import normalize_text
from pdf_parser.metadata_extractor import extract_document_metadata
from src.config import Config


class MinerUParseError(RuntimeError):
    """Raised when MinerU output is missing or unusable."""


def mineru_document_dir(pdf_path: str | Path, output_dir: str | Path | None = None) -> Path:
    return Path(output_dir or Config.MINERU_OUTPUT_DIR) / Path(pdf_path).stem / "auto"


def mineru_markdown_path(pdf_path: str | Path, output_dir: str | Path | None = None) -> Path:
    pdf_file = Path(pdf_path)
    return mineru_document_dir(pdf_file, output_dir) / f"{pdf_file.stem}.md"


def mineru_content_list_path(pdf_path: str | Path, output_dir: str | Path | None = None) -> Path:
    pdf_file = Path(pdf_path)
    return mineru_document_dir(pdf_file, output_dir) / f"{pdf_file.stem}_content_list.json"


def chunks_output_path(pdf_path: str | Path, output_dir: str | Path | None = None) -> Path:
    return Path(output_dir or Config.MINERU_OUTPUT_DIR) / f"{Path(pdf_path).stem}_chunks.json"


def _read_json(path: Path) -> Any:
    text = path.read_text(encoding="utf-8-sig")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(path.read_text(encoding="utf-8"))


def _first_int(*values: Any, default: int = 1) -> int:
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return max(1, value)
        if isinstance(value, str) and value.strip().isdigit():
            return max(1, int(value.strip()))
    return default


def _bbox_from_item(item: Dict[str, Any]) -> List[float]:
    for key in ("bbox", "poly", "position"):
        value = item.get(key)
        if isinstance(value, list) and len(value) >= 4:
            if all(isinstance(point, list) for point in value):
                xs = [float(point[0]) for point in value if len(point) >= 2]
                ys = [float(point[1]) for point in value if len(point) >= 2]
                if xs and ys:
                    return [min(xs), min(ys), max(xs), max(ys)]
            try:
                numbers = [float(v) for v in value[:4]]
                return numbers
            except (TypeError, ValueError):
                pass
    return [0.0, 0.0, 0.0, 0.0]


def _content_text(item: Dict[str, Any]) -> str:
    for key in ("text", "content", "md", "html", "table_body", "table_caption", "img_caption"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return normalize_text(value)
        if isinstance(value, list):
            text = "\n".join(str(part) for part in value if str(part).strip())
            if text.strip():
                return normalize_text(text)
    return ""


def _item_type(item: Dict[str, Any]) -> str:
    raw = str(item.get("type") or item.get("category") or item.get("block_type") or "").lower()
    if "table" in raw:
        return "table"
    if "image" in raw or "img" in raw or "figure" in raw:
        return "image"
    return "text"


def _iter_content_items(content_list: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(content_list, list):
        for item in content_list:
            if isinstance(item, dict):
                yield item
        return
    if isinstance(content_list, dict):
        for key in ("content", "content_list", "pages", "items"):
            value = content_list.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        if key == "pages":
                            for block in item.get("blocks", []) or item.get("items", []) or []:
                                if isinstance(block, dict):
                                    block = dict(block)
                                    block.setdefault("page_idx", item.get("page_idx", item.get("page_no")))
                                    yield block
                        else:
                            yield item


def _chunks_from_content_list(content_list: Any, pdf_file: Path, markdown_path: Path, content_list_path: Path) -> List[Dict]:
    chunks: List[Dict] = []
    page_counters: Dict[int, int] = {}

    for item in _iter_content_items(content_list):
        block_type = _item_type(item)
        content = _content_text(item)
        if not content:
            continue

        if item.get("page") is not None or item.get("page_no") is not None or item.get("page_num") is not None:
            page = _first_int(item.get("page"), item.get("page_no"), item.get("page_num"), default=1)
        else:
            page_idx = item.get("page_idx")
            page = int(page_idx) + 1 if isinstance(page_idx, int) and page_idx >= 0 else 1
        page_counters[page] = page_counters.get(page, 0) + 1

        metadata = {
            "source_file": pdf_file.name,
            "source_path": str(pdf_file),
            "parser_backend": "mineru",
            "mineru_markdown": str(markdown_path),
            "mineru_content_list": str(content_list_path),
            "char_count": len(content),
        }
        if block_type == "table":
            metadata["has_table"] = True
            metadata["table_format"] = "markdown"
        else:
            metadata["has_table"] = False

        chunks.append(
            {
                "page": page,
                "type": block_type,
                "content": content,
                "bbox": _bbox_from_item(item),
                "metadata": metadata,
                "chunk_id": f"p{page}_{page_counters[page]:03d}",
            }
        )
    return chunks


def _chunks_from_markdown(markdown: str, pdf_file: Path, markdown_path: Path, content_list_path: Path, max_text_chars: int) -> List[Dict]:
    parts = [part.strip() for part in re.split(r"\n{2,}", markdown) if part.strip()]
    chunks: List[Dict] = []
    counter = 0
    for part in parts:
        while part:
            segment = part[:max_text_chars].strip()
            part = part[max_text_chars:].strip()
            if not segment:
                continue
            counter += 1
            chunks.append(
                {
                    "page": 1,
                    "type": "text",
                    "content": normalize_text(segment),
                    "bbox": [0.0, 0.0, 0.0, 0.0],
                    "metadata": {
                        "source_file": pdf_file.name,
                        "source_path": str(pdf_file),
                        "parser_backend": "mineru",
                        "mineru_markdown": str(markdown_path),
                        "mineru_content_list": str(content_list_path),
                        "char_count": len(segment),
                        "has_table": False,
                    },
                    "chunk_id": f"p1_{counter:03d}",
                }
            )
    return chunks


def adapt_mineru_output(
    pdf_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    max_text_chars: int = 1500,
) -> Dict[str, Any]:
    pdf_file = Path(pdf_path)
    root = Path(output_dir or Config.MINERU_OUTPUT_DIR)
    markdown_path = mineru_markdown_path(pdf_file, root)
    content_list_path = mineru_content_list_path(pdf_file, root)

    missing = [str(path) for path in (markdown_path, content_list_path) if not path.exists()]
    if missing:
        raise MinerUParseError(
            "MinerU output is incomplete. Missing required file(s): "
            + ", ".join(missing)
            + ". Run MinerU first; local parsing fallback is disabled."
        )

    markdown = markdown_path.read_text(encoding="utf-8")
    content_list = _read_json(content_list_path)
    chunks = _chunks_from_content_list(content_list, pdf_file, markdown_path, content_list_path)
    if not chunks:
        chunks = _chunks_from_markdown(markdown, pdf_file, markdown_path, content_list_path, max_text_chars)

    table_count = sum(1 for chunk in chunks if chunk.get("type") == "table")
    image_count = sum(1 for chunk in chunks if chunk.get("type") == "image")
    page_count = max((int(chunk.get("page", 0) or 0) for chunk in chunks), default=0)
    document_metadata = extract_document_metadata(pdf_file)

    result = {
        "source_file": pdf_file.name,
        "source_path": str(pdf_file),
        "parser_backend": "mineru",
        "mineru_markdown": str(markdown_path),
        "mineru_content_list": str(content_list_path),
        "page_count": page_count,
        "chunk_count": len(chunks),
        "table_count": table_count,
        "image_count": image_count,
        "table_to_text": True,
        "table_format": "markdown",
        "document_metadata": document_metadata,
        "visual_index": [chunk for chunk in chunks if chunk.get("type") == "image"],
        "images": [chunk for chunk in chunks if chunk.get("type") == "image"],
        "chunks": chunks,
    }

    output_path = chunks_output_path(pdf_file, root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"result": result, "output_path": str(output_path)}
