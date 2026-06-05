# -*- coding: utf-8 -*-
"""MinerU adapter that preserves raw output and emits project-compatible chunks."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

try:
    import fitz
except Exception:  # pragma: no cover - optional outside parser tests
    fitz = None

from src.config import Config
from src.utils.wsl_mineru import run_mineru_wsl

from .metadata_extractor import extract_document_metadata


def _resolve_output_path(pdf_file: Path, output: str | None) -> Path:
    output_path = Path(output or Config.MINERU_OUTPUT_DIR)
    if output_path.suffix.lower() == ".json":
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path / f"{pdf_file.stem}_chunks.json"


def _candidate_auto_dirs(pdf_file: Path, output_dir: Path) -> Iterable[Path]:
    yield output_dir / pdf_file.stem / "auto"
    yield output_dir / pdf_file.stem / Config.MINERU_PARSE_METHOD
    yield output_dir / pdf_file.stem
    yield output_dir / "auto"
    yield output_dir


def _find_mineru_file(pdf_file: Path, output_dir: Path, suffix: str) -> Path | None:
    preferred = [directory / f"{pdf_file.stem}{suffix}" for directory in _candidate_auto_dirs(pdf_file, output_dir)]
    for path in preferred:
        if path.exists():
            return path
    matches = sorted(output_dir.rglob(f"*{suffix}"))
    exact = [path for path in matches if path.stem.replace("_content_list", "") == pdf_file.stem]
    return exact[0] if exact else None


def find_mineru_artifacts(pdf_path: str | Path, output_dir: str | Path) -> Dict[str, Path | None]:
    pdf_file = Path(pdf_path)
    output_root = Path(output_dir)
    return {
        "markdown": _find_mineru_file(pdf_file, output_root, ".md"),
        "content_list": _find_mineru_file(pdf_file, output_root, "_content_list.json"),
    }


def _load_content_list(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("content_list", "items", "data"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _page_count(pdf_file: Path, content_items: List[Dict[str, Any]]) -> int:
    pages = []
    for item in content_items:
        for key in ("page_idx", "page", "page_number"):
            value = item.get(key)
            if isinstance(value, int):
                pages.append(value + 1 if key == "page_idx" else value)
    if pages:
        return max(pages)
    if fitz is not None:
        try:
            with fitz.open(pdf_file) as doc:
                return len(doc)
        except Exception:
            return 0
    return 0


def _item_page(item: Dict[str, Any]) -> int:
    if isinstance(item.get("page_idx"), int):
        return int(item["page_idx"]) + 1
    if isinstance(item.get("page"), int):
        return int(item["page"])
    if isinstance(item.get("page_number"), int):
        return int(item["page_number"])
    return 0


def _item_bbox(item: Dict[str, Any]) -> List[float]:
    for key in ("bbox", "poly", "position"):
        value = item.get(key)
        if isinstance(value, list) and len(value) >= 4:
            if all(isinstance(part, (int, float)) for part in value[:4]):
                return [float(part) for part in value[:4]]
            if all(isinstance(part, list) and len(part) >= 2 for part in value):
                xs = [float(part[0]) for part in value]
                ys = [float(part[1]) for part in value]
                return [min(xs), min(ys), max(xs), max(ys)]
    return [0.0, 0.0, 0.0, 0.0]


def _item_type(item: Dict[str, Any]) -> str:
    raw = str(item.get("type") or item.get("category") or "").lower()
    if "table" in raw:
        return "table"
    if "image" in raw or "figure" in raw:
        return "image"
    return "text"


def _item_content(item: Dict[str, Any]) -> str:
    for key in ("text", "content", "md", "html", "caption"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    if isinstance(item.get("table_body"), str):
        return item["table_body"].strip()
    return ""


def _split_markdown(markdown_text: str, max_chars: int) -> List[str]:
    blocks = [block.strip() for block in re.split(r"\n{2,}", markdown_text) if block.strip()]
    chunks: List[str] = []
    for block in blocks:
        if len(block) <= max_chars:
            chunks.append(block)
            continue
        chunks.extend(block[index : index + max_chars] for index in range(0, len(block), max_chars))
    return chunks


def _chunks_from_content_list(content_items: List[Dict[str, Any]], max_text_chars: int) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    counters: Dict[int, int] = {}
    for item in content_items:
        content = _item_content(item)
        block_type = _item_type(item)
        if not content and block_type != "image":
            continue
        page = _item_page(item)
        counters[page] = counters.get(page, 0) + 1
        chunk = {
            "type": block_type,
            "page": page,
            "bbox": _item_bbox(item),
            "content": content,
            "chunk_id": f"p{page}_{counters[page]:03d}",
            "metadata": {
                "char_count": len(content),
                "parser_backend": "mineru",
                "mineru_type": item.get("type") or item.get("category") or block_type,
            },
        }
        chunks.append(chunk)
    return chunks


def _chunks_from_markdown(markdown_text: str, max_text_chars: int) -> List[Dict[str, Any]]:
    chunks = []
    for index, content in enumerate(_split_markdown(markdown_text, max_text_chars), start=1):
        chunks.append(
            {
                "type": "text",
                "page": 0,
                "bbox": [0.0, 0.0, 0.0, 0.0],
                "content": content,
                "chunk_id": f"md_{index:03d}",
                "metadata": {
                    "char_count": len(content),
                    "parser_backend": "mineru",
                    "mineru_type": "markdown",
                },
            }
        )
    return chunks


def parse_pdf_with_mineru(
    pdf_path: str | Path,
    *,
    output: str | None = None,
    max_text_chars: int = 1500,
) -> Dict[str, Any]:
    pdf_file = Path(pdf_path)
    output_path = _resolve_output_path(pdf_file, output)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = find_mineru_artifacts(pdf_file, output_dir)
    markdown_path = artifacts["markdown"]
    content_list_path = artifacts["content_list"]
    if markdown_path is None or content_list_path is None:
        run_mineru_wsl(pdf_file, output_dir)
        artifacts = find_mineru_artifacts(pdf_file, output_dir)
        markdown_path = artifacts["markdown"]
        content_list_path = artifacts["content_list"]

    missing = []
    if markdown_path is None or not markdown_path.exists():
        missing.append(".md")
    if content_list_path is None or not content_list_path.exists():
        missing.append("content_list.json")
    if missing:
        raise FileNotFoundError(
            "MinerU output is incomplete; missing "
            + ", ".join(missing)
            + f" for {pdf_file}. Expected files under {output_dir / pdf_file.stem / 'auto'}."
        )

    markdown_text = markdown_path.read_text(encoding="utf-8")
    content_items = _load_content_list(content_list_path)
    chunks = _chunks_from_content_list(content_items, max_text_chars)
    if not chunks:
        chunks = _chunks_from_markdown(markdown_text, max_text_chars)

    document_metadata = extract_document_metadata(pdf_file)
    visual_blocks = [
        {
            "kind": "image",
            "source_file": pdf_file.name,
            "page": chunk.get("page", 0),
            "title": "",
            "index": index,
            "xref": None,
            "path": chunk.get("metadata", {}).get("img_path", ""),
            "bbox": [chunk.get("bbox", [])],
            "rendered_page": False,
        }
        for index, chunk in enumerate(chunks)
        if chunk.get("type") == "image"
    ]
    result = {
        "source_file": pdf_file.name,
        "source_path": str(pdf_file),
        "page_count": _page_count(pdf_file, content_items),
        "chunk_count": len(chunks),
        "table_count": sum(1 for chunk in chunks if chunk.get("type") == "table"),
        "image_count": len(visual_blocks),
        "table_to_text": True,
        "table_format": "markdown",
        "parser_backend": "mineru",
        "mineru_markdown": str(markdown_path),
        "mineru_content_list": str(content_list_path),
        "document_metadata": document_metadata,
        "visual_index": visual_blocks,
        "images": visual_blocks,
        "chunks": chunks,
    }

    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metadata_path = output_path.parent / f"{pdf_file.stem}_metadata.json"
    metadata_path.write_text(json.dumps(document_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"result": result, "output_path": str(output_path)}
