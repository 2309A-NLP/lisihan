# -*- coding: utf-8 -*-
"""MinerU-backed PDF parsing adapter for the project parser contract."""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Dict, List

from src.config import Config

from .chunk_merger import DEFAULT_CHUNK_STRATEGY, merge_blocks, normalize_chunk_strategy
from .metadata_extractor import extract_document_metadata
from .visual_geometry import _resolve_output_path


TEXT_TYPES = {"text", "title"}
SKIPPED_TEXT_TYPES = {"header", "footer", "page_number"}


def parse_pdf_file_with_mineru(
    pdf_path: str | Path,
    *,
    output: str | None = None,
    table_to_text: bool = False,
    table_format: str = "markdown",
    max_text_chars: int = 1500,
    chunk_overlap: int = 0,
    chunk_strategy: str = DEFAULT_CHUNK_STRATEGY,
    parent_text_chars: int | None = None,
    deduplicate_table_text: bool = True,
) -> Dict:
    pdf_file = Path(pdf_path)
    chunk_strategy = normalize_chunk_strategy(chunk_strategy)
    mineru_dir = _ensure_mineru_output(pdf_file)
    markdown_path = _find_mineru_artifact(mineru_dir, pdf_file.stem, "*.md")
    if Config.MINERU_REQUIRE_MARKDOWN and markdown_path is None:
        raise FileNotFoundError(f"MinerU markdown not found for {pdf_file.name} in {mineru_dir}")
    content_list_path = _find_mineru_artifact(mineru_dir, pdf_file.stem, "*content_list*.json")
    if content_list_path is None:
        raise FileNotFoundError(f"MinerU content list not found for {pdf_file.name} in {mineru_dir}")

    content_items = json.loads(content_list_path.read_text(encoding="utf-8"))
    text_blocks, table_blocks, visual_blocks = _content_list_to_blocks(content_items, pdf_file, content_list_path.parent)
    document_metadata = extract_document_metadata(pdf_file)
    merged_blocks = merge_blocks(
        text_blocks,
        table_blocks,
        table_to_text=table_to_text,
        table_format=table_format,
        max_text_chars=max_text_chars,
        chunk_overlap=chunk_overlap,
        chunk_strategy=chunk_strategy,
        parent_text_chars=parent_text_chars,
        deduplicate_table_text=deduplicate_table_text,
    )

    result = {
        "source_file": pdf_file.name,
        "source_path": str(pdf_file),
        "page_count": max((block["page"] for block in text_blocks + table_blocks + visual_blocks), default=0),
        "chunk_count": len(merged_blocks),
        "table_count": len(table_blocks),
        "image_count": len(visual_blocks),
        "table_to_text": table_to_text,
        "table_format": table_format if table_to_text else "raw",
        "chunk_strategy": chunk_strategy,
        "chunk_size": max_text_chars,
        "chunk_overlap": chunk_overlap,
        "parent_chunk_size": parent_text_chars,
        "parser_backend": "mineru",
        "mineru_markdown": str(markdown_path) if markdown_path else "",
        "mineru_content_list": str(content_list_path),
        "document_metadata": document_metadata,
        "visual_index": visual_blocks,
        "images": visual_blocks,
        "chunks": merged_blocks,
    }

    output_path = _resolve_output_path(pdf_file, output)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metadata_path = output_path.parent / f"{pdf_file.stem}_metadata.json"
    metadata_path.write_text(json.dumps(document_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"result": result, "output_path": str(output_path)}


def has_mineru_content_list(pdf_path: str | Path) -> bool:
    pdf_file = Path(pdf_path)
    if not str(pdf_path) or not pdf_file.stem:
        return False
    output_root = Path(Config.MINERU_OUTPUT_DIR)
    return (
        _find_mineru_artifact(output_root / pdf_file.stem, pdf_file.stem, "*content_list*.json") is not None
        or _find_mineru_artifact(output_root, pdf_file.stem, "*content_list*.json") is not None
    )


def has_mineru_source_files(pdf_path: str | Path) -> bool:
    pdf_file = Path(pdf_path)
    if not str(pdf_path) or not pdf_file.stem:
        return False
    output_root = Path(Config.MINERU_OUTPUT_DIR)
    has_markdown = (
        _find_mineru_artifact(output_root / pdf_file.stem, pdf_file.stem, "*.md") is not None
        or _find_mineru_artifact(output_root, pdf_file.stem, "*.md") is not None
    )
    return has_markdown and has_mineru_content_list(pdf_file)


def _ensure_mineru_output(pdf_file: Path) -> Path:
    output_root = Path(Config.MINERU_OUTPUT_DIR)
    cached_dir = output_root / pdf_file.stem
    if has_mineru_source_files(pdf_file):
        if _find_mineru_artifact(cached_dir, pdf_file.stem, "*content_list*.json") is not None:
            return cached_dir
        return output_root
    if not Config.MINERU_REQUIRE_MARKDOWN and _find_mineru_artifact(cached_dir, pdf_file.stem, "*content_list*.json") is not None:
        return cached_dir
    if not Config.MINERU_REQUIRE_MARKDOWN and _find_mineru_artifact(output_root, pdf_file.stem, "*content_list*.json") is not None:
        return output_root
    if not Config.MINERU_AUTO_RUN:
        return cached_dir

    from src.utils.wsl_mineru import parse_pdf_with_mineru_wsl

    parse_pdf_with_mineru_wsl(
        pdf_file,
        output_dir=output_root,
        distro=Config.MINERU_WSL_DISTRO,
        conda_prefix=Config.MINERU_CONDA_PREFIX,
        conda_env=Config.MINERU_CONDA_ENV,
        method=Config.MINERU_METHOD,
        backend=Config.MINERU_BACKEND,
        lang=Config.MINERU_LANG,
        extra_env={"MINERU_MODEL_SOURCE": Config.MINERU_MODEL_SOURCE},
        timeout=Config.MINERU_TIMEOUT_SECONDS or None,
    )
    return output_root


def _find_mineru_artifact(root: Path, stem: str, pattern: str) -> Path | None:
    if not root.exists():
        return None
    candidates = [path for path in root.rglob(pattern) if stem in path.stem]
    if not candidates:
        return None
    if "content_list" in pattern:
        standard = [path for path in candidates if path.name == f"{stem}_content_list.json"]
        if standard:
            return max(standard, key=lambda path: path.stat().st_mtime)
        non_v2 = [path for path in candidates if "_content_list_v2" not in path.name]
        if non_v2:
            return max(non_v2, key=lambda path: path.stat().st_mtime)
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _content_list_to_blocks(items: List[Dict], pdf_file: Path, artifact_dir: Path) -> tuple[List[Dict], List[Dict], List[Dict]]:
    text_blocks: List[Dict] = []
    table_blocks: List[Dict] = []
    visual_blocks: List[Dict] = []

    for index, item in enumerate(_iter_content_items(items), start=1):
        item_type = str(item.get("type", "")).lower()
        page = int(item.get("page_idx", 0) or 0) + 1
        bbox = _bbox_to_list(item.get("bbox"))

        if item_type in TEXT_TYPES:
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            if item.get("text_level"):
                text = f"{'#' * int(item.get('text_level', 1))} {text}"
            text_blocks.append(
                {
                    "type": "text",
                    "content": text,
                    "page": page,
                    "bbox": bbox,
                    "metadata": {
                        "source": "mineru",
                        "text_level": item.get("text_level"),
                        "mineru_index": index,
                    },
                }
            )
            continue

        if item_type == "table":
            caption = _join_notes(item.get("table_caption", []))
            body = _html_table_to_text(str(item.get("table_body", "")))
            footnote = _join_notes(item.get("table_footnote", []))
            content = "\n".join(part for part in [caption, body, footnote] if part)
            table_blocks.append(
                {
                    "type": "table",
                    "content": content,
                    "page": page,
                    "bbox": bbox,
                    "metadata": {
                        "source": "mineru",
                        "mineru_index": index,
                        "table_format": "text",
                        "row_count": max(1, len(re.findall(r"</tr>", str(item.get("table_body", "")), flags=re.I))),
                        "column_count": max(1, len(re.findall(r"<td\b", str(item.get("table_body", "")), flags=re.I))),
                        "image_path": _resolve_mineru_media_path(item.get("img_path"), artifact_dir),
                    },
                }
            )

        if item_type in {"image", "table"} and item.get("img_path"):
            media_path = _resolve_mineru_media_path(item.get("img_path"), artifact_dir)
            visual_blocks.append(
                {
                    "kind": "table" if item_type == "table" else "image",
                    "source_file": pdf_file.name,
                    "page": page,
                    "title": _visual_title(item, index),
                    "index": index,
                    "xref": None,
                    "path": media_path,
                    "bbox": [bbox],
                    "rendered_region": True,
                    "source": "mineru",
                }
            )
            continue

        if item_type in SKIPPED_TEXT_TYPES:
            continue

    return text_blocks, table_blocks, visual_blocks


def _iter_content_items(items):
    for item in items or []:
        if isinstance(item, dict):
            yield item
        elif isinstance(item, list):
            yield from _iter_content_items(item)


def _bbox_to_list(value) -> List[float]:
    raw = list(value or [0, 0, 0, 0])[:4]
    raw += [0] * (4 - len(raw))
    return [float(item or 0) for item in raw]


def _join_notes(values) -> str:
    if not values:
        return ""
    if isinstance(values, list):
        return "\n".join(str(item).strip() for item in values if str(item).strip())
    return str(values).strip()


def _html_table_to_text(value: str) -> str:
    text = html.unescape(value or "")
    text = re.sub(r"</tr\s*>", "\n", text, flags=re.I)
    text = re.sub(r"</t[dh]\s*>\s*<t[dh][^>]*>", " | ", text, flags=re.I)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s+", "\n", text)
    return text.strip()


def _resolve_mineru_media_path(value, artifact_dir: Path) -> str:
    if not value:
        return ""
    media_path = Path(str(value))
    if not media_path.is_absolute():
        media_path = artifact_dir / media_path
    return str(media_path)


def _visual_title(item: Dict, index: int) -> str:
    for key in ("image_caption", "table_caption"):
        caption = _join_notes(item.get(key, []))
        if caption:
            return caption[:120]
    return f"{item.get('type', 'visual')}_{index}"
