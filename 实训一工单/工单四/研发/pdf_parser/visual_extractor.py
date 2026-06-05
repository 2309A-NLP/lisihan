# -*- coding: utf-8 -*-
"""工单编号：人工智能 NLP-RAG-图像内容解析及检索优化。

本文件属于 PDF 招股说明书智能问答系统，用于保留工单一到工单四的文本检索、
结构化问答、负向问题处理、图片内容解析和检索优化能力。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from src.config import Config

try:
    import fitz
except Exception:  # pragma: no cover - 可选运行依赖
    fitz = None

from .visual_detection import (
    MAX_DUPLICATE_IMAGE_COUNT,
    MIN_IMAGE_AREA,
    _build_page_table_groups,
    _chart_rects_from_text,
    _guess_title,
    _image_rects_from_page,
    _nearby_image_rects,
    _repeated_image_xrefs,
    _stitch_cross_page_table_groups,
)
from .visual_geometry import (
    _clip_to_main_content,
    _dedupe_rects,
    _expand_rect,
    _is_in_main_content,
    _merge_related_table_rects,
    _overlap_ratio,
    _rect_area,
    _rect_to_list,
    _safe_stem,
    _union_rects,
)
from .visual_render import _image_signature, _render_clip, _render_stitched_clips
from .table_extractor import extract_table_blocks
from .text_extractor import extract_text_blocks


def extract_pdf_visuals(
    pdf_path: str | Path,
    *,
    text_blocks: List[Dict],
    table_blocks: List[Dict],
    output_dir: str | Path = None,
) -> List[Dict]:
    """Extract only visible chart/image regions and table regions from a PDF."""
    if fitz is None:
        return []

    pdf_file = Path(pdf_path)
    image_dir = Path(output_dir or Config.IMAGES_EXTRACT_DIR)
    image_dir.mkdir(parents=True, exist_ok=True)
    visuals: List[Dict] = []
    duplicate_counts: Dict[str, int] = {}

    with fitz.open(str(pdf_file)) as doc:
        repeated_xrefs = _repeated_image_xrefs(doc)
        table_groups_by_page = _build_page_table_groups(table_blocks, doc)
        table_chains = _stitch_cross_page_table_groups(table_groups_by_page, doc)
        table_rects_by_page: Dict[int, List[fitz.Rect]] = {}

        for table_index, chain in enumerate(table_chains, start=1):
            pages_and_rects = [(doc[item["page"] - 1], item["rect"]) for item in chain]
            pages = [item["page"] for item in chain]
            if len(chain) == 1:
                filename = f"{_safe_stem(pdf_file.stem)}_p{pages[0]:03d}_table{table_index:03d}.png"
            else:
                filename = f"{_safe_stem(pdf_file.stem)}_p{pages[0]:03d}_p{pages[-1]:03d}_table{table_index:03d}.png"
            image_path = image_dir / filename

            if not _render_stitched_clips(pages_and_rects, image_path):
                continue

            first_rect = chain[0]["rect"]
            title = _guess_title(text_blocks, pages[0], first_rect, f"table_{table_index}")
            related_blocks = [block for item in chain for block in item.get("blocks", [])]
            for item in chain:
                table_rects_by_page.setdefault(item["page"], []).append(item["rect"])

            visuals.append(
                {
                    "kind": "table",
                    "source_file": pdf_file.name,
                    "page": pages[0],
                    "pages": pages,
                    "title": title,
                    "index": table_index,
                    "path": str(image_path),
                    "bbox": [_rect_to_list(item["rect"]) for item in chain],
                    "page_width": float(doc[pages[0] - 1].rect.width),
                    "page_height": float(doc[pages[0] - 1].rect.height),
                    "rendered_region": True,
                    "stitched_pages": len(chain) > 1,
                    "table_metadata": {
                        "source_blocks": len(related_blocks),
                        "row_count": sum(int(block.get("metadata", {}).get("row_count", 0) or 0) for block in related_blocks),
                        "column_count": max(
                            [int(block.get("metadata", {}).get("column_count", 0) or 0) for block in related_blocks] or [0]
                        ),
                    },
                }
            )

        for page_index, page in enumerate(doc, start=1):
            page_rect = page.rect
            table_rects = table_rects_by_page.get(page_index, [])
            raw_image_rects = _image_rects_from_page(page, repeated_xrefs)
            chart_rects = []
            for chart_rect in _chart_rects_from_text(text_blocks, page_index, page_rect):
                related = _nearby_image_rects(page, chart_rect, repeated_xrefs)
                chart_rects.append(_union_rects([chart_rect] + related))

            image_rects = _merge_related_table_rects(
                [
                    _clip_to_main_content(_expand_rect(rect, page_rect, margin=12.0), page_rect)
                    for rect in raw_image_rects + chart_rects
                    if _is_in_main_content(rect, page_rect)
                ]
            )
            image_rects = _dedupe_rects(image_rects, overlap_threshold=0.70)
            for image_index, rect in enumerate(image_rects, start=1):
                if any(_overlap_ratio(rect, table_rect) > 0.75 for table_rect in table_rects):
                    continue
                if _rect_area(rect) < MIN_IMAGE_AREA:
                    continue
                filename = f"{_safe_stem(pdf_file.stem)}_p{page_index:03d}_image{image_index:02d}.png"
                image_path = image_dir / filename
                if not _render_clip(page, rect, image_path):
                    continue
                signature = _image_signature(image_path)
                duplicate_counts[signature] = duplicate_counts.get(signature, 0) + 1
                if duplicate_counts[signature] > MAX_DUPLICATE_IMAGE_COUNT:
                    image_path.unlink(missing_ok=True)
                    continue
                title = _guess_title(text_blocks, page_index, rect, f"image_{image_index}")
                visuals.append(
                    {
                        "kind": "image",
                        "source_file": pdf_file.name,
                        "page": page_index,
                        "title": title,
                        "index": image_index,
                        "xref": None,
                        "path": str(image_path),
                        "bbox": [_rect_to_list(rect)],
                        "page_width": float(page_rect.width),
                        "page_height": float(page_rect.height),
                        "rendered_region": True,
                    }
                )

    return visuals


def extract_pdf_images(
    pdf_path: str | Path,
    output_dir: str | Path = None,
    *,
    text_blocks: List[Dict] | None = None,
    table_blocks: List[Dict] | None = None,
) -> List[Dict]:
    text_blocks = text_blocks if text_blocks is not None else extract_text_blocks(pdf_path)
    table_blocks = table_blocks if table_blocks is not None else extract_table_blocks(pdf_path)
    return extract_pdf_visuals(pdf_path, text_blocks=text_blocks, table_blocks=table_blocks, output_dir=output_dir)
