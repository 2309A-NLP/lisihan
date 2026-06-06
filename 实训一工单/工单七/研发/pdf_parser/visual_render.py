# -*- coding: utf-8 -*-
"""工单编号：人工智能 NLP-RAG-图像内容解析及检索优化。

本文件属于 PDF 招股说明书智能问答系统，用于保留工单一到工单四的文本检索、
结构化问答、负向问题处理、图片内容解析和检索优化能力。
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Dict, List

from src.config import Config

try:
    import fitz
except Exception:  # pragma: no cover - 可选运行依赖
    fitz = None

try:
    from PIL import Image, ImageStat
except Exception:  # pragma: no cover - 可选运行依赖
    Image = None
    ImageStat = None

from .visual_geometry import _expand_rect


def _safe_stem(path: str) -> str:
    from pathlib import Path
    return Path(path).stem


def _is_low_information_image(image_path: Path) -> bool:
    if Image is None or ImageStat is None:
        return False

    try:
        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
            width, height = rgb_image.size
            if width < 32 or height < 32:
                return True

            aspect_ratio = max(width / max(height, 1), height / max(width, 1))
            if aspect_ratio > 8:
                return True

            extrema = rgb_image.getextrema()
            channel_ranges = [high - low for low, high in extrema]
            means = ImageStat.Stat(rgb_image).mean
            if width * height < 12000 and max(channel_ranges) < 80:
                return True

            is_nearly_solid = max(channel_ranges) <= 3
            is_black_or_white = max(means) <= 5 or min(means) >= 250
            return is_nearly_solid and is_black_or_white
    except Exception:
        return False


def _image_signature(image_path: Path) -> str:
    if Image is None:
        return hashlib.sha1(image_path.read_bytes()).hexdigest()
    try:
        with Image.open(image_path) as image:
            small = image.convert("L").resize((16, 16))
            return hashlib.sha1(small.tobytes()).hexdigest()
    except Exception:
        return hashlib.sha1(image_path.read_bytes()).hexdigest()


def _render_clip(
    page: "fitz.Page",
    rect: "fitz.Rect",
    image_path: Path,
    *,
    zoom: float = 2.0,
) -> bool:
    if rect.is_empty or rect.width <= 2 or rect.height <= 2:
        return False
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=rect, alpha=False)
    pix.save(str(image_path))
    if _is_low_information_image(image_path):
        image_path.unlink(missing_ok=True)
        return False
    return True


def _render_stitched_clips(
    pages_and_rects: List[tuple["fitz.Page", "fitz.Rect"]],
    image_path: Path,
    *,
    zoom: float = 2.0,
) -> bool:
    if Image is None or not pages_and_rects:
        return False

    rendered: List[Image.Image] = []
    try:
        for page, rect in pages_and_rects:
            if rect.is_empty or rect.width <= 2 or rect.height <= 2:
                continue
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=rect, alpha=False)
            image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            rendered.append(image)

        if not rendered:
            return False

        width = max(image.width for image in rendered)
        height = sum(image.height for image in rendered)
        stitched = Image.new("RGB", (width, height), "white")
        y = 0
        for image in rendered:
            stitched.paste(image, (0, y))
            y += image.height
        stitched.save(image_path)
        stitched.close()

        if _is_low_information_image(image_path):
            image_path.unlink(missing_ok=True)
            return False
        return True
    finally:
        for image in rendered:
            image.close()


def render_pdf_page_to_image(
    pdf_path: str | Path,
    page_number: int,
    output_dir: str | Path = None,
    *,
    zoom: float = 2.0,
) -> Dict | None:
    """Render a whole page for known charts that span vector content."""
    if fitz is None:
        return None

    pdf_file = Path(pdf_path)
    image_dir = Path(output_dir or Config.IMAGES_EXTRACT_DIR)
    image_dir.mkdir(parents=True, exist_ok=True)
    with fitz.open(str(pdf_file)) as doc:
        if page_number < 1 or page_number > len(doc):
            return None
        page = doc[page_number - 1]
        filename = f"{_safe_stem(pdf_file.stem)}_p{page_number:03d}_render.png"
        image_path = image_dir / filename
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        pix.save(str(image_path))
        return {
            "kind": "page",
            "source_file": pdf_file.name,
            "page": page_number,
            "title": f"page_{page_number}",
            "index": 0,
            "xref": None,
            "path": str(image_path),
            "bbox": [[0.0, 0.0, float(page.rect.width), float(page.rect.height)]],
            "page_width": float(page.rect.width),
            "page_height": float(page.rect.height),
            "rendered_page": True,
        }
