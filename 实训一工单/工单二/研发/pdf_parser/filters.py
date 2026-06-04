# -*- coding: utf-8 -*-
"""Noise filters for PDF parsing."""

from __future__ import annotations

import re
from typing import Iterable, Optional, Sequence, Tuple

COMPANY_NAME_PATTERNS = [
    "武汉兴图新科电子股份有限公司",
]

NOISE_PATTERNS = [
    re.compile(r"^=+\s*Page\s+\d+\s*=+$", re.IGNORECASE),
    re.compile(r"^\d+-\d+-[A-Za-z0-9]+$"),
    re.compile(r"^\d+\s*/\s*\d+$"),
    re.compile(r"^第\s*\d+\s*页$"),
    re.compile(r"^image\[\[\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\]\]$", re.IGNORECASE),
]

HEADER_TITLE_PATTERNS = [
    "招股说明书",
]


def normalize_text(text: str) -> str:
    text = text or ""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def bbox_to_list(bbox: Optional[Sequence[float]]) -> list[float]:
    if not bbox:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(v) for v in bbox[:4]]


def _is_top_or_bottom_noise(
    bbox: Optional[Sequence[float]],
    page_width: Optional[float],
    page_height: Optional[float],
) -> bool:
    if not bbox or not page_height:
        return False
    _, y0, _, y1 = bbox[:4]
    header_zone = max(60.0, page_height * 0.12)
    footer_zone = min(page_height - 60.0, page_height * 0.88)
    return y1 <= header_zone or y0 >= footer_zone


def should_drop_text_block(
    text: str,
    bbox: Optional[Sequence[float]] = None,
    page_width: Optional[float] = None,
    page_height: Optional[float] = None,
) -> bool:
    normalized = normalize_text(text)
    if not normalized or len(normalized) < 5:
        return True

    if any(pattern.fullmatch(normalized) for pattern in NOISE_PATTERNS):
        return True

    if normalized in COMPANY_NAME_PATTERNS:
        return True

    if any(marker in normalized for marker in HEADER_TITLE_PATTERNS) and _is_top_or_bottom_noise(
        bbox, page_width, page_height
    ):
        return True

    if _is_top_or_bottom_noise(bbox, page_width, page_height):
        if any(marker in normalized for marker in COMPANY_NAME_PATTERNS):
            return True
        if any(pattern in normalized for pattern in HEADER_TITLE_PATTERNS):
            return True
        if re.fullmatch(r"[A-Za-z0-9一-龥\s·\-_,.()（）]{1,40}", normalized):
            # Very short repetitive header/footer-style text near edges.
            return True

    return False


def is_image_block(text: str) -> bool:
    normalized = normalize_text(text)
    return bool(re.fullmatch(r"image\[\[[^\]]+\]\]", normalized, re.IGNORECASE))

