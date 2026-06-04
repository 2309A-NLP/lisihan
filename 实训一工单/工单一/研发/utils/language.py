# -*- coding: utf-8 -*-
"""Small language helpers for Chinese/English QA flows."""

from __future__ import annotations

import re


def detect_language(text: str) -> str:
    """Return ``en`` for mostly English text, otherwise ``zh``."""
    text = text or ""
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    english_words = len(re.findall(r"[A-Za-z]{2,}", text))
    if english_words > 0 and english_words >= chinese_chars:
        return "en"
    return "zh"


def is_english(text: str) -> bool:
    return detect_language(text) == "en"
