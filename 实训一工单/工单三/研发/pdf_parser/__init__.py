# -*- coding: utf-8 -*-
"""Standalone PDF parser package."""

__all__ = [
    "MinerUResult",
    "MinerUWSLError",
    "parse_pdf_with_mineru_wsl",
    "read_mineru_markdown",
]


def __getattr__(name: str):
    if name in __all__:
        from . import mineru_wsl

        return getattr(mineru_wsl, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
