# -*- coding: utf-8 -*-
"""Batch-run MinerU in WSL and adapt outputs to project chunks JSON."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List

from pdf_parser.mineru_backend import adapt_mineru_output, chunks_output_path, mineru_content_list_path, mineru_markdown_path
from src.config import Config
from src.utils.wsl_mineru import WSLMinerUError, run_mineru_for_pdf


def _safe_print(message: str = "") -> None:
    try:
        print(message)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        sys.stdout.buffer.write((message + "\n").encode(encoding, errors="replace"))


def _expand_pdf_inputs(inputs: Iterable[str]) -> List[Path]:
    pdfs: List[Path] = []
    seen: set[str] = set()
    for raw in inputs:
        path = Path(raw)
        candidates = sorted(path.glob("*.pdf")) if path.is_dir() else [path]
        for candidate in candidates:
            if candidate.suffix.lower() != ".pdf":
                continue
            resolved = candidate.resolve()
            key = str(resolved).lower()
            if key not in seen:
                pdfs.append(resolved)
                seen.add(key)
    return pdfs


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MinerU for one or more PDFs, then create *_chunks.json.")
    parser.add_argument("inputs", nargs="+", help="PDF files or directories containing PDF files")
    parser.add_argument("--output-dir", default=Config.MINERU_OUTPUT_DIR, help="MinerU output directory")
    parser.add_argument("--reuse-output", action="store_true", default=Config.MINERU_REUSE_OUTPUT)
    parser.add_argument("--no-reuse-output", action="store_false", dest="reuse_output")
    return parser


def main(argv: List[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    pdfs = _expand_pdf_inputs(args.inputs)
    if not pdfs:
        _safe_print("No PDF files found.")
        return 1

    output_dir = Path(args.output_dir)
    _safe_print(f"PDF count: {len(pdfs)}")
    _safe_print(f"Output dir: {output_dir}")

    failures = 0
    for index, pdf in enumerate(pdfs, start=1):
        _safe_print(f"[{index}/{len(pdfs)}] {pdf}")
        markdown = mineru_markdown_path(pdf, output_dir)
        content_list = mineru_content_list_path(pdf, output_dir)
        try:
            if not args.reuse_output or not (markdown.exists() and content_list.exists()):
                run_mineru_for_pdf(pdf, output_dir)
            outcome = adapt_mineru_output(pdf, output_dir=output_dir)
            result = outcome["result"]
            _safe_print(
                "  OK "
                f"chunks={result['chunk_count']} "
                f"tables={result['table_count']} "
                f"images={result['image_count']} "
                f"pages={result['page_count']}"
            )
            _safe_print(f"  markdown={result['mineru_markdown']}")
            _safe_print(f"  content_list={result['mineru_content_list']}")
            _safe_print(f"  chunks={chunks_output_path(pdf, output_dir)}")
        except (WSLMinerUError, Exception) as exc:
            failures += 1
            _safe_print(f"  FAILED: {exc}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
