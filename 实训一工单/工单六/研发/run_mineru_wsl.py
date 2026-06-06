# -*- coding: utf-8 -*-
"""PyCharm entry point for parsing PDFs with MinerU installed in WSL.

Examples:
    python run_mineru_wsl.py data/招股说明书1.pdf
    python run_mineru_wsl.py data/招股说明书1.pdf --output-dir mineru_output
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from pdf_parser.mineru_backend import parse_mineru_pdf_file
from src.utils.wsl_mineru import MinerUError, parse_pdf_with_mineru_wsl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Call WSL MinerU from Windows/PyCharm.")
    parser.add_argument(
        "pdf",
        nargs="+",
        help="PDF file(s) or directory, for example data or data/招股说明书1.pdf",
    )
    parser.add_argument("--output-dir", default="mineru_output", help="Windows output directory")
    parser.add_argument("--distro", default=os.getenv("MINERU_WSL_DISTRO"), help="Optional WSL distro name")
    parser.add_argument(
        "--conda-prefix",
        default=os.getenv("MINERU_CONDA_PREFIX", "/home/li/miniconda3"),
        help="Conda install path inside WSL",
    )
    parser.add_argument(
        "--conda-env",
        default=os.getenv("MINERU_CONDA_ENV", "base"),
        help="Conda environment name inside WSL",
    )
    parser.add_argument("--method", default=os.getenv("MINERU_METHOD", "auto"), help="MinerU parse method")
    parser.add_argument("--backend", default=os.getenv("MINERU_BACKEND", "pipeline"), help="MinerU backend, optional")
    parser.add_argument("--device", default=os.getenv("MINERU_DEVICE"), help="MinerU device, for example cpu or cuda")
    parser.add_argument(
        "--model-source",
        default=os.getenv("MINERU_MODEL_SOURCE", "modelscope"),
        help="MinerU model source, for example modelscope or huggingface",
    )
    parser.add_argument("--lang", default=os.getenv("MINERU_LANG", "ch"), help="OCR language")
    parser.add_argument("--start-page", type=int, default=None, help="Optional first page")
    parser.add_argument("--end-page", type=int, default=None, help="Optional last page")
    parser.add_argument("--timeout", type=float, default=None, help="Timeout in seconds")
    parser.add_argument(
        "--command-template",
        default=os.getenv("MINERU_COMMAND_TEMPLATE"),
        help="Custom command template. Available fields: {input}, {output}, {method}, {options}",
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Extra CLI argument passed to MinerU. Repeat for multiple args.",
    )
    return parser


def _expand_pdf_inputs(inputs: list[str]) -> list[Path]:
    pdfs: list[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            pdfs.extend(sorted(path.glob("*.pdf")))
        else:
            pdfs.append(path)
    return pdfs


def main() -> int:
    args = build_parser().parse_args()

    pdfs = _expand_pdf_inputs(args.pdf)
    if not pdfs:
        print("[MinerU] failed: no PDF files found")
        return 1

    failures = 0
    for pdf in pdfs:
        try:
            result = parse_pdf_with_mineru_wsl(
                pdf,
                output_dir=args.output_dir,
                distro=args.distro,
                conda_prefix=args.conda_prefix,
                conda_env=args.conda_env,
                method=args.method,
                backend=args.backend,
                device=args.device,
                lang=args.lang,
                start_page=args.start_page,
                end_page=args.end_page,
                extra_args=args.extra_arg,
                extra_env={"MINERU_MODEL_SOURCE": args.model_source} if args.model_source else None,
                command_template=args.command_template,
                timeout=args.timeout,
            )
        except (MinerUError, FileNotFoundError, ValueError) as exc:
            failures += 1
            print(f"[MinerU] failed: {pdf}")
            print(exc)
            continue

        print("[MinerU] success")
        print(f"PDF: {result.input_pdf}")
        print(f"Output: {result.output_dir}")
        print(f"Command: {result.command}")
        print(f"Markdown: {result.markdown_path or 'not found'}")
        print(f"Content list: {result.content_list_path or 'not found'}")
        adapted = parse_mineru_pdf_file(
            pdf,
            output=args.output_dir,
            table_to_text=True,
            table_format="markdown",
        )
        print(f"Chunks JSON: {adapted['output_path']}")
        if result.stderr.strip():
            print("\n[MinerU stderr]")
            print(result.stderr.strip())
        print()

    print(
        f"[MinerU] done: success={len(pdfs) - failures} "
        f"failed={failures} total={len(pdfs)}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
