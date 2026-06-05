# -*- coding: utf-8 -*-
"""PyCharm-friendly entry point for calling MinerU in WSL."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from pdf_parser.mineru_wsl import parse_pdf_with_mineru_wsl, read_mineru_markdown


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Call WSL MinerU from PyCharm.")
    parser.add_argument("pdf_path", help="PDF path on Windows, for example data\\file.pdf.")
    parser.add_argument("-o", "--output", default="mineru_output", help="Windows output directory.")
    parser.add_argument("--conda-prefix", default="/home/li/miniconda3", help="Miniconda path in WSL.")
    parser.add_argument("--conda-env", default="base", help="Conda env name in WSL.")
    parser.add_argument("--wsl-distro", default="", help="Optional WSL distribution name.")
    parser.add_argument("-m", "--method", default="auto", choices=["auto", "ocr", "txt"], help="MinerU parse method.")
    parser.add_argument("--preview-chars", type=int, default=500, help="Print the first N markdown chars.")
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    result = parse_pdf_with_mineru_wsl(
        args.pdf_path,
        output_dir=args.output,
        conda_prefix=args.conda_prefix,
        conda_env=args.conda_env,
        wsl_distro=args.wsl_distro,
        method=args.method,
    )

    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    if result.markdown_path and args.preview_chars > 0:
        markdown = read_mineru_markdown(result)
        print("\n--- markdown preview ---")
        print(markdown[: args.preview_chars])
    else:
        print(f"\nNo markdown file found under {Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
