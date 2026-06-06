# -*- coding: utf-8 -*-
"""Parse a large PDF with MinerU in page segments and merge the artifacts.

This keeps the parser backend as MinerU. It only avoids sending a very large
document to one MinerU API task, which can disconnect on constrained machines.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import fitz

from pdf_parser.mineru_backend import parse_mineru_pdf_file
from src.utils.wsl_mineru import parse_pdf_with_mineru_wsl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Segmented MinerU parser for large PDFs.")
    parser.add_argument("pdf", help="PDF file path")
    parser.add_argument("--output-dir", default="mineru_output", help="Final MinerU output directory")
    parser.add_argument("--work-dir", default="mineru_output_segments", help="Temporary segment output directory")
    parser.add_argument("--segment-pages", type=int, default=20, help="Pages per MinerU task")
    parser.add_argument("--device", default=os.getenv("MINERU_DEVICE", "cpu"), help="MinerU device")
    parser.add_argument("--timeout", type=float, default=1800, help="Timeout per segment in seconds")
    parser.add_argument("--distro", default=os.getenv("MINERU_WSL_DISTRO"), help="Optional WSL distro name")
    parser.add_argument("--conda-prefix", default=os.getenv("MINERU_CONDA_PREFIX", "/home/li/miniconda3"))
    parser.add_argument("--conda-env", default=os.getenv("MINERU_CONDA_ENV", "base"))
    parser.add_argument("--method", default=os.getenv("MINERU_METHOD", "auto"))
    parser.add_argument("--backend", default=os.getenv("MINERU_BACKEND", "pipeline"))
    parser.add_argument("--model-source", default=os.getenv("MINERU_MODEL_SOURCE", "modelscope"))
    parser.add_argument("--lang", default=os.getenv("MINERU_LANG", "ch"))
    parser.add_argument("--keep-work-dir", action="store_true")
    parser.add_argument("--merge-only", action="store_true", help="Merge existing segment artifacts without running MinerU")
    return parser


def _page_count(pdf: Path) -> int:
    with fitz.open(pdf) as doc:
        return doc.page_count


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _copy_images(segment_auto: Path, final_auto: Path) -> None:
    source = segment_auto / "images"
    target = final_auto / "images"
    if not source.exists():
        return
    target.mkdir(parents=True, exist_ok=True)
    for image in source.iterdir():
        if image.is_file():
            shutil.copy2(image, target / image.name)


def _find_artifact(auto_dir: Path, stem: str, suffix: str) -> Path:
    matches = sorted(auto_dir.glob(f"{stem}{suffix}"))
    if not matches:
        raise FileNotFoundError(f"Missing segment artifact: {auto_dir / (stem + suffix)}")
    return matches[-1]


def _offset_page_indexes(items: list, offset: int) -> list:
    adjusted = []
    for item in items:
        if isinstance(item, dict):
            copied = dict(item)
            if isinstance(copied.get("page_idx"), int):
                copied["page_idx"] = copied["page_idx"] + offset
            adjusted.append(copied)
        else:
            adjusted.append(item)
    return adjusted


def _merge_segments(pdf: Path, work_dir: Path, output_dir: Path, segment_ranges: list[tuple[int, int]]) -> None:
    stem = pdf.stem
    final_auto = output_dir / stem / "auto"
    final_auto.mkdir(parents=True, exist_ok=True)

    merged_content: list = []
    merged_v2: list = []
    markdown_parts: list[str] = []

    for start, end in segment_ranges:
        segment_auto = work_dir / f"{stem}_{start}_{end}" / stem / "auto"
        content_path = _find_artifact(segment_auto, stem, "_content_list.json")
        merged_content.extend(_offset_page_indexes(_load_json(content_path), start))

        v2_path = segment_auto / f"{stem}_content_list_v2.json"
        if v2_path.exists():
            v2_data = _load_json(v2_path)
            if isinstance(v2_data, list):
                merged_v2.extend(v2_data)

        md_path = segment_auto / f"{stem}.md"
        if md_path.exists():
            markdown_parts.append(md_path.read_text(encoding="utf-8").strip())

        _copy_images(segment_auto, final_auto)

    (final_auto / f"{stem}_content_list.json").write_text(
        json.dumps(merged_content, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if merged_v2:
        (final_auto / f"{stem}_content_list_v2.json").write_text(
            json.dumps(merged_v2, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    (final_auto / f"{stem}.md").write_text("\n\n".join(markdown_parts).strip() + "\n", encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    pdf = Path(args.pdf)
    output_dir = Path(args.output_dir)
    work_dir = Path(args.work_dir)
    total_pages = _page_count(pdf)
    ranges = [
        (start, min(start + args.segment_pages - 1, total_pages - 1))
        for start in range(0, total_pages, args.segment_pages)
    ]

    if not args.merge_only:
        for start, end in ranges:
            segment_output = work_dir / f"{pdf.stem}_{start}_{end}"
            print(f"[MinerU] segment {start}-{end} / {total_pages - 1}", flush=True)
            parse_pdf_with_mineru_wsl(
                pdf,
                output_dir=segment_output,
                distro=args.distro,
                conda_prefix=args.conda_prefix,
                conda_env=args.conda_env,
                method=args.method,
                backend=args.backend,
                lang=args.lang,
                start_page=start,
                end_page=end,
                device=args.device,
                extra_env={"MINERU_MODEL_SOURCE": args.model_source} if args.model_source else None,
                timeout=args.timeout,
            )

    _merge_segments(pdf, work_dir, output_dir, ranges)
    adapted = parse_mineru_pdf_file(pdf, output=args.output_dir, table_to_text=True, table_format="markdown")
    print(f"[MinerU] merged chunks: {adapted['output_path']}", flush=True)

    if not args.keep_work_dir:
        shutil.rmtree(work_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
