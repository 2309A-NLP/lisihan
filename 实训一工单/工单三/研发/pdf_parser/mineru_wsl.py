# -*- coding: utf-8 -*-
"""Run MinerU installed inside WSL from Windows/PyCharm Python."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List


DEFAULT_CONDA_PREFIX = os.getenv("MINERU_WSL_CONDA_PREFIX", "/home/li/miniconda3")
DEFAULT_CONDA_ENV = os.getenv("MINERU_WSL_CONDA_ENV", "base")
DEFAULT_WSL_DISTRO = os.getenv("MINERU_WSL_DISTRO", "")
DEFAULT_OUTPUT_DIR = os.getenv("MINERU_OUTPUT_DIR", "mineru_output")


@dataclass
class MinerUResult:
    source_pdf: str
    output_dir: str
    markdown_path: str | None
    content_list_path: str | None
    middle_json_path: str | None
    model_json_path: str | None
    used_cache: bool
    stdout: str
    stderr: str


class MinerUWSLError(RuntimeError):
    """Raised when the WSL MinerU command fails."""


def _run_command(args: List[str], *, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def _wsl_base_args(wsl_distro: str = "") -> List[str]:
    args = ["wsl"]
    if wsl_distro:
        args.extend(["-d", wsl_distro])
    return args


def windows_path_to_wsl(path: str | Path, *, wsl_distro: str = "") -> str:
    """Convert a Windows path such as C:\\data\\a.pdf to /mnt/c/data/a.pdf."""
    completed = _run_command(
        [*_wsl_base_args(wsl_distro), "wslpath", "-a", str(Path(path).resolve())],
        timeout=15,
    )
    if completed.returncode != 0:
        raise MinerUWSLError(
            "Failed to convert Windows path to WSL path.\n"
            f"stderr: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def _quote(value: str) -> str:
    return shlex.quote(value)


def _build_mineru_script(
    pdf_wsl_path: str,
    output_wsl_dir: str,
    *,
    conda_prefix: str,
    conda_env: str,
    method: str,
    device: str | None,
) -> str:
    activate = ""
    conda_sh = f"{conda_prefix.rstrip('/')}/etc/profile.d/conda.sh"
    if conda_env:
        activate = (
            f"if [ -f {_quote(conda_sh)} ]; then "
            f". {_quote(conda_sh)} && conda activate {_quote(conda_env)}; "
            "fi"
        )

    device_prefix = f"CUDA_VISIBLE_DEVICES={_quote(device)} " if device else ""
    common_args = f"-p {_quote(pdf_wsl_path)} -o {_quote(output_wsl_dir)} -m {_quote(method)}"
    mineru_bin = f"{conda_prefix.rstrip('/')}/bin/mineru"
    magic_pdf_bin = f"{conda_prefix.rstrip('/')}/bin/magic-pdf"

    return "\n".join(
        [
            "set -e",
            activate,
            f"mkdir -p {_quote(output_wsl_dir)}",
            "if command -v mineru >/dev/null 2>&1; then",
            f"  {device_prefix}mineru {common_args}",
            "elif command -v magic-pdf >/dev/null 2>&1; then",
            f"  {device_prefix}magic-pdf {common_args}",
            f"elif [ -x {_quote(mineru_bin)} ]; then",
            f"  {device_prefix}{_quote(mineru_bin)} {common_args}",
            f"elif [ -x {_quote(magic_pdf_bin)} ]; then",
            f"  {device_prefix}{_quote(magic_pdf_bin)} {common_args}",
            "else",
            "  echo 'MinerU executable not found. Check MINERU_WSL_CONDA_ENV or PATH.' >&2",
            "  exit 127",
            "fi",
        ]
    )


def _first_existing(paths: List[Path]) -> str | None:
    for path in paths:
        if path.exists():
            return str(path)
    return None


def _find_mineru_outputs(pdf_path: Path, output_dir: Path) -> dict[str, str | None]:
    stem = pdf_path.stem
    preferred_dir = output_dir / stem / "auto"

    markdown_path = _first_existing([preferred_dir / f"{stem}.md"])
    content_list_path = _first_existing(
        [
            preferred_dir / f"{stem}_content_list.json",
            preferred_dir / f"{stem}_content_list_v2.json",
        ]
    )
    middle_json_path = _first_existing([preferred_dir / f"{stem}_middle.json"])
    model_json_path = _first_existing([preferred_dir / f"{stem}_model.json"])

    if markdown_path is None:
        candidates = [
            item
            for item in output_dir.rglob("*.md")
            if item.stem == stem or item.parent.parent.name == stem or item.parent.name == stem
        ]
        candidates = sorted(candidates, key=lambda item: (item.name != f"{stem}.md", len(item.parts)))
        markdown_path = str(candidates[0]) if candidates else None

    if content_list_path is None:
        candidates = [
            item
            for item in output_dir.rglob("*content_list*.json")
            if item.name.startswith(f"{stem}_") or item.parent.parent.name == stem or item.parent.name == stem
        ]
        candidates = sorted(candidates, key=lambda item: len(item.parts))
        content_list_path = str(candidates[0]) if candidates else None

    return {
        "markdown_path": markdown_path,
        "content_list_path": content_list_path,
        "middle_json_path": middle_json_path,
        "model_json_path": model_json_path,
    }


def parse_pdf_with_mineru_wsl(
    pdf_path: str | Path,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    conda_prefix: str = DEFAULT_CONDA_PREFIX,
    conda_env: str = DEFAULT_CONDA_ENV,
    wsl_distro: str = DEFAULT_WSL_DISTRO,
    method: str = "auto",
    device: str | None = None,
    timeout: int = 3600,
    use_cache: bool = True,
) -> MinerUResult:
    """Parse a PDF with MinerU inside WSL and return the generated files."""
    pdf_file = Path(pdf_path).resolve()
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_file}")

    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    cached_outputs = _find_mineru_outputs(pdf_file, output_path)
    if use_cache and (cached_outputs.get("markdown_path") or cached_outputs.get("content_list_path")):
        return MinerUResult(
            source_pdf=str(pdf_file),
            output_dir=str(output_path),
            stdout="Using cached MinerU output.",
            stderr="",
            used_cache=True,
            **cached_outputs,
        )

    pdf_wsl_path = windows_path_to_wsl(pdf_file, wsl_distro=wsl_distro)
    output_wsl_dir = windows_path_to_wsl(output_path, wsl_distro=wsl_distro)
    script = _build_mineru_script(
        pdf_wsl_path,
        output_wsl_dir,
        conda_prefix=conda_prefix,
        conda_env=conda_env,
        method=method,
        device=device,
    )

    completed = _run_command(
        [*_wsl_base_args(wsl_distro), "bash", "-lc", script],
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise MinerUWSLError(
            "MinerU failed in WSL.\n"
            f"returncode: {completed.returncode}\n"
            f"stdout: {completed.stdout.strip()}\n"
            f"stderr: {completed.stderr.strip()}"
        )

    outputs = _find_mineru_outputs(pdf_file, output_path)
    return MinerUResult(
        source_pdf=str(pdf_file),
        output_dir=str(output_path),
        stdout=completed.stdout,
        stderr=completed.stderr,
        used_cache=False,
        **outputs,
    )


def read_mineru_markdown(result: MinerUResult) -> str:
    if not result.markdown_path:
        return ""
    return Path(result.markdown_path).read_text(encoding="utf-8")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run WSL MinerU from Windows Python.")
    parser.add_argument("pdf_path", help="PDF file path on Windows.")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT_DIR, help="Output directory on Windows.")
    parser.add_argument("--conda-prefix", default=DEFAULT_CONDA_PREFIX, help="Miniconda/Anaconda prefix in WSL.")
    parser.add_argument("--conda-env", default=DEFAULT_CONDA_ENV, help="Conda environment name in WSL.")
    parser.add_argument("--wsl-distro", default=DEFAULT_WSL_DISTRO, help="Optional WSL distribution name.")
    parser.add_argument("-m", "--method", default="auto", choices=["auto", "ocr", "txt"], help="MinerU parse method.")
    parser.add_argument("--device", default=None, help="Optional CUDA_VISIBLE_DEVICES value, for example 0.")
    parser.add_argument("--timeout", type=int, default=3600, help="MinerU timeout in seconds.")
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    result = parse_pdf_with_mineru_wsl(
        args.pdf_path,
        output_dir=args.output,
        conda_prefix=args.conda_prefix,
        conda_env=args.conda_env,
        wsl_distro=args.wsl_distro,
        method=args.method,
        device=args.device,
        timeout=args.timeout,
        use_cache=True,
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
