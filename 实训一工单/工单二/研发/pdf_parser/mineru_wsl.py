# -*- coding: utf-8 -*-
"""Run MinerU installed in WSL and adapt its output to local parser blocks."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Sequence


DEFAULT_WSL_DISTRO = os.getenv("MINERU_WSL_DISTRO", "").strip()
DEFAULT_CONDA_ROOT = os.getenv("MINERU_CONDA_ROOT", "/home/li/miniconda3")
DEFAULT_CONDA_ENV = os.getenv("MINERU_CONDA_ENV", "base")
DEFAULT_COMMAND = os.getenv("MINERU_COMMAND", "mineru")
DEFAULT_BACKEND = os.getenv("MINERU_BACKEND", "").strip()
DEFAULT_MODEL_SOURCE = os.getenv("MINERU_MODEL_SOURCE", "").strip()
DEFAULT_TIMEOUT = int(os.getenv("MINERU_TIMEOUT", "3600"))


class MinerUWSLError(RuntimeError):
    """Raised when WSL or MinerU execution fails."""


def _run_wsl(args: Sequence[str], *, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    command = ["wsl"]
    if DEFAULT_WSL_DISTRO:
        command.extend(["-d", DEFAULT_WSL_DISTRO])
    command.extend(args)
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def windows_path_to_wsl(path: str | Path) -> str:
    """Convert a Windows path into the path visible from WSL."""
    resolved = str(Path(path).resolve())
    if len(resolved) >= 3 and resolved[1:3] in {":\\", ":/"}:
        drive = resolved[0].lower()
        tail = resolved[3:].replace("\\", "/")
        return f"/mnt/{drive}/{tail}"

    proc = _run_wsl(["wslpath", "-a", resolved], timeout=30)
    if proc.returncode != 0:
        details = (proc.stderr or proc.stdout or "").strip()
        raise MinerUWSLError(f"wslpath failed for {resolved}: {details}")
    return proc.stdout.strip()


def _build_mineru_script(input_path: str, output_dir: str) -> str:
    exports = []
    if DEFAULT_MODEL_SOURCE:
        exports.append(f"export MINERU_MODEL_SOURCE={shlex.quote(DEFAULT_MODEL_SOURCE)}")

    activate = (
        f"source {shlex.quote(DEFAULT_CONDA_ROOT + '/etc/profile.d/conda.sh')} && "
        f"conda activate {shlex.quote(DEFAULT_CONDA_ENV)}"
    )
    command = [
        DEFAULT_COMMAND,
        "-p",
        input_path,
        "-o",
        output_dir,
    ]
    if DEFAULT_BACKEND:
        command.extend(["-b", DEFAULT_BACKEND])

    lines = [
        "set -e",
        *exports,
        activate,
        " ".join(shlex.quote(part) for part in command),
    ]
    return "\n".join(lines)


def run_mineru_in_wsl(
    pdf_path: str | Path,
    output_dir: str | Path,
    *,
    timeout: int = DEFAULT_TIMEOUT,
) -> subprocess.CompletedProcess[str]:
    """Invoke MinerU in WSL for a local Windows PDF path."""
    pdf_wsl_path = windows_path_to_wsl(pdf_path)
    output_wsl_dir = windows_path_to_wsl(output_dir)
    script = _build_mineru_script(pdf_wsl_path, output_wsl_dir)
    proc = _run_wsl(["bash", "-lc", script], timeout=timeout)
    if proc.returncode != 0:
        details = (proc.stderr or proc.stdout or "").strip()
        if "huggingface.co" in details or "LocalEntryNotFoundError" in details:
            details = (
                "MinerU could not find/download its model files. "
                "The log shows HuggingFace is unreachable from WSL. "
                "Use MINERU_MODEL_SOURCE=modelscope, or pre-download MinerU models into the WSL cache. "
                f"Original error:\n{details}"
            )
        raise MinerUWSLError(f"MinerU failed with exit code {proc.returncode}: {details}")
    return proc


def _find_latest_content_list(output_dir: Path, pdf_stem: str) -> Path:
    candidates = list(output_dir.rglob(f"{pdf_stem}_content_list.json"))
    if not candidates:
        candidates = list(output_dir.rglob("*_content_list.json"))
    if not candidates:
        raise MinerUWSLError(f"No MinerU content_list JSON found under {output_dir}")
    return max(candidates, key=lambda item: item.stat().st_mtime)


def find_existing_content_list(output_root: str | Path, pdf_stem: str) -> Path | None:
    output_dir = Path(output_root)
    candidates = list(output_dir.rglob(f"{pdf_stem}_content_list.json"))
    if not candidates:
        candidates = list(output_dir.rglob("*_content_list.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.stat().st_mtime)


def _find_latest_markdown(output_dir: Path, pdf_stem: str) -> Path | None:
    candidates = list(output_dir.rglob(f"{pdf_stem}.md"))
    if not candidates:
        candidates = list(output_dir.rglob("*.md"))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.stat().st_mtime)


def _normalize_bbox(value: object) -> List[float]:
    if not isinstance(value, list) or len(value) < 4:
        return [0.0, 0.0, 1000.0, 1000.0]
    return [float(item) for item in value[:4]]


def _as_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [_as_text(item) for item in value]
        return "\n".join(part for part in parts if part).strip()
    if isinstance(value, dict):
        for key in ("text", "content", "html", "latex", "markdown"):
            text = _as_text(value.get(key))
            if text:
                return text
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def _extract_content(item: Dict) -> str:
    block_type = str(item.get("type", "")).lower()
    if block_type == "list":
        text = _as_text(item.get("list_items"))
        if text:
            return text
    if block_type == "code":
        text = _as_text(item.get("code_body"))
        if text:
            return text
    if block_type == "table":
        for key in ("table_body", "content", "text", "html"):
            text = _as_text(item.get(key))
            if text:
                return text
    if block_type in {"image", "chart"}:
        for key in ("content", "img_caption", "chart_caption", "img_path"):
            text = _as_text(item.get(key))
            if text:
                return text
    for key in ("text", "content", "equation", "latex"):
        text = _as_text(item.get(key))
        if text:
            return text
    return ""


def _block_type(item: Dict) -> str:
    item_type = str(item.get("type", "text")).lower()
    if item_type in {"table", "image", "chart", "equation", "code", "list"}:
        return item_type
    return "text"


def load_mineru_blocks(content_list_path: str | Path, source_pdf: str | Path) -> List[Dict]:
    """Load MinerU content_list.json and convert it to this project block format."""
    content_file = Path(content_list_path)
    pdf_file = Path(source_pdf)
    content_list = json.loads(content_file.read_text(encoding="utf-8"))
    if not isinstance(content_list, list):
        raise MinerUWSLError(f"Unexpected MinerU content list format: {content_file}")

    blocks: List[Dict] = []
    for index, item in enumerate(content_list, start=1):
        if not isinstance(item, dict):
            continue
        content = _extract_content(item)
        if not content:
            continue

        page = int(item.get("page_idx", 0)) + 1
        block_type = _block_type(item)
        bbox = _normalize_bbox(item.get("bbox"))
        metadata = {
            "source_file": pdf_file.name,
            "source_path": str(pdf_file),
            "mineru_content_index": index,
            "mineru_type": item.get("type", block_type),
            "char_count": len(content),
            "has_table": block_type == "table",
            "bbox_unit": "normalized_0_1000",
        }
        if item.get("text_level") is not None:
            metadata["text_level"] = item.get("text_level")
        if item.get("sub_type") is not None:
            metadata["sub_type"] = item.get("sub_type")

        blocks.append(
            {
                "page": page,
                "type": "table" if block_type == "table" else "text",
                "content": content,
                "bbox": bbox,
                "metadata": metadata,
            }
        )
    return blocks


def extract_blocks_with_mineru(
    pdf_path: str | Path,
    *,
    output_root: str | Path,
    timeout: int = DEFAULT_TIMEOUT,
) -> Dict:
    """Run MinerU and return parser blocks plus raw output paths."""
    pdf_file = Path(pdf_path)
    raw_root = Path(output_root) / "mineru_raw" / f"{pdf_file.stem}_{int(time.time())}"
    raw_root.mkdir(parents=True, exist_ok=True)

    proc = run_mineru_in_wsl(pdf_file, raw_root, timeout=timeout)
    content_list_path = _find_latest_content_list(raw_root, pdf_file.stem)
    markdown_path = _find_latest_markdown(raw_root, pdf_file.stem)
    exported_markdown_path = None
    if markdown_path is not None:
        exported_markdown_path = Path(output_root) / f"{pdf_file.stem}.md"
        shutil.copy2(markdown_path, exported_markdown_path)

    blocks = load_mineru_blocks(content_list_path, pdf_file)
    table_blocks = [block for block in blocks if block.get("type") == "table"]
    text_blocks = [block for block in blocks if block.get("type") != "table"]
    return {
        "text_blocks": text_blocks,
        "table_blocks": table_blocks,
        "content_list_path": str(content_list_path),
        "markdown_path": str(markdown_path) if markdown_path else "",
        "exported_markdown_path": str(exported_markdown_path) if exported_markdown_path else "",
        "raw_output_dir": str(raw_root),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
