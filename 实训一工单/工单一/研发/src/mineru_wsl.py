# -*- coding: utf-8 -*-
"""Run MinerU installed in WSL from Windows Python."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from src.config import Config
from utils.logger import get_logger


logger = get_logger(__name__)


class MinerUError(RuntimeError):
    """Raised when MinerU cannot parse the requested PDF."""


@dataclass
class MinerUResult:
    pdf_path: Path
    output_dir: Path
    markdown_path: Path
    command: List[str]
    stdout: str = ""
    stderr: str = ""
    cached: bool = False


def _is_windows() -> bool:
    return os.name == "nt"


def _quote_for_bash(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _wsl_base_command() -> List[str]:
    command = ["wsl"]
    if Config.MINERU_WSL_DISTRO:
        command.extend(["-d", Config.MINERU_WSL_DISTRO])
    return command


def _run_wsl_bash(script: str, timeout: Optional[int] = None) -> subprocess.CompletedProcess:
    command = _wsl_base_command() + ["bash", "-lc", script]
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def windows_path_to_wsl(path: Path) -> str:
    """Convert a Windows path to a WSL path."""

    resolved = Path(path).resolve()
    if not _is_windows():
        return str(resolved)

    text = str(resolved)
    try:
        result = _run_wsl_bash(f"wslpath -a {_quote_for_bash(text)}", timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception as exc:
        logger.warning("wslpath failed, using manual path conversion | path=%s | error=%s", text, exc)

    drive = resolved.drive.rstrip(":").lower()
    if not drive:
        raise MinerUError(f"无法把 Windows 路径转换为 WSL 路径: {resolved}")
    relative = str(resolved)[3:].replace("\\", "/")
    return f"/mnt/{drive}/{relative}"


def _discover_mineru_command() -> str:
    prefix = Config.MINERU_CONDA_PREFIX.rstrip("/")
    script = (
        f"if test -x {_quote_for_bash(f'{prefix}/bin/mineru')}; then echo {_quote_for_bash(f'{prefix}/bin/mineru')}; "
        f"elif test -x {_quote_for_bash(f'{prefix}/bin/magic-pdf')}; then echo {_quote_for_bash(f'{prefix}/bin/magic-pdf')}; "
        "elif command -v mineru >/dev/null 2>&1; then command -v mineru; "
        "elif command -v magic-pdf >/dev/null 2>&1; then command -v magic-pdf; "
        "fi"
    )
    result = _run_wsl_bash(script, timeout=10)
    command = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    if not command:
        raise MinerUError(
            "没有在 WSL 中找到 MinerU 命令。请确认 /home/li/miniconda3/bin/mineru "
            "或 /home/li/miniconda3/bin/magic-pdf 存在。"
        )
    return command


def _find_markdown(output_dir: Path, pdf_stem: str) -> Optional[Path]:
    exact = output_dir / pdf_stem / Config.MINERU_METHOD / f"{pdf_stem}.md"
    if exact.exists():
        return exact

    exact_auto = output_dir / pdf_stem / "auto" / f"{pdf_stem}.md"
    if exact_auto.exists():
        return exact_auto

    matches = sorted(output_dir.glob(f"**/{pdf_stem}.md"), key=lambda item: item.stat().st_mtime, reverse=True)
    if matches:
        return matches[0]

    all_markdown = sorted(output_dir.glob("**/*.md"), key=lambda item: item.stat().st_mtime, reverse=True)
    return all_markdown[0] if all_markdown else None


def parse_pdf_with_mineru_wsl(
    pdf_path: str | Path,
    output_dir: str | Path | None = None,
    method: str | None = None,
    use_cache: bool | None = None,
    timeout: int | None = None,
) -> MinerUResult:
    """Parse a PDF by invoking MinerU inside WSL and return the Markdown file."""

    pdf_file = Path(pdf_path).resolve()
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF 不存在: {pdf_file}")

    output_root = Path(output_dir or Config.MINERU_OUTPUT_DIR).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    method = method or Config.MINERU_METHOD
    use_cache = Config.MINERU_USE_CACHE if use_cache is None else use_cache
    if use_cache:
        cached_markdown = _find_markdown(output_root, pdf_file.stem)
        if cached_markdown and cached_markdown.exists():
            return MinerUResult(
                pdf_path=pdf_file,
                output_dir=output_root,
                markdown_path=cached_markdown,
                command=[],
                cached=True,
            )

    if shutil.which("wsl") is None:
        raise MinerUError("当前 Windows 环境找不到 wsl.exe，请先确认 PyCharm 使用的是 Windows Python 且 WSL 已安装。")

    mineru_command = _discover_mineru_command()
    wsl_pdf = windows_path_to_wsl(pdf_file)
    wsl_output = windows_path_to_wsl(output_root)

    command_parts = [
        _quote_for_bash(mineru_command),
        "-p",
        _quote_for_bash(wsl_pdf),
        "-o",
        _quote_for_bash(wsl_output),
        "-m",
        _quote_for_bash(method),
    ]
    script = " ".join(command_parts)
    logger.info("mineru parse start | pdf=%s | output=%s | method=%s", pdf_file, output_root, method)
    result = _run_wsl_bash(script, timeout=timeout)
    if result.returncode != 0:
        raise MinerUError(
            "MinerU 解析失败。\n"
            f"command: {' '.join(_wsl_base_command() + ['bash', '-lc', script])}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    markdown = _find_markdown(output_root, pdf_file.stem)
    if markdown is None or not markdown.exists():
        raise MinerUError(f"MinerU 已运行，但没有找到 Markdown 输出: {output_root}")

    logger.info("mineru parse done | pdf=%s | markdown=%s", pdf_file, markdown)
    return MinerUResult(
        pdf_path=pdf_file,
        output_dir=output_root,
        markdown_path=markdown,
        command=_wsl_base_command() + ["bash", "-lc", script],
        stdout=result.stdout,
        stderr=result.stderr,
    )
