# -*- coding: utf-8 -*-
"""Helpers for invoking MinerU from Windows through WSL."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import locale
from pathlib import Path
from typing import Iterable

from src.config import Config


class MinerUExecutionError(RuntimeError):
    """Raised when the MinerU command cannot complete successfully."""


def windows_path_to_wsl(path: str | Path) -> str:
    """Convert a Windows path to a WSL path while preserving non-ASCII names."""
    resolved = Path(path).resolve()
    value = str(resolved)
    match = re.match(r"^([A-Za-z]):\\(.*)$", value)
    if not match:
        return value.replace("\\", "/")
    drive = match.group(1).lower()
    rest = match.group(2).replace("\\", "/")
    return f"/mnt/{drive}/{rest}"


def _wsl_prefix() -> list[str]:
    command = ["wsl.exe"]
    distro = getattr(Config, "MINERU_WSL_DISTRO", "")
    if distro:
        command.extend(["-d", distro])
    return command


def _shell_script(command: Iterable[str]) -> str:
    conda_root = getattr(Config, "MINERU_CONDA_ROOT", "~/miniconda3") or "~/miniconda3"
    conda_env = getattr(Config, "MINERU_CONDA_ENV", "") or ""
    if conda_root.startswith("~/"):
        conda_root_assignment = f'CONDA_ROOT="$HOME/{conda_root[2:]}"'
    else:
        conda_root_assignment = f"CONDA_ROOT={shlex.quote(conda_root)}"
    exports = [
        "set -e",
        "export LANG=C.UTF-8",
        "export LC_ALL=C.UTF-8",
        "export PYTHONIOENCODING=utf-8",
        f"export MINERU_MODEL_SOURCE={shlex.quote(Config.MINERU_MODEL_SOURCE)}",
        conda_root_assignment,
        'export PATH="$CONDA_ROOT/condabin:$CONDA_ROOT/bin:$PATH"',
    ]
    if conda_env:
        exports.append('source "$CONDA_ROOT/etc/profile.d/conda.sh"')
        exports.append(f"conda activate {shlex.quote(conda_env)}")
    exports.append(" ".join(shlex.quote(item) for item in command))
    return "\n".join(exports)


def _diagnostic_script(command_name: str) -> str:
    conda_root = getattr(Config, "MINERU_CONDA_ROOT", "~/miniconda3") or "~/miniconda3"
    conda_env = getattr(Config, "MINERU_CONDA_ENV", "") or ""
    if conda_root.startswith("~/"):
        conda_root_assignment = f'CONDA_ROOT="$HOME/{conda_root[2:]}"'
    else:
        conda_root_assignment = f"CONDA_ROOT={shlex.quote(conda_root)}"

    lines = [
        "set +e",
        "export LANG=C.UTF-8",
        "export LC_ALL=C.UTF-8",
        "export PYTHONIOENCODING=utf-8",
        conda_root_assignment,
        'export PATH="$CONDA_ROOT/condabin:$CONDA_ROOT/bin:$HOME/.local/bin:$PATH"',
    ]
    if conda_env:
        lines.append('test -f "$CONDA_ROOT/etc/profile.d/conda.sh" && source "$CONDA_ROOT/etc/profile.d/conda.sh"')
        lines.append(f"conda activate {shlex.quote(conda_env)} 2>/dev/null")
    lines.extend(
        [
            "echo MINERU_DIAGNOSTIC_START",
            "echo PATH=$PATH",
            "echo SHELL=$SHELL",
            "command -v python3 || true",
            "python3 --version 2>/dev/null || true",
            f"command -v {shlex.quote(command_name)} || true",
            "command -v magic-pdf || true",
            "python3 -m pip show mineru magic-pdf 2>/dev/null || true",
            "echo MINERU_DIAGNOSTIC_END",
        ]
    )
    return "\n".join(lines)


def _run_wsl_bash(script: str, env: dict[str, str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run([*_wsl_prefix(), "bash", "-lc", script], capture_output=True, env=env)


def _decode_process_output(data: bytes | None) -> str:
    if not data:
        return ""
    candidates = []
    if data.count(b"\x00") > max(2, len(data) // 8):
        candidates.extend(["utf-16-le", "utf-16"])
    candidates.extend(["utf-8", locale.getpreferredencoding(False), "gbk", "cp936"])

    seen = set()
    best = ""
    best_score = -1
    for encoding in candidates:
        if not encoding or encoding in seen:
            continue
        seen.add(encoding)
        try:
            text = data.decode(encoding, errors="replace")
        except LookupError:
            continue
        score = len(text) - text.count("\ufffd") * 8 - text.count("\x00") * 4
        if score > best_score:
            best = text
            best_score = score
    return best.replace("\x00", "").strip()


def _wsl_setup_help(stdout: str, stderr: str) -> str:
    combined = f"{stdout}\n{stderr}".lower()
    if (
        "no installed distributions" not in combined
        and "没有已安装的分发" not in combined
        and "wslregisterdistribution failed" not in combined
    ):
        return ""
    return (
        "\n\nWSL setup help:\n"
        "1. Install a Linux distribution for WSL, for example: wsl.exe --install Ubuntu\n"
        "2. Open that distribution once and finish its first-time setup.\n"
        "3. Install MinerU inside the WSL Linux environment.\n"
        "4. If you use a named distribution, set MINERU_WSL_DISTRO in .env.\n"
    )


def run_mineru_wsl(pdf_path: str | Path, output_dir: str | Path) -> None:
    pdf_wsl = windows_path_to_wsl(pdf_path)
    output_wsl = windows_path_to_wsl(output_dir)
    mineru_cli = getattr(Config, "MINERU_CLI", "mineru") or "mineru"
    parse_method = getattr(Config, "MINERU_PARSE_METHOD", "auto") or "auto"

    command = [
        mineru_cli,
        "-p",
        pdf_wsl,
        "-o",
        output_wsl,
        "-m",
        parse_method,
        "-b",
        Config.MINERU_BACKEND,
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env["MINERU_MODEL_SOURCE"] = Config.MINERU_MODEL_SOURCE
    try:
        process = _run_wsl_bash(_shell_script(command), env)
    except FileNotFoundError as exc:
        raise MinerUExecutionError(
            "MinerU WSL execution could not start. "
            "Please confirm WSL is installed and wsl.exe is available in PATH.\n"
            f"pdf={Path(pdf_path)}\n"
            f"output_dir={Path(output_dir)}\n"
            f"backend={Config.MINERU_BACKEND}\n"
            f"model_source={Config.MINERU_MODEL_SOURCE}"
        ) from exc
    if process.returncode != 0:
        stdout = _decode_process_output(process.stdout)
        stderr = _decode_process_output(process.stderr)
        diagnostic = _wsl_setup_help(stdout, stderr)
        if process.returncode == 127 or "command not found" in stderr.lower():
            try:
                check = _run_wsl_bash(_diagnostic_script(mineru_cli), env)
                diagnostic_stdout = _decode_process_output(check.stdout)
                diagnostic_stderr = _decode_process_output(check.stderr)
                diagnostic = (
                    "\n\nMinerU environment diagnostic:\n"
                    f"{diagnostic_stdout}\n"
                    f"{diagnostic_stderr}".strip()
                    + "\n"
                )
            except Exception:
                diagnostic = ""
            diagnostic += (
                "\nHow to fix:\n"
                "1. If you want to use MinerU, install and initialize a WSL Linux distribution, then install MinerU "
                "inside that Linux environment.\n"
                "2. If MinerU is installed in a conda env, set MINERU_CONDA_ENV and MINERU_CONDA_ROOT in .env.\n"
                "3. If the executable name is different, set MINERU_CLI in .env, for example MINERU_CLI=magic-pdf.\n"
                "4. To run the project without MinerU on this machine, set PDF_PARSER_BACKEND=local in .env.\n"
            )
        message = (
            "MinerU WSL execution failed.\n"
            f"pdf={Path(pdf_path)}\n"
            f"output_dir={Path(output_dir)}\n"
            f"backend={Config.MINERU_BACKEND}\n"
            f"model_source={Config.MINERU_MODEL_SOURCE}\n"
            f"returncode={process.returncode}\n"
            f"stdout:\n{stdout}\n"
            f"stderr:\n{stderr}"
            f"{diagnostic}"
        )
        raise MinerUExecutionError(message)
