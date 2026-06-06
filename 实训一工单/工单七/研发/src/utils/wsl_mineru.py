# -*- coding: utf-8 -*-
"""Run MinerU in WSL from Windows paths."""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import List

from src.config import Config


class WSLMinerUError(RuntimeError):
    """Raised when WSL or MinerU returns an error."""


def _decode_output(data: bytes | None) -> str:
    if not data:
        return ""
    if b"\x00" in data[:200]:
        try:
            return data.decode("utf-16-le")
        except UnicodeDecodeError:
            pass
    for encoding in ("utf-8", "utf-8-sig", "gb18030", "gbk"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def windows_path_to_wsl(path: str | Path) -> str:
    resolved = Path(path).resolve()
    if os.name != "nt":
        return str(resolved)
    drive = resolved.drive.rstrip(":").lower()
    rest = resolved.as_posix()[3:]
    return f"/mnt/{drive}/{rest}"


def _wsl_base_command() -> List[str]:
    command = ["wsl"]
    distro = str(getattr(Config, "MINERU_WSL_DISTRO", "") or "").strip()
    if distro:
        command.extend(["-d", distro])
    return command


def _run_wsl_script(script: str) -> str:
    command = _wsl_base_command() + ["bash", "-lc", script]
    try:
        completed = subprocess.run(command, capture_output=True, check=False)
    except FileNotFoundError as exc:
        raise WSLMinerUError("WSL executable was not found. Please install/enable WSL first.") from exc

    stdout = _decode_output(completed.stdout)
    stderr = _decode_output(completed.stderr)
    if completed.returncode != 0:
        details = "\n".join(part for part in [stdout.strip(), stderr.strip()] if part)
        raise WSLMinerUError(
            f"MinerU failed in WSL with exit code {completed.returncode}.\n"
            f"Command: {' '.join(command)}\n"
            f"Output:\n{details}"
        )
    return stdout


def _build_mineru_script(
    *,
    pdf_wsl: str,
    output_wsl: str,
    conda_root: str,
    conda_env: str,
    command: str,
    method: str,
    backend: str,
    model_source: str,
    device: str,
    processing_window_size: int,
    max_concurrent_requests: int,
) -> str:
    forced_device = "cpu"
    command_path = f"{conda_root.rstrip('/')}/bin/{command}" if not conda_root.startswith("$") else command
    python_path = f"{conda_root.rstrip('/')}/bin/python" if not conda_root.startswith("$") else "python"
    mineru_args = [
        command_path,
        "-p",
        pdf_wsl,
        "-o",
        output_wsl,
        "-m",
        method,
        "-b",
        backend,
        "--device",
        forced_device,
    ]
    quoted_mineru_args = " ".join(shlex.quote(arg) for arg in mineru_args)

    conda_root_assignment = conda_root if conda_root.startswith("$") else shlex.quote(conda_root)
    return f"""
set -e
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
CONDA_ROOT={conda_root_assignment}
CONDA_ENV={shlex.quote(conda_env)}
MINERU_COMMAND={shlex.quote(command)}
export MINERU_MODEL_SOURCE={shlex.quote(model_source)}
export PATH="$CONDA_ROOT/bin:$PATH"
export CUDA_VISIBLE_DEVICES=""
export NVIDIA_VISIBLE_DEVICES=""
export MINERU_DEVICE=cpu
export MINERU_DEVICE_MODE=cpu
export MINERU_ROUTER_LOCAL_GPUS=auto
export MINERU_ROUTER_ENABLE_VLM_PRELOAD=0
export MINERU_ROUTER_UPSTREAM_URLS_JSON="[]"
export MINERU_LMDEPLOY_DEVICE=cpu
export MINERU_VIRTUAL_VRAM_SIZE=1
export MINERU_PROCESSING_WINDOW_SIZE={shlex.quote(str(processing_window_size))}
export MINERU_API_MAX_CONCURRENT_REQUESTS={shlex.quote(str(max_concurrent_requests))}
export PADDLE_DEVICE=cpu
export FLAGS_selected_gpus=""
export TORCH_CUDNN_V8_API_DISABLED=1
if [ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
  . "$CONDA_ROOT/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
else
  export PATH="$CONDA_ROOT/envs/$CONDA_ENV/bin:$CONDA_ROOT/bin:$PATH"
fi
export CUDA_VISIBLE_DEVICES=""
export NVIDIA_VISIBLE_DEVICES=""
export MINERU_DEVICE=cpu
export MINERU_DEVICE_MODE=cpu
export MINERU_ROUTER_LOCAL_GPUS=auto
export MINERU_ROUTER_ENABLE_VLM_PRELOAD=0
export MINERU_ROUTER_UPSTREAM_URLS_JSON="[]"
export MINERU_LMDEPLOY_DEVICE=cpu
export MINERU_VIRTUAL_VRAM_SIZE=1
export MINERU_PROCESSING_WINDOW_SIZE={shlex.quote(str(processing_window_size))}
export MINERU_API_MAX_CONCURRENT_REQUESTS={shlex.quote(str(max_concurrent_requests))}
export PADDLE_DEVICE=cpu
export FLAGS_selected_gpus=""
export TORCH_CUDNN_V8_API_DISABLED=1
if [ ! -x {shlex.quote(command_path)} ] && ! command -v "$MINERU_COMMAND" >/dev/null 2>&1; then
  echo "MinerU command not found: $MINERU_COMMAND" >&2
  echo "CONDA_ROOT=$CONDA_ROOT" >&2
  echo "CONDA_ENV=$CONDA_ENV" >&2
  echo "PATH=$PATH" >&2
  exit 127
fi
{shlex.quote(python_path)} -c "from mineru.utils.config_reader import get_device; import sys; device = get_device(); print('MinerU device=' + device); sys.exit(0 if device == 'cpu' else 88)"
{shlex.quote(python_path)} -c "from mineru.cli.router import parse_local_gpus; import os, sys; local_devices = parse_local_gpus(os.environ.get('MINERU_ROUTER_LOCAL_GPUS', 'auto')); print('MinerU router local devices=' + str(local_devices)); sys.exit(0 if local_devices == [None] else 89)"
{shlex.quote(python_path)} -c "from mineru.utils.config_reader import get_processing_window_size, get_max_concurrent_requests; import sys; window = get_processing_window_size(); concurrent = get_max_concurrent_requests(); print('MinerU processing window=' + str(window)); print('MinerU API concurrency=' + str(concurrent)); sys.exit(0 if window <= {processing_window_size} and concurrent <= {max_concurrent_requests} else 90)"
{quoted_mineru_args}
"""


def run_mineru_for_pdf(pdf_path: str | Path, output_dir: str | Path) -> str:
    pdf_file = Path(pdf_path).resolve()
    if not pdf_file.exists():
        raise WSLMinerUError(f"PDF not found: {pdf_file}")

    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    pdf_wsl = windows_path_to_wsl(pdf_file)
    output_wsl = windows_path_to_wsl(output_root)

    conda_root = str(getattr(Config, "MINERU_WSL_CONDA_ROOT", "$HOME/miniconda3") or "$HOME/miniconda3")
    conda_env = str(getattr(Config, "MINERU_WSL_CONDA_ENV", "mineru") or "mineru")
    command = str(getattr(Config, "MINERU_COMMAND", "mineru") or "mineru")
    method = str(getattr(Config, "MINERU_METHOD", "auto") or "auto")
    backend = str(getattr(Config, "MINERU_BACKEND", "pipeline") or "pipeline")
    model_source = str(getattr(Config, "MINERU_MODEL_SOURCE", "modelscope") or "modelscope")
    device = "cpu"
    processing_window_size = int(getattr(Config, "MINERU_PROCESSING_WINDOW_SIZE", 8) or 8)
    max_concurrent_requests = int(getattr(Config, "MINERU_API_MAX_CONCURRENT_REQUESTS", 1) or 1)

    script = _build_mineru_script(
        pdf_wsl=pdf_wsl,
        output_wsl=output_wsl,
        conda_root=conda_root,
        conda_env=conda_env,
        command=command,
        method=method,
        backend=backend,
        model_source=model_source,
        device=device,
        processing_window_size=processing_window_size,
        max_concurrent_requests=max_concurrent_requests,
    )
    return _run_wsl_script(script)
