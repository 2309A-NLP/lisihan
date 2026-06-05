# -*- coding: utf-8 -*-
"""Run MinerU installed in WSL from a Windows Python process."""

from __future__ import annotations

import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


DEFAULT_CONDA_PREFIX = "/home/li/miniconda3"


class MinerUError(RuntimeError):
    """Raised when MinerU cannot be invoked or returns a non-zero exit code."""


@dataclass(frozen=True)
class MinerUResult:
    input_pdf: Path
    output_dir: Path
    command: str
    returncode: int
    stdout: str
    stderr: str
    markdown_path: Path | None
    content_list_path: Path | None

    def read_markdown(self, encoding: str = "utf-8") -> str:
        if self.markdown_path is None:
            raise FileNotFoundError("MinerU did not produce a markdown file.")
        return self.markdown_path.read_text(encoding=encoding)


def parse_pdf_with_mineru_wsl(
    pdf_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str] = "mineru_output",
    *,
    distro: str | None = None,
    conda_prefix: str = DEFAULT_CONDA_PREFIX,
    conda_env: str = "base",
    method: str = "auto",
    backend: str | None = None,
    lang: str | None = "ch",
    start_page: int | None = None,
    end_page: int | None = None,
    device: str | None = None,
    api_url: str | None = None,
    extra_args: Sequence[str] = (),
    command_template: str | None = None,
    extra_env: Mapping[str, str] | None = None,
    timeout: float | None = None,
) -> MinerUResult:
    """Parse a PDF by calling the MinerU CLI inside WSL.

    This function is meant for PyCharm on Windows: give it a normal Windows
    path, and it will convert paths to WSL paths before invoking MinerU.
    """

    input_pdf = Path(pdf_path).expanduser().resolve()
    if not input_pdf.exists():
        raise FileNotFoundError(f"PDF not found: {input_pdf}")
    if input_pdf.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {input_pdf}")

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    wsl_input = _windows_to_wsl_path(input_pdf, distro=distro)
    wsl_output = _windows_to_wsl_path(output_path, distro=distro)
    candidates = _build_command_candidates(
        wsl_input=wsl_input,
        wsl_output=wsl_output,
        method=method,
        backend=backend,
        lang=lang,
        start_page=start_page,
        end_page=end_page,
        device=device,
        api_url=api_url,
        extra_args=extra_args,
        command_template=command_template or os.getenv("MINERU_COMMAND_TEMPLATE"),
    )

    started_at = time.time()
    failures: list[str] = []
    for command in candidates:
        bash_command = _with_conda(
            command,
            conda_prefix=conda_prefix,
            conda_env=conda_env,
            extra_env=extra_env,
        )
        completed = _run_wsl_bash(
            bash_command,
            distro=distro,
            timeout=timeout,
        )
        if completed.returncode == 0:
            markdown_path = _find_artifact(output_path, input_pdf.stem, "*.md", started_at)
            content_list_path = _find_artifact(
                output_path,
                input_pdf.stem,
                "*content_list*.json",
                started_at,
            )
            return MinerUResult(
                input_pdf=input_pdf,
                output_dir=output_path,
                command=command,
                returncode=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
                markdown_path=markdown_path,
                content_list_path=content_list_path,
            )

        failures.append(
            "\n".join(
                [
                    f"$ {command}",
                    f"returncode={completed.returncode}",
                    f"stdout:\n{completed.stdout.strip()}",
                    f"stderr:\n{completed.stderr.strip()}",
                ]
            )
        )

    raise MinerUError("MinerU failed in WSL.\n\n" + "\n\n".join(failures))


def _build_command_candidates(
    *,
    wsl_input: str,
    wsl_output: str,
    method: str,
    backend: str | None,
    lang: str | None,
    start_page: int | None,
    end_page: int | None,
    device: str | None,
    api_url: str | None,
    extra_args: Sequence[str],
    command_template: str | None,
) -> list[str]:
    quoted_input = shlex.quote(wsl_input)
    quoted_output = shlex.quote(wsl_output)
    options = _render_common_options(
        method=method,
        backend=backend,
        lang=lang,
        start_page=start_page,
        end_page=end_page,
        device=device,
        api_url=api_url,
        extra_args=extra_args,
    )

    if command_template:
        return [
            command_template.format(
                input=quoted_input,
                output=quoted_output,
                method=shlex.quote(method),
                options=options,
            )
        ]

    old_cli_options = _render_common_options(
        method=method,
        backend=None,
        lang=lang,
        start_page=start_page,
        end_page=end_page,
        device=None,
        api_url=None,
        extra_args=extra_args,
    )
    return [
        f"mineru -p {quoted_input} -o {quoted_output} {options}".strip(),
        f"magic-pdf -p {quoted_input} -o {quoted_output} {old_cli_options}".strip(),
    ]


def _render_common_options(
    *,
    method: str,
    backend: str | None,
    lang: str | None,
    start_page: int | None,
    end_page: int | None,
    device: str | None,
    api_url: str | None,
    extra_args: Sequence[str],
) -> str:
    args: list[str] = []
    if method:
        args.extend(["-m", method])
    if backend:
        args.extend(["-b", backend])
    if lang:
        args.extend(["-l", lang])
    if start_page is not None:
        args.extend(["-s", str(start_page)])
    if end_page is not None:
        args.extend(["-e", str(end_page)])
    if device:
        args.extend(["--device", device])
    if api_url:
        args.extend(["--api-url", api_url])
    args.extend(extra_args)
    return " ".join(shlex.quote(str(arg)) for arg in args)


def _with_conda(
    command: str,
    *,
    conda_prefix: str,
    conda_env: str,
    extra_env: Mapping[str, str] | None,
) -> str:
    exports = [
        f"export PATH={shlex.quote(conda_prefix + '/bin')}:\"$PATH\"",
        "export PYTHONIOENCODING=utf-8",
    ]
    for key, value in (extra_env or {}).items():
        exports.append(f"export {key}={shlex.quote(value)}")

    conda_sh = f"{conda_prefix.rstrip('/')}/etc/profile.d/conda.sh"
    activate = (
        f"if [ -f {shlex.quote(conda_sh)} ]; then "
        f". {shlex.quote(conda_sh)} && conda activate {shlex.quote(conda_env)}; "
        "fi"
    )
    return "set -e; " + "; ".join(exports + [activate, command])


def _windows_to_wsl_path(path: Path, *, distro: str | None) -> str:
    if os.name != "nt":
        return str(path)

    drive = path.drive.rstrip(":").lower()
    if drive and path.is_absolute():
        parts = [part for part in path.parts[1:] if part not in {"\\", "/"}]
        return "/mnt/" + drive + "/" + "/".join(parts)

    args = ["wsl.exe"]
    if distro:
        args.extend(["-d", distro])
    args.extend(["wslpath", "-a", str(path)])
    completed = _run(args)
    if completed.returncode != 0:
        raise MinerUError(
            "Failed to convert Windows path to WSL path with wslpath.\n"
            f"path={path}\nstdout={completed.stdout}\nstderr={completed.stderr}"
        )
    return completed.stdout.strip()


def _run_wsl_bash(
    command: str,
    *,
    distro: str | None,
    timeout: float | None,
) -> subprocess.CompletedProcess[str]:
    if os.name == "nt":
        args = ["wsl.exe"]
        if distro:
            args.extend(["-d", distro])
        args.extend(["bash", "-lc", command])
    else:
        args = ["bash", "-lc", command]
    completed = _run(args, timeout=timeout)
    if completed.returncode != 0 and _looks_like_wsl_unavailable(completed.stdout, completed.stderr):
        raise MinerUError(
            "WSL is unavailable or has no installed Linux distribution. "
            "Please install/start WSL and MinerU, then rerun parsing.\n"
            f"stdout:\n{completed.stdout.strip()}\n"
            f"stderr:\n{completed.stderr.strip()}"
        )
    return completed


def _run(
    args: Sequence[str],
    *,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        list(args),
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    return subprocess.CompletedProcess(
        completed.args,
        completed.returncode,
        stdout=_decode_process_output(completed.stdout),
        stderr=_decode_process_output(completed.stderr),
    )


def _decode_process_output(data: bytes | str | None) -> str:
    if data is None:
        return ""
    if isinstance(data, str):
        return data
    for encoding in ("utf-8", "gbk", "utf-16le"):
        try:
            text = data.decode(encoding)
        except UnicodeDecodeError:
            continue
        if "\x00" not in text and "�" not in text:
            return text
    return data.decode("utf-8", errors="replace")


def _looks_like_wsl_unavailable(stdout: str, stderr: str) -> bool:
    text = f"{stdout}\n{stderr}".lower()
    markers = [
        "wsl.exe --install",
        "wsl.exe --list --online",
        "no installed distributions",
        "没有已安装的分发",
        "适用于 linux 的 windows 子系统没有已安装的分发",
    ]
    return any(marker in text for marker in markers)


def _find_artifact(
    output_dir: Path,
    stem: str,
    pattern: str,
    started_at: float,
) -> Path | None:
    candidates = list(output_dir.rglob(pattern))
    if not candidates:
        return None

    def score(path: Path) -> tuple[int, int, float]:
        name_score = 1 if stem in path.stem else 0
        fresh_score = 1 if path.stat().st_mtime >= started_at - 5 else 0
        return (fresh_score, name_score, path.stat().st_mtime)

    return max(candidates, key=score)


def parse_many_with_mineru_wsl(
    pdf_paths: Iterable[str | os.PathLike[str]],
    output_dir: str | os.PathLike[str] = "mineru_output",
    **kwargs,
) -> list[MinerUResult]:
    return [
        parse_pdf_with_mineru_wsl(pdf_path, output_dir=output_dir, **kwargs)
        for pdf_path in pdf_paths
    ]
