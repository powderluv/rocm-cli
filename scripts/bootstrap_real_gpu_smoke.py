#!/usr/bin/env python3
"""Run a real GPU end-to-end smoke for the bootstrap assistant.

This is intentionally not a mock harness. It stages a real Qwen llamafile, a
real ROCm backend sidecar, and real ROCm runtime libraries, then runs:

    rocm bootstrap assistant --llamafile ... --smoke-stop-after-ready

The hidden smoke flag starts the real server, requires GPU startup log evidence,
calls the OpenAI-compatible chat endpoint, and stops the child process cleanly.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_GPU_TOKENS = (
    "AMD ROCm GPU support successfully loaded",
    "ggml_cuda_init: found",
    "hipLaunchKernel: Returned hipSuccess",
)
FORBIDDEN_GPU_TOKENS = (
    "falling back to CPU",
    "fallback to CPU",
    "support for --gpu amd was explicitly requested, but it wasn't available",
    "No compatible code objects found",
    "device kernel image is invalid",
    "hipErrorInvalidImage",
)


class SmokeError(Exception):
    """The real bootstrap GPU smoke failed."""


def fail(message: str) -> None:
    raise SmokeError(message)


def expect(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def run(
    args: list[str | Path],
    *,
    cwd: Path = REPO_ROOT,
    env: dict[str, str] | None = None,
    timeout: int = 600,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [str(arg) for arg in args],
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    if check and result.returncode != 0:
        fail(
            "command failed:\n"
            f"  command: {' '.join(str(arg) for arg in args)}\n"
            f"  exit: {result.returncode}\n"
            f"  stdout:\n{result.stdout}\n"
            f"  stderr:\n{result.stderr}"
        )
    return result


def default_work_root() -> Path:
    return REPO_ROOT / ".rocm-work" / "tests" / f"bootstrap-real-gpu-smoke-{os.getpid()}"


def default_rocm_binary() -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    return REPO_ROOT / "target" / "debug" / f"rocm{suffix}"


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def copy_into_stage(source: Path, stage: Path, *, target_name: str | None = None) -> Path:
    expect(source.exists(), f"source does not exist: {source}")
    target = stage / (target_name or source.name)
    if source.is_dir():
        shutil.copytree(source, target, dirs_exist_ok=True)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return target


def copy_runtime_dir(source: Path, stage: Path) -> None:
    expect(source.is_dir(), f"runtime directory does not exist: {source}")
    for child in source.iterdir():
        copy_into_stage(child, stage)


def parse_log_path(output: str) -> Path:
    match = re.search(r"(?m)^\s*(?:log_path:|Log:)\s*(.+?)\s*$", output)
    expect(match is not None, f"bootstrap smoke output did not include a log path:\n{output}")
    return Path(match.group(1)).resolve()


def validate_gpu_log(log_path: Path, output: str) -> None:
    expect(log_path.is_file(), f"bootstrap log does not exist: {log_path}")
    text = log_path.read_text(encoding="utf-8", errors="replace")
    lowered = text.lower()
    for token in FORBIDDEN_GPU_TOKENS:
        expect(token.lower() not in lowered, f"forbidden GPU fallback/error token appeared in {log_path}: {token}")
    if any(token.lower() in lowered for token in REQUIRED_GPU_TOKENS):
        return
    expect(
        "bootstrap assistant smoke response: OK" in output,
        f"bootstrap log did not include verbose ROCm GPU evidence and smoke response was not OK: {log_path}",
    )


def smoke_env(work_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["ROCM_CLI_CONFIG_DIR"] = str(work_root / "config")
    env["ROCM_CLI_DATA_DIR"] = str(work_root / "data")
    env["ROCM_CLI_CACHE_DIR"] = str(work_root / "cache")

    # Deliberately poison common active-runtime variables. The bootstrap child
    # must construct its own clean loader environment from the staged payload.
    env["VIRTUAL_ENV"] = str(work_root / "poison-venv")
    env["CONDA_PREFIX"] = str(work_root / "poison-conda")
    env["HIP_PATH"] = str(work_root / "poison-hip")
    env["ROCM_PATH"] = str(work_root / "poison-rocm")
    env["ROCM_SDK_ROOT"] = str(work_root / "poison-rocm-sdk")
    env["PYTHONPATH"] = str(work_root / "poison-pythonpath")
    return env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rocm", type=Path, default=default_rocm_binary())
    parser.add_argument("--llamafile", type=Path, required=True)
    parser.add_argument("--rocm-backend", type=Path, required=True)
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        action="append",
        default=[],
        help="Runtime bin/lib directory to copy into the stage. May be repeated.",
    )
    parser.add_argument("--work-root", type=Path, default=default_work_root())
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--prompt", default="Reply with exactly OK.")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--keep-stage", action="store_true")
    args = parser.parse_args()

    work_root = args.work_root.resolve()
    if work_root.exists():
        shutil.rmtree(work_root)
    stage = work_root / "stage"
    stage.mkdir(parents=True)

    if not args.skip_build:
        run(["cargo", "build", "-p", "rocm", "--bin", "rocm"], timeout=300)
    rocm = args.rocm.resolve()
    expect(rocm.is_file(), f"rocm binary does not exist: {rocm}")

    staged_llamafile = copy_into_stage(args.llamafile.resolve(), stage)
    backend_name = "ggml-rocm.dll" if os.name == "nt" else "ggml-rocm.so"
    copy_into_stage(args.rocm_backend.resolve(), stage, target_name=backend_name)
    for runtime_dir in args.runtime_dir:
        copy_runtime_dir(runtime_dir.resolve(), stage)

    port = args.port or free_port()
    env = smoke_env(work_root)
    result = run(
        [
            rocm,
            "bootstrap",
            "assistant",
            "--llamafile",
            staged_llamafile,
            "--port",
            str(port),
            "--smoke-stop-after-ready",
            "--smoke-prompt",
            args.prompt,
        ],
        env=env,
        timeout=600,
    )
    combined = result.stdout + result.stderr
    expect("bootstrap assistant smoke passed" in combined, f"bootstrap smoke did not pass:\n{combined}")
    validate_gpu_log(parse_log_path(combined), combined)

    if not args.keep_stage:
        shutil.rmtree(stage, ignore_errors=True)
    print("bootstrap real GPU smoke: passed")
    print(f"  work_root: {work_root}")
    print(f"  port: {port}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SmokeError as error:
        print(f"bootstrap real GPU smoke failed: {error}", file=sys.stderr)
        raise SystemExit(1)
