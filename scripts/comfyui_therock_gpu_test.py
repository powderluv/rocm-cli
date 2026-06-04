#!/usr/bin/env python3
"""End-to-end ComfyUI GPU smoke test for rocm-cli managed TheRock runtimes.

This opt-in test installs or reuses the rocm-cli managed ComfyUI app, starts it
with the active managed TheRock runtime, verifies the local ComfyUI HTTP
endpoint, checks rocm-cli status/log output, then stops the process it started
unless --keep-running is set. It never uses CPU fallback.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 18188


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_test()

    repo_root = Path(__file__).resolve().parents[1]
    rocm = resolve_path(args.rocm, repo_root)
    if not rocm.is_file():
        raise SystemExit(
            f"rocm binary not found: {rocm}\n"
            "Build it with `cargo build -p rocm --bin rocm`, or pass --rocm."
        )

    env = os.environ.copy()
    started_pid: int | None = None
    try:
        if not args.skip_install:
            install_command = build_install_command(args, rocm)
            print_step("Installing ComfyUI with the selected managed ROCm runtime.")
            install_output = run_text(install_command, env=env, timeout=args.timeout)
            print(install_output, end="" if install_output.endswith("\n") else "\n")
            assert_install_output(install_output)

        start_command = build_start_command(args, rocm)
        print_step("Starting ComfyUI in GPU-required mode.")
        start_output = run_text(start_command, env=env, timeout=args.timeout)
        print(start_output, end="" if start_output.endswith("\n") else "\n")
        assert_start_output(start_output, args.host, args.port)
        started_pid = parse_pid(start_output)

        wait_comfyui_endpoint(args.host, args.port, args.timeout)

        status_output = run_text(
            [str(rocm), "comfyui", "status"], env=env, timeout=args.timeout
        )
        print(status_output, end="" if status_output.endswith("\n") else "\n")
        assert_status_output(status_output, args.host, args.port)

        logs_output = run_text(
            [str(rocm), "comfyui", "logs", "--lines", "80"],
            env=env,
            timeout=args.timeout,
        )
        print(logs_output, end="" if logs_output.endswith("\n") else "\n")
        assert_logs_output(logs_output)

        print_step("Success: ComfyUI is reachable through ROCm CLI with AMD GPU checks.")
        print(json.dumps(
            {
                "ok": True,
                "url": f"http://{args.host}:{args.port}",
                "pid": started_pid,
                "installed": not args.skip_install,
            },
            indent=2,
        )
        )
    finally:
        if started_pid and not args.keep_running:
            print_step(f"Stopping ComfyUI process {started_pid}.")
            stop_pid(started_pid)

    return 0


def parse_args() -> argparse.Namespace:
    default_rocm = cargo_binary_path("debug", "rocm")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rocm", default=str(default_rocm))
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--timeout", type=int, default=360)
    parser.add_argument("--runtime-id", help="exact managed TheRock runtime key")
    parser.add_argument("--reinstall", action="store_true")
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="reuse an existing rocm-cli managed ComfyUI install",
    )
    parser.add_argument("--keep-running", action="store_true")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run offline checks for this script and exit",
    )
    return parser.parse_args()


def print_step(message: str) -> None:
    print(f"[comfyui-gpu-test] {message}", flush=True)


def cargo_binary_path(profile: str, name: str) -> Path:
    target_root = Path(os.environ.get("CARGO_TARGET_DIR", "target")).expanduser()
    return target_root / profile / exe_name(name)


def exe_name(name: str) -> str:
    return f"{name}.exe" if platform.system() == "Windows" else name


def resolve_path(value: str, repo_root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def build_install_command(args: argparse.Namespace, rocm: Path) -> list[str]:
    command = [str(rocm), "comfyui", "install"]
    if args.runtime_id:
        command.extend(["--runtime-id", args.runtime_id])
    if args.reinstall:
        command.append("--reinstall")
    return command


def build_start_command(args: argparse.Namespace, rocm: Path) -> list[str]:
    return [
        str(rocm),
        "comfyui",
        "start",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--no-open-browser",
    ]


def run_text(
    command: list[str],
    *,
    env: dict[str, str],
    timeout: int,
    check: bool = True,
) -> str:
    completed = subprocess.run(
        command,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    output = completed.stdout
    if completed.stderr:
        output += completed.stderr
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {format_command(command)}\n{output}"
        )
    return output


def format_command(command: list[str]) -> str:
    return " ".join(command)


def parse_pid(output: str) -> int | None:
    match = re.search(r"(?m)^\s*pid:\s*(\d+)\s*$", output)
    if not match:
        return None
    pid = int(match.group(1))
    return pid if pid > 0 else None


def wait_comfyui_endpoint(host: str, port: int, timeout: int) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_error = "endpoint was not checked"
    while time.monotonic() < deadline:
        try:
            conn = http.client.HTTPConnection(host, port, timeout=5)
            conn.request("GET", "/system_stats")
            response = conn.getresponse()
            body = response.read()
            if response.status == 200:
                text = body.decode("utf-8", errors="replace")
                if "system" in text or "devices" in text:
                    return {"status": response.status, "body": text[:400]}
                last_error = "GET /system_stats returned unexpected body"
            else:
                last_error = f"GET /system_stats returned HTTP {response.status}"
        except OSError as error:
            last_error = str(error)
        finally:
            try:
                conn.close()  # type: ignore[name-defined]
            except Exception:
                pass
        time.sleep(2)
    raise RuntimeError(
        f"ComfyUI endpoint did not become reachable before timeout: {last_error}"
    )


def assert_install_output(output: str) -> None:
    require_contains(output, "AMD GPU check: ready", "install output")
    require_contains(output, "next step: rocm comfyui start", "install output")
    reject_cpu_fallback(output, "install output")


def assert_start_output(output: str, host: str, port: int) -> None:
    require_contains(output, "status: starting", "start output")
    require_contains(output, "AMD GPU check: ready", "start output")
    require_contains(output, f"url: http://{host}:{port}", "start output")
    require_contains(output, "browser: not opened", "start output")
    if parse_pid(output) is None:
        raise RuntimeError(f"start output did not include a process id:\n{output}")
    reject_cpu_fallback(output, "start output")


def assert_status_output(output: str, host: str, port: int) -> None:
    require_contains(output, "status: running", "status output")
    require_contains(output, f"url: http://{host}:{port}", "status output")
    if "status: stopped" in output:
        raise RuntimeError(f"ComfyUI status reports stopped:\n{output}")
    reject_cpu_fallback(output, "status output")


def assert_logs_output(output: str) -> None:
    require_contains(output, "ComfyUI logs", "logs output")
    if "Run log" not in output and "Install log" not in output:
        raise RuntimeError(f"ComfyUI logs did not include saved app logs:\n{output}")
    reject_cpu_fallback(output, "logs output")


def require_contains(output: str, needle: str, label: str) -> None:
    if needle not in output:
        raise RuntimeError(f"{label} did not contain `{needle}`:\n{output}")


def reject_cpu_fallback(output: str, label: str) -> None:
    forbidden = [
        "CPU fallback",
        "cpu fallback",
        "falling back to CPU",
        "Running on CPU",
        "cpu_only",
    ]
    for needle in forbidden:
        if needle in output:
            raise RuntimeError(f"{label} contained forbidden `{needle}`:\n{output}")


def stop_pid(pid: int) -> None:
    if platform.system() == "Windows":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        subprocess.run(["kill", str(pid)], check=False)


def run_self_test() -> int:
    fake_rocm = Path("target/debug") / exe_name("rocm")
    args = argparse.Namespace(
        host="127.0.0.1",
        port=18188,
        runtime_id="release-pip-gfx120x-all-7-14-0a20260601",
        reinstall=True,
    )
    install = build_install_command(args, fake_rocm)
    assert install[:3] == [str(fake_rocm), "comfyui", "install"]
    assert "--runtime-id" in install
    assert "--reinstall" in install
    assert "cpu" not in " ".join(install).lower()

    start = build_start_command(args, fake_rocm)
    assert start[:3] == [str(fake_rocm), "comfyui", "start"]
    assert "--no-open-browser" in start
    assert "--port" in start and "18188" in start

    assert parse_pid("ComfyUI\n  pid: 12345\n") == 12345
    assert parse_pid("ComfyUI\n  pid: 0\n") is None

    assert_install_output(
        "ComfyUI\n  installed: yes\n  AMD GPU check: ready (1 device)\n"
        "  next step: rocm comfyui start\n"
    )
    assert_start_output(
        "ComfyUI\n  status: starting\n  AMD GPU check: ready (1 device)\n"
        "  url: http://127.0.0.1:18188\n  browser: not opened (--no-open-browser)\n"
        "  pid: 12345\n",
        "127.0.0.1",
        18188,
    )
    assert_status_output(
        "ComfyUI\n\nRunning\n  status: running\n  url: http://127.0.0.1:18188\n",
        "127.0.0.1",
        18188,
    )
    assert_logs_output("ComfyUI logs\n\nRun log\n  latest output:\n    started\n")
    try:
        assert_status_output(
            "ComfyUI\n\nRunning\n  status: stopped\n  url: http://127.0.0.1:18188\n",
            "127.0.0.1",
            18188,
        )
    except RuntimeError as error:
        assert "status: running" in str(error) or "stopped" in str(error)
    else:
        raise AssertionError("stopped status was incorrectly accepted")
    try:
        assert_start_output(
            "ComfyUI\n  status: starting\n  AMD GPU check: ready (1 device)\n"
            "  url: http://127.0.0.1:18188\n  browser: not opened (--no-open-browser)\n"
            "  pid: 12345\n  CPU fallback enabled\n",
            "127.0.0.1",
            18188,
        )
    except RuntimeError as error:
        assert "CPU fallback" in str(error)
    else:
        raise AssertionError("CPU fallback output was incorrectly accepted")
    print("ComfyUI GPU script self-test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
