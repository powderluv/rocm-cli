#!/usr/bin/env python3
"""End-to-end bootstrap assistant acceptance checks.

This harness compiles a tiny fake llamafile executable and runs the real
`rocm bootstrap assistant --llamafile ...` path against it. The fake process
acts like the embedded GPU assistant server just enough to prove the CLI launch
path:

- bootstrap subcommand routing reaches the hidden bootstrap command;
- the child process gets GPU-required server args;
- active venv/Python/ROCm/HIP environment leaks are stripped before launch;
- a listening server with ROCm/HIP log evidence is accepted;
- CPU fallback log evidence is rejected and the child process is stopped.

It intentionally does not download Qwen, llamafile, TheRock wheels, ROCm
packages, or drivers. Pair it with `ape_bootstrap_package.py self-test` and
`build_ape_bootstrap.py self-test` for the packaging contract.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
FAKE_LLAMA_RUST = r"""
use std::env;
use std::io::{self, Write};
use std::net::TcpListener;
use std::process;
use std::thread;
use std::time::{Duration, Instant};

fn arg_value(args: &[String], name: &str, default: &str) -> String {
    for index in 0..args.len() {
        if args[index] == name && index + 1 < args.len() {
            return args[index + 1].clone();
        }
        let prefix = format!("{name}=");
        if args[index].starts_with(&prefix) {
            return args[index][prefix.len()..].to_owned();
        }
    }
    default.to_owned()
}

fn emit(line: &str) {
    let mut out = io::stdout().lock();
    writeln!(out, "{line}").expect("fake llamafile stdout should be writable");
    out.flush().expect("fake llamafile stdout should flush");
}

fn main() {
    let args = env::args().skip(1).collect::<Vec<_>>();
    let host = arg_value(&args, "--host", "127.0.0.1");
    let port = arg_value(&args, "--port", "11435")
        .parse::<u16>()
        .expect("fake llamafile needs numeric --port");
    let gpu_arg = arg_value(&args, "--gpu", "<missing>");
    let ngl_arg = arg_value(&args, "-ngl", "<missing>");
    let log_level_arg = arg_value(&args, "-lv", "<missing>");

    emit(&format!("FAKE_BOOTSTRAP_PID={}", process::id()));
    emit(&format!("FAKE_BOOTSTRAP_ARGS={}", args.join(" ")));
    emit(&format!("FAKE_BOOTSTRAP_GPU_ARG={gpu_arg}"));
    emit(&format!("FAKE_BOOTSTRAP_NGL_ARG={ngl_arg}"));
    emit(&format!("FAKE_BOOTSTRAP_LOG_LEVEL_ARG={log_level_arg}"));
    emit(&format!("AMD_LOG_LEVEL={}", env::var("AMD_LOG_LEVEL").unwrap_or_default()));

    for name in [
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "PYTHONHOME",
        "PYTHONPATH",
        "ROCM_PATH",
        "ROCM_SDK_ROOT",
        "HIP_PATH",
        "DYLD_LIBRARY_PATH",
    ] {
        if env::var_os(name).is_some() {
            emit(&format!("ENV_LEAK {name}"));
        } else {
            emit(&format!("ENV_CLEAN {name}"));
        }
    }
    match env::var("LD_LIBRARY_PATH") {
        Ok(value) => emit(&format!("ENV_SET LD_LIBRARY_PATH={value}")),
        Err(_) => emit("ENV_CLEAN LD_LIBRARY_PATH"),
    }
    emit(&format!("FAKE_BOOTSTRAP_PATH={}", env::var("PATH").unwrap_or_default()));

    if env::var("ROCM_CLI_FAKE_BOOTSTRAP_EXIT_EARLY").ok().as_deref() == Some("1") {
        emit("HIP runtime failed to load before listening");
        process::exit(7);
    }

    if gpu_arg != "amd" || ngl_arg == "0" || ngl_arg == "<missing>" {
        emit("GPU support could not be loaded; falling back to CPU");
    } else if env::var("ROCM_CLI_FAKE_BOOTSTRAP_CPU_FALLBACK").ok().as_deref() == Some("1") {
        emit("GPU support could not be loaded; falling back to CPU");
    } else {
        emit("AMD_LOG_LEVEL=0");
        emit("HIP runtime initialized");
        emit("rocBLAS loaded");
        emit("llama server listening");
    }

    let listener = TcpListener::bind(format!("{host}:{port}"))
        .expect("fake llamafile should bind requested loopback port");
    listener
        .set_nonblocking(true)
        .expect("fake llamafile should set nonblocking listener");
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline {
        match listener.accept() {
            Ok((_stream, _addr)) => break,
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(25));
            }
            Err(error) => panic!("fake llamafile accept failed: {error}"),
        }
    }

    let hold_secs = env::var("ROCM_CLI_FAKE_BOOTSTRAP_HOLD_SECS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(1);
    thread::sleep(Duration::from_secs(hold_secs));
}
"""


def default_work_root() -> Path:
    return REPO_ROOT / ".rocm-work" / "tests" / f"bootstrap-workflow-{os.getpid()}"


class AcceptanceError(Exception):
    """The bootstrap workflow acceptance test failed."""


def fail(message: str) -> None:
    raise AcceptanceError(message)


def expect(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def run(
    args: list[str | Path],
    *,
    cwd: Path = REPO_ROOT,
    env: dict[str, str] | None = None,
    timeout: int = 120,
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


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def rocm_binary_from_args(value: str | None) -> Path:
    if value:
        return Path(value).resolve()
    suffix = ".exe" if os.name == "nt" else ""
    return REPO_ROOT / "target" / "debug" / f"rocm{suffix}"


def ensure_rocm_binary(path: Path, *, skip_build: bool) -> Path:
    if not skip_build:
        run(["cargo", "build", "-p", "rocm", "--bin", "rocm"], timeout=300)
    expect(path.is_file(), f"rocm binary does not exist: {path}")
    return path


def compile_fake_llamafile(work_root: Path) -> Path:
    rustc = shutil.which("rustc")
    expect(rustc is not None, "rustc is required to compile the fake embedded llamafile")
    source = work_root / "fake_bootstrap_llamafile.rs"
    binary = work_root / ("fake_bootstrap_llamafile.exe" if os.name == "nt" else "fake_bootstrap_llamafile")
    work_root.mkdir(parents=True, exist_ok=True)
    source.write_text(FAKE_LLAMA_RUST, encoding="utf-8")
    run([rustc, "--edition=2021", source, "-o", binary], cwd=work_root, timeout=120)
    expect(binary.is_file(), f"fake llamafile did not build: {binary}")
    return binary


def polluted_env(work_root: Path, *, cpu_fallback: bool = False) -> dict[str, str]:
    env = os.environ.copy()
    venv_root = work_root / "poison-venv"
    venv_bin = venv_root / ("Scripts" if os.name == "nt" else "bin")
    path_only_venv_bin = work_root / "path-only-venv" / ("Scripts" if os.name == "nt" else "bin")
    therock_bin = work_root / "therock_venvs" / "bin"
    rocm_bin = work_root / "ROCm" / "bin"
    venv_bin.mkdir(parents=True, exist_ok=True)
    path_only_venv_bin.mkdir(parents=True, exist_ok=True)
    therock_bin.mkdir(parents=True, exist_ok=True)
    rocm_bin.mkdir(parents=True, exist_ok=True)

    env["VIRTUAL_ENV"] = str(venv_root)
    env["CONDA_PREFIX"] = str(work_root / "poison-conda")
    env["PYTHONPATH"] = str(work_root / "poison-pythonpath")
    env["ROCM_PATH"] = str(work_root / "poison-rocm")
    env["ROCM_SDK_ROOT"] = str(work_root / "poison-rocm-sdk")
    env["HIP_PATH"] = str(work_root / "poison-hip")
    env["LD_LIBRARY_PATH"] = str(work_root / "poison-ld")
    env["DYLD_LIBRARY_PATH"] = str(work_root / "poison-dyld")
    env["PATH"] = os.pathsep.join(
        [str(venv_bin), str(path_only_venv_bin), str(therock_bin), str(rocm_bin), env.get("PATH", "")]
    )
    env["ROCM_CLI_CONFIG_DIR"] = str(work_root / "config")
    env["ROCM_CLI_DATA_DIR"] = str(work_root / "data")
    env["ROCM_CLI_CACHE_DIR"] = str(work_root / "cache")
    env["ROCM_CLI_FAKE_BOOTSTRAP_HOLD_SECS"] = "20" if cpu_fallback else "1"
    if cpu_fallback:
        env["ROCM_CLI_FAKE_BOOTSTRAP_CPU_FALLBACK"] = "1"
    else:
        env.pop("ROCM_CLI_FAKE_BOOTSTRAP_CPU_FALLBACK", None)
    return env


def parse_log_path(output: str) -> Path:
    match = re.search(r"(?m)^\s*(?:log_path:|Log:)\s*(.+?)\s*$", output)
    expect(match is not None, f"bootstrap output did not include a log path:\n{output}")
    return Path(match.group(1)).resolve()


def parse_fake_pid(log_text: str) -> int:
    match = re.search(r"FAKE_BOOTSTRAP_PID=(\d+)", log_text)
    expect(match is not None, f"fake bootstrap pid missing from log:\n{log_text}")
    return int(match.group(1))


def process_is_running(pid: int) -> bool:
    if os.name == "nt":
        probe = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                f"if (Get-Process -Id {pid} -ErrorAction SilentlyContinue) {{ 'yes' }}",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return "yes" in probe.stdout.lower()
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def run_success_flow(rocm: Path, fake_llamafile: Path, work_root: Path) -> None:
    env = polluted_env(work_root / "success")
    port = free_port()
    result = run(
        [
            rocm,
            "bootstrap",
            "assistant",
            "--llamafile",
            fake_llamafile,
            "--port",
            str(port),
        ],
        env=env,
        timeout=60,
    )
    combined = result.stdout + result.stderr
    expect(
        "ROCm setup assistant is ready." in combined or "bootstrap assistant ready" in combined,
        f"bootstrap did not report ready:\n{combined}",
    )
    log_path = parse_log_path(combined)
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    expect("HIP runtime initialized" in log_text, "fake GPU HIP evidence missing from bootstrap log")
    expect("rocBLAS loaded" in log_text, "fake rocBLAS evidence missing from bootstrap log")
    expect("FAKE_BOOTSTRAP_GPU_ARG=amd" in log_text, "embedded server did not get --gpu amd")
    expect("FAKE_BOOTSTRAP_NGL_ARG=999" in log_text, "embedded server did not get full GPU offload")
    expect("FAKE_BOOTSTRAP_LOG_LEVEL_ARG=0" in log_text, "embedded server did not get quiet -lv 0")
    expect("AMD_LOG_LEVEL=0" in log_text, "bootstrap did not set AMD_LOG_LEVEL=0")
    expect("ENV_LEAK" not in log_text, f"bootstrap child inherited forbidden env:\n{log_text}")
    expect("poison-venv" not in log_text, "bootstrap child PATH still contains active venv path")
    expect("path-only-venv" not in log_text, "bootstrap child PATH still contains PATH-only venv path")
    expect("therock_venvs" not in log_text, "bootstrap child PATH still contains TheRock path")
    expect("ROCm" not in log_text and "rocm/bin" not in log_text.lower(), "bootstrap child PATH still contains ROCm path")

    services = list((work_root / "success" / "data" / "services").glob("*.json"))
    expect(services, "bootstrap did not write a managed service record")
    service_text = services[0].read_text(encoding="utf-8")
    expect('"status": "stopped"' in service_text, "successful non-interactive bootstrap should end stopped")


def run_cpu_fallback_rejection(rocm: Path, fake_llamafile: Path, work_root: Path) -> None:
    env = polluted_env(work_root / "cpu-fallback", cpu_fallback=True)
    port = free_port()
    result = run(
        [
            rocm,
            "bootstrap",
            "assistant",
            "--llamafile",
            fake_llamafile,
            "--port",
            str(port),
        ],
        env=env,
        timeout=60,
        check=False,
    )
    combined = result.stdout + result.stderr
    expect(result.returncode != 0, "CPU fallback bootstrap unexpectedly succeeded")
    expect("CPU fallback" in combined, f"CPU fallback error was not clear:\n{combined}")
    if os.name == "nt":
        expect(
            "https://www.amd.com/en/support/download/drivers.html" in combined,
            f"Windows CPU fallback error did not include AMD driver URL:\n{combined}",
        )
    log_path = parse_log_path(combined)
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    pid = parse_fake_pid(log_text)
    deadline = time.time() + 5
    while time.time() < deadline and process_is_running(pid):
        time.sleep(0.1)
    expect(not process_is_running(pid), f"CPU fallback fake server was left running: pid={pid}")
    services = list((work_root / "cpu-fallback" / "data" / "services").glob("*.json"))
    expect(services, "failed bootstrap did not write a managed service record")
    service_text = services[0].read_text(encoding="utf-8")
    expect('"status": "failed"' in service_text, "CPU fallback bootstrap should mark service failed")


def run_early_exit_driver_guidance(rocm: Path, fake_llamafile: Path, work_root: Path) -> None:
    env = polluted_env(work_root / "early-exit")
    env["ROCM_CLI_FAKE_BOOTSTRAP_EXIT_EARLY"] = "1"
    port = free_port()
    result = run(
        [
            rocm,
            "bootstrap",
            "assistant",
            "--llamafile",
            fake_llamafile,
            "--port",
            str(port),
        ],
        env=env,
        timeout=60,
        check=False,
    )
    combined = result.stdout + result.stderr
    expect(result.returncode != 0, "early-exit bootstrap unexpectedly succeeded")
    expect("exited before it became ready" in combined, f"early-exit error was not clear:\n{combined}")
    if os.name == "nt":
        expect(
            "https://www.amd.com/en/support/download/drivers.html" in combined,
            f"Windows early-exit error did not include AMD driver URL:\n{combined}",
        )
    log_path = parse_log_path(combined)
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    expect("HIP runtime failed to load before listening" in log_text, "early-exit runtime failure log missing")


def run_validate_only(rocm: Path, work_root: Path) -> None:
    env = polluted_env(work_root / "validate-only")
    result = run(
        [rocm, "bootstrap", "assistant", "--validate-only", "--port", str(free_port())],
        env=env,
        timeout=60,
    )
    combined = result.stdout + result.stderr
    expect(
        "bootstrap assistant validation" in combined,
        f"bootstrap validate-only did not reach bootstrap command:\n{combined}",
    )
    expect("request plan" not in combined, "bootstrap validate-only fell into natural-language planner")
    expect("device_policy: gpu_required" in combined, "bootstrap validate-only did not report gpu_required")
    expect(
        "cpu_fallback_policy: rejected_by_rocm_cli" in combined,
        "bootstrap validate-only did not report CPU fallback rejection",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rocm", help="Path to a rocm binary. Defaults to target/debug/rocm(.exe).")
    parser.add_argument("--work-root", type=Path, default=default_work_root())
    parser.add_argument("--skip-build", action="store_true", help="Do not run cargo build before acceptance.")
    args = parser.parse_args()

    work_root = args.work_root.resolve()
    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True)
    rocm = ensure_rocm_binary(rocm_binary_from_args(args.rocm), skip_build=args.skip_build)
    fake_llamafile = compile_fake_llamafile(work_root)

    run_validate_only(rocm, work_root)
    run_success_flow(rocm, fake_llamafile, work_root)
    run_cpu_fallback_rejection(rocm, fake_llamafile, work_root)
    run_early_exit_driver_guidance(rocm, fake_llamafile, work_root)

    print("bootstrap workflow acceptance: validate-only passed")
    print("bootstrap workflow acceptance: GPU-required fake embedded assistant passed")
    print("bootstrap workflow acceptance: CPU fallback rejected and child stopped")
    print("bootstrap workflow acceptance: early runtime exit shows driver guidance")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AcceptanceError as error:
        print(f"bootstrap workflow acceptance failed: {error}", file=sys.stderr)
        raise SystemExit(1)
