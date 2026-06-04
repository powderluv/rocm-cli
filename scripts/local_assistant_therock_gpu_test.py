#!/usr/bin/env python3
"""Live local-assistant acceptance for rocm-cli managed TheRock GPU services.

This opt-in test starts a managed PyTorch service with GPU-required policy,
runs `rocm chat --tools --provider local`, verifies the request reached a
ready managed local service, then stops the service it started. It never uses
CPU fallback.
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
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse
from typing import Any


DEFAULT_MODEL = "qwen"
DEFAULT_PROMPT = "Use ROCm tools to check this computer and summarize the setup."


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
    localize_huggingface_cache(env, repo_root)
    data_dir = rocm_cli_data_dir(env)
    service_id = args.service_id
    started_service = False

    try:
        if args.skip_serve:
            if not service_id:
                service_id = find_ready_service_id(data_dir, args.chat_model or args.model)
            print_step(f"Using existing managed service {service_id}.")
        else:
            serve_cmd = build_serve_command(args, rocm)
            print_step("Starting managed PyTorch local assistant service in GPU-required mode.")
            serve_output = run_text(serve_cmd, env=env, timeout=args.timeout)
            print(serve_output, end="" if serve_output.endswith("\n") else "\n")
            service_id = parse_service_id(serve_output)
            started_service = True

        manifest = wait_ready_manifest(data_dir, service_id, args.timeout)
        assert_service_manifest(manifest, expected_engine="pytorch")
        wait_local_endpoint(manifest, args.timeout)
        chat_model = args.chat_model or manifest.get("canonical_model_id") or args.model
        chat_cmd = build_chat_command(args, rocm, str(chat_model))
        print_step("Running local assistant chat with ROCm tools enabled.")
        chat_output = run_text(chat_cmd, env=env, timeout=args.timeout)
        print(chat_output, end="" if chat_output.endswith("\n") else "\n")
        assert_chat_output(chat_output, require_tool_call=args.require_tool_call)

        summary = {
            "ok": True,
            "message": "Success: local assistant reached a managed ROCm GPU service.",
            "service_id": service_id,
            "model": chat_model,
            "endpoint": manifest.get("endpoint_url"),
            "runtime_id": manifest.get("runtime_id"),
            "env_id": manifest.get("env_id"),
            "device_policy": manifest.get("device_policy"),
            "manifest_path": str(service_manifest_path(data_dir, service_id)),
        }
        print_step("Success: local assistant used the managed local service.")
        print(json.dumps(summary, indent=2))
    finally:
        if started_service and not args.keep_running and service_id:
            print_step(f"Stopping managed service {service_id}.")
            stop_cmd = [str(rocm), "services", "stop", service_id, "--yes"]
            stop_output = run_text(stop_cmd, env=env, timeout=args.timeout, check=False)
            print(stop_output, end="" if stop_output.endswith("\n") else "\n")

    return 0


def parse_args() -> argparse.Namespace:
    default_rocm = cargo_binary_path("debug", "rocm")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rocm", default=str(default_rocm))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--chat-model",
        help="model filter passed to `rocm chat`; defaults to the launched service canonical id",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=11442)
    parser.add_argument("--timeout", type=int, default=360)
    parser.add_argument("--runtime-id", help="exact managed TheRock runtime key")
    parser.add_argument("--env-id", help="managed PyTorch engine env id")
    parser.add_argument("--service-id", help="service id to use with --skip-serve")
    parser.add_argument(
        "--skip-serve",
        action="store_true",
        help="reuse an existing ready managed service instead of launching one",
    )
    parser.add_argument(
        "--require-tool-call",
        action="store_true",
        help="fail unless the model requests and rocm-cli renders a ROCm tool result",
    )
    parser.add_argument("--keep-running", action="store_true")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run offline checks for this script and exit",
    )
    return parser.parse_args()


def print_step(message: str) -> None:
    print(f"[local-assistant-gpu-test] {message}", flush=True)


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


def build_serve_command(args: argparse.Namespace, rocm: Path) -> list[str]:
    command = [
        str(rocm),
        "serve",
        args.model,
        "--engine",
        "pytorch",
        "--device",
        "gpu_required",
        "--managed",
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    if args.runtime_id:
        command.extend(["--runtime-id", args.runtime_id])
    if args.env_id:
        command.extend(["--env-id", args.env_id])
    return command


def build_chat_command(args: argparse.Namespace, rocm: Path, model: str) -> list[str]:
    return [
        str(rocm),
        "chat",
        "--tools",
        "--provider",
        "local",
        "--model",
        model,
        "--prompt",
        args.prompt,
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


def parse_service_id(output: str) -> str:
    match = re.search(r"(?m)^\s*service_id:\s*([A-Za-z0-9_.-]+)\s*$", output)
    if not match:
        raise RuntimeError(f"managed serve output did not include a service_id:\n{output}")
    return match.group(1)


def rocm_cli_data_dir(env: dict[str, str]) -> Path:
    override = env.get("ROCM_CLI_DATA_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".rocm"


def service_manifest_path(data_dir: Path, service_id: str) -> Path:
    return data_dir / "services" / f"{service_id}.json"


def wait_ready_manifest(data_dir: Path, service_id: str, timeout: int) -> dict[str, Any]:
    path = service_manifest_path(data_dir, service_id)
    deadline = time.monotonic() + timeout
    last_status = "<missing>"
    while time.monotonic() < deadline:
        if path.is_file():
            manifest = json.loads(path.read_text(encoding="utf-8"))
            last_status = str(manifest.get("status", "<missing>"))
            if last_status in {"ready", "running"}:
                return manifest
            if last_status in {"failed", "exited", "unreachable"}:
                raise RuntimeError(
                    f"managed service {service_id} reached {last_status}; inspect {path}"
                )
        time.sleep(1)
    raise RuntimeError(
        f"managed service {service_id} did not become ready before timeout; "
        f"last status: {last_status}; manifest: {path}"
    )


def assert_service_manifest(manifest: dict[str, Any], *, expected_engine: str) -> None:
    if manifest.get("engine") != expected_engine:
        raise RuntimeError(f"expected {expected_engine} service, got {manifest.get('engine')}")
    if manifest.get("device_policy") != "gpu_required":
        raise RuntimeError(
            "managed service did not record gpu_required device policy; "
            f"manifest device_policy={manifest.get('device_policy')!r}"
        )
    endpoint = manifest.get("endpoint_url")
    if not isinstance(endpoint, str) or not endpoint.startswith("http://"):
        raise RuntimeError(f"managed service endpoint is invalid: {endpoint!r}")
    canonical = manifest.get("canonical_model_id")
    if not isinstance(canonical, str) or not canonical.strip():
        raise RuntimeError("managed service manifest is missing canonical_model_id")


def wait_local_endpoint(manifest: dict[str, Any], timeout: int) -> None:
    endpoint = str(manifest.get("endpoint_url") or "")
    parsed = urlparse(endpoint)
    host = parsed.hostname
    port = parsed.port
    if not host or not port:
        raise RuntimeError(f"managed service endpoint is invalid: {endpoint!r}")
    deadline = time.monotonic() + timeout
    last_error = "endpoint was not checked"
    while time.monotonic() < deadline:
        try:
            conn = http.client.HTTPConnection(host, port, timeout=5)
            conn.request("GET", "/v1/models")
            response = conn.getresponse()
            response.read()
            if response.status == 200:
                return
            last_error = f"GET /v1/models returned HTTP {response.status}"
        except OSError as error:
            last_error = str(error)
        finally:
            try:
                conn.close()  # type: ignore[name-defined]
            except Exception:
                pass
        time.sleep(2)
    raise RuntimeError(
        f"managed service endpoint did not become reachable before timeout: {last_error}"
    )


def assert_chat_output(output: str, *, require_tool_call: bool) -> None:
    required = [
        "chat response",
        "provider: local",
        "rocm tools: enabled",
    ]
    for needle in required:
        if needle not in output:
            raise RuntimeError(f"chat output did not contain `{needle}`:\n{output}")
    forbidden = [
        "No local assistant is running yet",
        "local provider has no ready managed service",
        "CPU fallback",
        "cpu_only",
    ]
    for needle in forbidden:
        if needle in output:
            raise RuntimeError(f"chat output contained forbidden `{needle}`:\n{output}")
    if require_tool_call:
        if "ROCm checks used" not in output or "none requested" in output:
            raise RuntimeError(
                "local assistant did not request a ROCm tool call; retry without "
                "--require-tool-call or choose a tool-calling model/prompt.\n"
                + output
            )


def find_ready_service_id(data_dir: Path, model: str) -> str:
    services_dir = data_dir / "services"
    if not services_dir.is_dir():
        raise RuntimeError(f"no managed services directory found: {services_dir}")
    matches: list[tuple[int, str]] = []
    for path in services_dir.glob("*.json"):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if manifest.get("status") not in {"ready", "running"}:
            continue
        if manifest.get("engine") != "pytorch":
            continue
        names = {
            str(manifest.get("model_ref", "")),
            str(manifest.get("canonical_model_id", "")),
        }
        if model and not any(name.lower() == model.lower() for name in names):
            continue
        created = int(manifest.get("created_at_unix_ms") or 0)
        service_id = manifest.get("service_id")
        if isinstance(service_id, str) and service_id:
            matches.append((created, service_id))
    if not matches:
        raise RuntimeError(
            f"no ready PyTorch managed service for `{model}`; run without --skip-serve"
        )
    matches.sort(reverse=True)
    return matches[0][1]


def localize_huggingface_cache(env: dict[str, str], repo_root: Path) -> None:
    cache_root = Path(env.get("ROCM_CLI_CACHE_DIR", repo_root / "target" / "test-cache"))
    hf_root = cache_root / "huggingface"
    env.setdefault("HF_HOME", str(hf_root))
    env.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_root / "hub"))
    env.setdefault("TRANSFORMERS_CACHE", str(hf_root / "transformers"))


def run_self_test() -> int:
    fake_rocm = Path("target/debug") / exe_name("rocm")
    args = argparse.Namespace(
        model="qwen",
        chat_model=None,
        prompt=DEFAULT_PROMPT,
        host="127.0.0.1",
        port=11442,
        runtime_id="release-pip-gfx120x-all-7-14-0a20260601",
        env_id="windows-release-pip-gfx120x-all-7-14-0a20260601-3-12",
    )
    serve = build_serve_command(args, fake_rocm)
    assert "--device" in serve
    assert "gpu_required" in serve
    assert "cpu" not in " ".join(serve).lower()
    chat = build_chat_command(args, fake_rocm, "Qwen/Qwen2.5-1.5B-Instruct")
    assert "--tools" in chat
    assert "--provider" in chat and "local" in chat
    assert parse_service_id("managed service launched\n  service_id: svc-qwen\n") == "svc-qwen"
    with tempfile.TemporaryDirectory() as temp:
        data_dir = Path(temp)
        services = data_dir / "services"
        services.mkdir()
        manifest_path = services / "svc-qwen.json"
        manifest = {
            "service_id": "svc-qwen",
            "engine": "pytorch",
            "model_ref": "qwen",
            "canonical_model_id": "Qwen/Qwen2.5-1.5B-Instruct",
            "endpoint_url": "http://127.0.0.1:11442/v1",
            "status": "ready",
            "device_policy": "gpu_required",
            "created_at_unix_ms": 1,
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        loaded = wait_ready_manifest(data_dir, "svc-qwen", timeout=1)
        assert_service_manifest(loaded, expected_engine="pytorch")
        assert find_ready_service_id(data_dir, "qwen") == "svc-qwen"
    assert rocm_cli_data_dir({"ROCM_CLI_DATA_DIR": "custom-data"}) == Path("custom-data")
    assert_chat_output(
        "chat response\n  provider: local\n  model: qwen\n  rocm tools: enabled\n\nhello",
        require_tool_call=False,
    )
    print("local-assistant-gpu-test self-test: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
