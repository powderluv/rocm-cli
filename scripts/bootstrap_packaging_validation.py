#!/usr/bin/env python3
"""Offline validation harness for future bootstrap assistant and launcher spikes.

The harness intentionally uses fakes and fixtures. It does not download a
llamafile, model weights, TheRock wheels, ROCm artifacts, or driver packages,
and it does not prove live GPU execution. It validates the contract that
production code must keep when those future pieces are implemented.
"""

from __future__ import annotations

import argparse
import hashlib
import http.server
import ipaddress
import json
import os
import posixpath
import shutil
import socket
import threading
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "bootstrap-packaging-validation-v1"
LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}
READ_ONLY_TOOLS = {
    "doctor",
    "bridge_snapshot",
    "gpu_snapshot",
    "engines",
    "services",
    "service_logs",
    "automations",
    "natural_language_plan",
    "rocm_command",
    "update_check",
    "install_sdk_dry_run",
    "driver_plan",
    "windows_driver_guidance",
    "wsl_rocdxg_guidance",
}
MUTATING_TOOLS = {
    "install_sdk",
    "install_engine",
    "launch_server",
    "stop_server",
    "watcher_enable",
    "watcher_disable",
}
UNSUPPORTED_SHELL_TOKENS = {
    "apt",
    "apt-get",
    "bash",
    "choco",
    "cmd",
    "cmd.exe",
    "curl",
    "dnf",
    "iwr",
    "irm",
    "pacman",
    "powershell",
    "powershell.exe",
    "pwsh",
    "sh",
    "sudo",
    "su",
    "winget",
    "wget",
    "yum",
    "zsh",
    "zypper",
}
SHELL_SEPARATORS = ("&&", "||", ";", "|", "`", "$(", "<(")


class HarnessError(Exception):
    """A bootstrap packaging validation check failed."""


@dataclass(frozen=True)
class ToolDecision:
    kind: str
    reason: str
    argv: list[str] | None = None


@dataclass(frozen=True)
class LauncherResult:
    target_dir: Path
    reused: bool


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_fixture_path() -> Path:
    return repo_root() / "tests" / "fixtures" / "bootstrap_packaging" / "fake_bootstrap_matrix.json"


def default_self_test_root() -> Path:
    return repo_root() / ".rocm-work" / "tests" / f"bootstrap-packaging-{os.getpid()}"


def fail(message: str) -> None:
    raise HarnessError(message)


def expect(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def load_matrix(path: Path) -> dict[str, Any]:
    try:
        matrix = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        fail(f"fixture matrix is not valid JSON: {path}: {error}")
    expect(matrix.get("schema_version") == SCHEMA_VERSION, "fixture matrix schema version drifted")
    for key in ("doctor_fixtures", "prompt_cases", "adversarial_cases"):
        expect(key in matrix, f"fixture matrix is missing {key}")
    return matrix


def is_loopback_host(host: str) -> bool:
    if host in LOOPBACK_HOSTS:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def normalized_rocm_command_args(arguments: dict[str, Any]) -> list[str]:
    values = arguments.get("args")
    if not isinstance(values, list) or not values or len(values) > 64:
        fail("rocm_command requires 1 to 64 argv-style args")
    args: list[str] = []
    for value in values:
        if not isinstance(value, str):
            fail("rocm_command args entries must be strings")
        arg = value.strip()
        if not arg:
            fail("rocm_command args entries must be non-empty")
        if any(control in arg for control in ("\0", "\n", "\r")):
            fail("rocm_command args must not contain control characters")
        if len(arg) > 512:
            fail("rocm_command arg is too long")
        args.append(arg)
    if args and args[0].lower() == "rocm":
        args = args[1:]
    if args and args[0].lower() == "comfy":
        args[0] = "comfyui"
    if not args:
        fail("rocm_command args should omit the leading rocm program name")
    return args


def contains_unsupported_shell_flow(args: list[str]) -> str | None:
    for arg in args:
        lower = arg.lower()
        token = Path(lower).name
        if lower in UNSUPPORTED_SHELL_TOKENS or token in UNSUPPORTED_SHELL_TOKENS:
            return lower
        if any(separator in arg for separator in SHELL_SEPARATORS):
            return arg
    return None


def contains_unreviewed_yes(args: list[str]) -> str | None:
    for arg in args:
        if arg.lower() in {"--yes", "-y", "/y"}:
            return arg
    return None


def rocm_command_is_read_only(args: list[str]) -> bool:
    first = args[0].lower() if args else ""
    second = args[1].lower() if len(args) > 1 else None
    if first in {"doctor", "version", "model", "models", "daemon", "logs"}:
        return True
    if first == "update":
        return "--apply" not in args
    if first == "runtimes":
        return second is None or second == "list"
    if first == "engines":
        return second == "list"
    if first == "services":
        return second is None or second in {"list", "logs"}
    if first == "automations":
        return second is None or second == "list"
    if first == "config":
        return second == "show"
    if first == "comfyui":
        return second is None or second in {"status", "logs", "log"}
    if first == "uninstall":
        return "--dry-run" in args
    return False


def rocm_command_is_known_mutation(args: list[str]) -> bool:
    first = args[0].lower() if args else ""
    second = args[1].lower() if len(args) > 1 else None
    if first in {"serve", "install", "uninstall"}:
        return True
    if first == "engines" and second == "install":
        return True
    if first == "services" and second in {"stop", "restart"}:
        return True
    if first == "runtimes" and second in {"activate", "adopt", "import", "rollback", "uninstall"}:
        return True
    if first == "config" and second not in {None, "show"}:
        return True
    if first == "automations" and second in {"enable", "disable", "approve", "reject", "edit"}:
        return True
    if first == "comfyui" and second in {"install", "start", "stop", "restart", "uninstall"}:
        return True
    if first == "update" and "--apply" in args:
        return True
    return False


def system_prefix_requires_ack(prefix: str) -> bool:
    path = Path(prefix)
    if not path.is_absolute():
        return False
    home = Path.home()
    try:
        path.resolve().relative_to(home.resolve())
        return False
    except (OSError, ValueError):
        return True


def build_install_sdk_args(arguments: dict[str, Any], *, dry_run: bool) -> list[str]:
    version = arguments.get("version")
    build_date = arguments.get("build_date")
    if version and build_date:
        fail("install_sdk accepts either version or build_date, not both")
    channel = str(arguments.get("channel", "release"))
    install_format = str(arguments.get("format", "pip"))
    argv = ["install", "sdk", "--channel", channel, "--format", install_format]
    prefix = arguments.get("prefix")
    if prefix is not None:
        if not isinstance(prefix, str) or not prefix.strip():
            fail("install_sdk prefix must be a non-empty string")
        if system_prefix_requires_ack(prefix) and not arguments.get("allow_system_prefix"):
            fail("install_sdk system prefix requires allow_system_prefix=true")
        argv.extend(["--prefix", prefix])
    if version:
        argv.extend(["--version", str(version)])
    if build_date:
        argv.extend(["--build-date", str(build_date)])
    if dry_run:
        argv.append("--dry-run")
    return argv


def build_install_engine_args(arguments: dict[str, Any]) -> list[str]:
    engine = arguments.get("engine")
    if not isinstance(engine, str) or not engine.strip():
        fail("install_engine requires engine")
    return ["engines", "install", engine.strip()]


def build_launch_server_args(arguments: dict[str, Any]) -> list[str]:
    model = arguments.get("model")
    if not isinstance(model, str) or not model.strip():
        fail("launch_server requires model")
    host = str(arguments.get("host", "127.0.0.1"))
    if not is_loopback_host(host):
        fail("launch_server rejected public bind; bootstrap server must stay on loopback")
    device = str(arguments.get("device", arguments.get("device_policy", "gpu_required")))
    if device != "gpu_required":
        fail("launch_server rejected CPU fallback; bootstrap serving requires gpu_required")
    argv = ["serve", model.strip(), "--managed"]
    engine = arguments.get("engine")
    if engine:
        argv.extend(["--engine", str(engine)])
    argv.extend(["--device", "gpu_required", "--host", host])
    port = arguments.get("port")
    if port is not None:
        argv.extend(["--port", str(port)])
    return argv


def validate_tool_call(tool_call: dict[str, Any], *, gpu_state: str = "supported") -> ToolDecision:
    name = tool_call.get("name")
    arguments = tool_call.get("arguments", {})
    if not isinstance(name, str) or not name:
        fail("tool call is missing a function name")
    if not isinstance(arguments, dict):
        fail(f"tool call {name} arguments must be an object")

    if name == "shell":
        return ToolDecision("rejected", "unknown tool: shell access is not part of the bootstrap contract")
    if name not in READ_ONLY_TOOLS and name not in MUTATING_TOOLS:
        return ToolDecision("rejected", f"unknown tool: {name}")

    try:
        if name == "rocm_command":
            argv = normalized_rocm_command_args(arguments)
            if unsupported := contains_unsupported_shell_flow(argv):
                return ToolDecision("rejected", f"unsupported shell or package-manager flow: {unsupported}")
            if yes_arg := contains_unreviewed_yes(argv):
                return ToolDecision("rejected", f"model-proposed {yes_arg} is not allowed")
            if rocm_command_is_read_only(argv):
                return ToolDecision("run", "read-only rocm_command", argv)
            if gpu_state == "no_supported_gpu":
                return ToolDecision(
                    "rejected",
                    "hard no supported AMD GPU preflight blocks mutating bootstrap action",
                    argv,
                )
            if rocm_command_is_known_mutation(argv):
                return ToolDecision("approval_required", "mutating rocm command requires approval", argv)
            return ToolDecision("rejected", "unsupported rocm_command")
        if name == "install_sdk_dry_run":
            return ToolDecision("run", "TheRock dry-run is read-only", build_install_sdk_args(arguments, dry_run=True))
        if name == "windows_driver_guidance":
            if arguments.get("source") != "official_amd_driver_page":
                return ToolDecision("rejected", "Windows driver guidance must use AMD official driver flow")
            return ToolDecision("run", "manual Windows driver guidance only")
        if name in {"driver_plan", "wsl_rocdxg_guidance"}:
            return ToolDecision("run", f"{name} is read-only guidance")
        if name in {"doctor", "bridge_snapshot", "gpu_snapshot", "engines", "services", "service_logs", "automations", "natural_language_plan", "update_check"}:
            return ToolDecision("run", f"{name} is read-only")
        if gpu_state == "no_supported_gpu":
            return ToolDecision(
                "rejected",
                "hard no supported AMD GPU preflight blocks mutating bootstrap action",
            )
        if name == "install_sdk":
            return ToolDecision("approval_required", "install_sdk requires approval", build_install_sdk_args(arguments, dry_run=False))
        if name == "install_engine":
            return ToolDecision("approval_required", "install_engine requires approval", build_install_engine_args(arguments))
        if name == "launch_server":
            return ToolDecision("approval_required", "launch_server requires approval", build_launch_server_args(arguments))
        if name in {"stop_server", "watcher_enable", "watcher_disable"}:
            return ToolDecision("approval_required", f"{name} requires approval")
    except HarnessError as error:
        return ToolDecision("rejected", str(error))
    return ToolDecision("rejected", f"unhandled tool: {name}")


class FakeBootstrapChatHandler(http.server.BaseHTTPRequestHandler):
    server: "FakeBootstrapHTTPServer"

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(200, {"status": "ok", "device_policy": self.server.device_policy})
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self._send_json(404, {"error": "not found"})
            return
        length = int(self.headers.get("content-length", "0"))
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid json"})
            return
        if not payload.get("tools"):
            self._send_json(400, {"error": "OpenAI-style tools are required"})
            return
        messages = payload.get("messages", [])
        prompt = ""
        if messages and isinstance(messages[-1], dict):
            prompt = str(messages[-1].get("content", ""))
        tool_call = self.server.response_by_prompt.get(prompt)
        if tool_call is None:
            tool_call = {"name": "doctor", "arguments": {}}
        self._send_json(
            200,
            {
                "id": "fake-bootstrap-chat",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "fixture response",
                            "tool_calls": [
                                {
                                    "id": "call-fixture",
                                    "type": "function",
                                    "function": {
                                        "name": tool_call["name"],
                                        "arguments": json.dumps(tool_call.get("arguments", {})),
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            },
        )


class FakeBootstrapHTTPServer(http.server.ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        response_by_prompt: dict[str, dict[str, Any]],
        *,
        device_policy: str,
    ) -> None:
        super().__init__(server_address, FakeBootstrapChatHandler)
        self.response_by_prompt = response_by_prompt
        self.device_policy = device_policy


class FakeBootstrapServer:
    def __init__(
        self,
        *,
        host: str,
        device_policy: str,
        jinja: bool,
        response_by_prompt: dict[str, dict[str, Any]],
    ) -> None:
        if not is_loopback_host(host):
            fail("fake bootstrap server rejected public bind")
        if device_policy != "gpu_required":
            fail("fake bootstrap server rejected CPU fallback")
        if not jinja:
            fail("fake bootstrap server requires --jinja for tool-call validation")
        self.host = host
        self.response_by_prompt = response_by_prompt
        self.device_policy = device_policy
        self.httpd: FakeBootstrapHTTPServer | None = None
        self.thread: threading.Thread | None = None

    def __enter__(self) -> "FakeBootstrapServer":
        self.httpd = FakeBootstrapHTTPServer(
            (self.host, 0),
            self.response_by_prompt,
            device_policy=self.device_policy,
        )
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()
        if self.thread is not None:
            self.thread.join(timeout=5)

    @property
    def url(self) -> str:
        expect(self.httpd is not None, "fake server has not started")
        host, port = self.httpd.server_address[:2]
        return f"http://{host}:{port}"


def chat_completion_tool_call(url: str, prompt: str) -> dict[str, Any]:
    payload = {
        "model": "fixture-qwen-0.8b",
        "messages": [{"role": "user", "content": prompt}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "rocm_command",
                    "parameters": {"type": "object", "properties": {"args": {"type": "array"}}},
                },
            }
        ],
    }
    request = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        body = json.loads(response.read().decode("utf-8"))
    function = body["choices"][0]["message"]["tool_calls"][0]["function"]
    return {
        "name": function["name"],
        "arguments": json.loads(function.get("arguments") or "{}"),
    }


def validate_prompt_matrix(matrix: dict[str, Any]) -> None:
    for case in matrix["prompt_cases"]:
        decision = validate_tool_call(case["tool_call"])
        expect(
            decision.kind == case["expected_decision"],
            f"{case['id']} expected {case['expected_decision']}, got {decision.kind}: {decision.reason}",
        )
        expected_argv = case.get("expected_argv")
        if expected_argv is not None:
            expect(decision.argv == expected_argv, f"{case['id']} argv drifted: {decision.argv}")
        if decision.argv is not None:
            expect("--yes" not in decision.argv and "-y" not in decision.argv, f"{case['id']} smuggled auto-approval")


def validate_adversarial_matrix(matrix: dict[str, Any]) -> None:
    for case in matrix["adversarial_cases"]:
        decision = validate_tool_call(case["tool_call"])
        expect(decision.kind == "rejected", f"{case['id']} should have been rejected, got {decision.kind}")
        needle = case.get("reject_contains")
        if needle:
            expect(needle.lower() in decision.reason.lower(), f"{case['id']} rejection reason drifted: {decision.reason}")


def validate_no_supported_gpu_preflight(matrix: dict[str, Any]) -> None:
    fixture = matrix["doctor_fixtures"]["no_supported_gpu"]
    expect(fixture["status"] == "no_supported_gpu", "no-supported-GPU fixture drifted")
    read_only = validate_tool_call({"name": "doctor", "arguments": {}}, gpu_state="no_supported_gpu")
    expect(read_only.kind == "run", "doctor must still be available on no-supported-GPU hosts")
    for case in matrix["prompt_cases"]:
        if case["expected_decision"] != "approval_required":
            continue
        decision = validate_tool_call(case["tool_call"], gpu_state="no_supported_gpu")
        expect(decision.kind == "rejected", f"{case['id']} mutation passed no-supported-GPU preflight")
        expect(
            "no supported AMD GPU" in decision.reason,
            f"{case['id']} preflight rejection should be explicit: {decision.reason}",
        )


def validate_fake_model_server(matrix: dict[str, Any]) -> None:
    response_by_prompt = {case["prompt"]: case["tool_call"] for case in matrix["prompt_cases"]}
    with FakeBootstrapServer(
        host="127.0.0.1",
        device_policy="gpu_required",
        jinja=True,
        response_by_prompt=response_by_prompt,
    ) as server:
        for case in matrix["prompt_cases"]:
            tool_call = chat_completion_tool_call(server.url, case["prompt"])
            expect(tool_call == case["tool_call"], f"{case['id']} fake model response drifted")
            decision = validate_tool_call(tool_call)
            expect(decision.kind == case["expected_decision"], f"{case['id']} fake server decision drifted")

    for kwargs, expected in [
        ({"host": "0.0.0.0", "device_policy": "gpu_required", "jinja": True}, "public bind"),
        ({"host": "127.0.0.1", "device_policy": "cpu", "jinja": True}, "CPU fallback"),
        ({"host": "127.0.0.1", "device_policy": "gpu_required", "jinja": False}, "--jinja"),
    ]:
        try:
            FakeBootstrapServer(response_by_prompt=response_by_prompt, **kwargs)
        except HarnessError as error:
            expect(expected.lower() in str(error).lower(), f"fake server rejection drifted: {error}")
        else:
            fail(f"fake server accepted invalid startup: {kwargs}")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_zip_payload(payload: Path) -> None:
    seen: set[str] = set()
    try:
        with zipfile.ZipFile(payload) as archive:
            for info in archive.infolist():
                normalized = posixpath.normpath(info.filename.replace("\\", "/"))
                if normalized in {"", "."}:
                    fail("launcher payload contains empty path")
                if normalized.startswith("/") or info.filename.startswith("\\"):
                    fail(f"launcher payload contains absolute path: {info.filename}")
                parts = normalized.split("/")
                if any(part in {"", ".", ".."} for part in parts):
                    fail(f"launcher payload contains unsafe path: {info.filename}")
                if ":" in parts[0]:
                    fail(f"launcher payload contains drive-qualified path: {info.filename}")
                if normalized in seen:
                    fail(f"launcher payload contains duplicate path: {normalized}")
                seen.add(normalized)
    except zipfile.BadZipFile as error:
        fail(f"launcher payload is not a zip archive: {error}")


def extract_launcher_payload(manifest: dict[str, Any], payload: Path, root: Path) -> LauncherResult:
    actual = sha256_bytes(payload.read_bytes())
    expected_hash = manifest.get("payload_sha256")
    if actual != expected_hash:
        fail(f"launcher payload hash mismatch: expected {expected_hash}, got {actual}")
    validate_zip_payload(payload)
    version = manifest.get("version")
    platform_id = manifest.get("platform")
    if not isinstance(version, str) or not isinstance(platform_id, str):
        fail("launcher manifest requires version and platform")
    target_dir = root / "launcher" / version / platform_id
    marker = target_dir / ".rocm-cli-manifest"
    if marker.is_file():
        recorded = json.loads(marker.read_text(encoding="utf-8"))
        if recorded.get("payload_sha256") == actual:
            return LauncherResult(target_dir=target_dir, reused=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(payload) as archive:
        archive.extractall(target_dir)
    marker.write_text(json.dumps({**manifest, "payload_sha256": actual}, indent=2) + "\n", encoding="utf-8")
    return LauncherResult(target_dir=target_dir, reused=False)


def create_zip(path: Path, files: dict[str, str]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)


def validate_launcher_hash_and_extraction(root: Path) -> None:
    launcher_root = root / "launcher-self-test"
    launcher_root.mkdir(parents=True, exist_ok=True)
    payload = launcher_root / "rocm-cli-v0.2.0-test-windows-amd64.zip"
    create_zip(
        payload,
        {
            "rocm-cli/bin/rocm.exe": "fake rocm binary\n",
            "rocm-cli/bin/rocmd.exe": "fake rocmd binary\n",
            "rocm-cli/README.md": "fake readme\n",
        },
    )
    manifest = {
        "version": "0.2.0-test",
        "platform": "windows-amd64",
        "payload_name": payload.name,
        "payload_sha256": sha256_bytes(payload.read_bytes()),
    }

    first = extract_launcher_payload(manifest, payload, launcher_root)
    expect(not first.reused, "first launcher run should extract")
    expect((first.target_dir / "rocm-cli" / "bin" / "rocm.exe").is_file(), "launcher extraction missed rocm.exe")
    second = extract_launcher_payload(manifest, payload, launcher_root)
    expect(second.reused, "second launcher run should reuse extracted version")

    tampered = launcher_root / "tampered.zip"
    create_zip(tampered, {"rocm-cli/bin/rocm.exe": "tampered\n"})
    try:
        extract_launcher_payload(manifest, tampered, launcher_root / "tampered-target")
    except HarnessError as error:
        expect("hash mismatch" in str(error), f"tamper rejection drifted: {error}")
    else:
        fail("tampered launcher payload was accepted")

    unsafe = launcher_root / "unsafe.zip"
    create_zip(unsafe, {"../escape.txt": "bad\n"})
    unsafe_manifest = {**manifest, "payload_name": unsafe.name, "payload_sha256": sha256_bytes(unsafe.read_bytes())}
    try:
        extract_launcher_payload(unsafe_manifest, unsafe, launcher_root / "unsafe-target")
    except HarnessError as error:
        expect("unsafe path" in str(error), f"unsafe payload rejection drifted: {error}")
    else:
        fail("unsafe launcher payload was accepted")


LIVE_GPU_MATRIX = """Live GPU validation still required before production UX:

- Native Windows: run signed/pinned Qwen llamafile with CPU fallback disabled,
  confirm GPU helper use, loopback server bind, OpenAI-style tools with --jinja,
  AMD official driver guidance only, and Windows payload size below 4 GB.
- Native Linux: repeat with the Linux llamafile, verify HIP/GPU path from logs or
  counters, exercise driver dry-run/approval rails, and do not emit raw sudo,
  package-manager, shell, or --yes flows.
- WSL: repeat as a distinct target, verify ROCDXG readiness and WSLInterop/APE
  behavior, and keep driver readiness as ROCDXG guidance instead of native DKMS.
- Single-file launcher: test real signed archive payload extraction, reuse,
  uninstall, and upgrade on Windows, Linux, and WSL.
- After TheRock setup: verify normal local assistant serving remains
  gpu_required and no CPU-backed bootstrap path satisfies rocm serve.
"""


def run_self_test(matrix_path: Path, root: Path) -> None:
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    try:
        matrix = load_matrix(matrix_path)
        validate_prompt_matrix(matrix)
        print("bootstrap packaging validation: prompt matrix ok")
        validate_adversarial_matrix(matrix)
        print("bootstrap packaging validation: adversarial safety matrix ok")
        validate_no_supported_gpu_preflight(matrix)
        print("bootstrap packaging validation: no-supported-GPU preflight ok")
        validate_fake_model_server(matrix)
        print("bootstrap packaging validation: fake OpenAI tool server ok")
        validate_launcher_hash_and_extraction(root)
        print("bootstrap packaging validation: fake launcher extraction ok")
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("bootstrap packaging validation: ok")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=default_fixture_path(), help="Fixture matrix JSON path.")
    parser.add_argument(
        "--self-test-root",
        type=Path,
        default=default_self_test_root(),
        help="Workspace-local root used for fake launcher extraction tests.",
    )
    parser.add_argument("--self-test", action="store_true", help="Run the offline validation harness.")
    parser.add_argument("--print-live-matrix", action="store_true", help="Print live GPU/platform checks not covered by fakes.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.print_live_matrix:
        print(LIVE_GPU_MATRIX.strip())
    if args.self_test or not args.print_live_matrix:
        try:
            run_self_test(args.fixtures, args.self_test_root)
        except HarnessError as error:
            print(f"bootstrap packaging validation failed: {error}", file=os.sys.stderr)
            return 1
        except (OSError, socket.timeout, urllib.error.URLError) as error:
            print(f"bootstrap packaging validation failed: {error}", file=os.sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
