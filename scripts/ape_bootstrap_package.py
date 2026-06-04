#!/usr/bin/env python3
"""Validate and stage the P0 APE bootstrap package contract.

This is an offline packaging harness. It does not download Cosmopolitan,
llamafile, Qwen weights, TheRock wheels, or ROCm artifacts. It makes the
single-exe requirement testable before a production APE builder is wired in:

- one universal AMD64 APE artifact carries Windows and Linux rocm-cli release
  payloads;
- the bootstrap assistant payload embeds a pinned Qwen 0.8B-class llamafile;
- startup is loopback-only, OpenAI/Jinja tool-call ready, AMD GPU-required, and
  never CPU fallback;
- production packaging must prove GPU execution from logs/health, not just a
  process start.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import ipaddress
import io
import json
import os
import posixpath
import re
import shutil
import sys
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "rocm-cli-ape-bootstrap/v1"
DEFAULT_MANIFEST_PATH = "payload/ape-bootstrap.json"
DEFAULT_WINDOWS_SIZE_CAP = 4 * 1024 * 1024 * 1024
DEFAULT_MODEL_NAME = "Qwen3.5-0.8B-Q8_0.llamafile"
DEFAULT_MODEL_FAMILY = "qwen"
DEFAULT_MODEL_PARAMETERS = "0.8B"
DEFAULT_PORT = 11435
REQUIRED_PLATFORMS = {"windows-amd64", "linux-amd64"}
LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}
REQUIRED_GPU_PROOFS = {"llamafile_gpu_log", "health_backend_gpu"}
REQUIRED_ROCM_BACKENDS = {
    "windows-amd64": "ggml-rocm.dll",
    "linux-amd64": "ggml-rocm.so",
}
REQUIRED_LINUX_RUNTIME_DEPENDENCIES = {"ape-x86_64.elf"}


def default_self_test_root() -> Path:
    return repo_root() / ".rocm-work" / "tests" / f"ape-bootstrap-package-{os.getpid()}"
ROCM_RELEASE_FORMAT_BY_PLATFORM = {
    "windows-amd64": "zip",
    "linux-amd64": "tar.gz",
}
ROCM_RELEASE_EXT_BY_PLATFORM = {
    "windows-amd64": ".zip",
    "linux-amd64": ".tar.gz",
}
FORBIDDEN_ARG_TOKENS = {
    "--allow-cpu-fallback",
    "--cpu",
    "cpu",
    "cpu_only",
    "cpu-only",
}


class ApePackageError(Exception):
    """The APE bootstrap packaging contract was violated."""


@dataclass(frozen=True)
class PathSource:
    payload_path: str
    source_path: Path
    expected_sha256: str
    expected_size: int


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def fail(message: str) -> None:
    print(f"APE bootstrap package failed: {message}", file=sys.stderr)
    raise SystemExit(1)


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise ApePackageError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize_sha256(value: Any, label: str) -> str:
    expect(isinstance(value, str), f"{label} must be a sha256 string")
    digest = value.strip().lower()
    expect(
        len(digest) == 64 and all(ch in "0123456789abcdef" for ch in digest),
        f"{label} is not a sha256 digest",
    )
    return digest


def is_loopback_host(host: str) -> bool:
    if host in LOOPBACK_HOSTS:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def normalized_payload_path(value: Any, *, label: str) -> str:
    expect(isinstance(value, str) and value.strip(), f"{label} must be a non-empty path")
    raw = value.strip().replace("\\", "/")
    normalized = posixpath.normpath(raw)
    expect(normalized not in {"", "."}, f"{label} must not be empty")
    expect(not normalized.startswith("/") and not raw.startswith("/"), f"{label} must be relative")
    parts = normalized.split("/")
    expect(
        all(part not in {"", ".", ".."} for part in parts),
        f"{label} contains an unsafe path: {value}",
    )
    expect(":" not in parts[0], f"{label} contains a drive or URI separator: {value}")
    expect(normalized.startswith("payload/"), f"{label} must live under payload/: {value}")
    return normalized


def source_path_from(value: Any, *, base_dir: Path | None, label: str) -> Path | None:
    if value is None:
        return None
    expect(isinstance(value, str) and value.strip(), f"{label} must be a path string")
    path = Path(value)
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path


def arg_values(args: Iterable[Any]) -> list[str]:
    values: list[str] = []
    for value in args:
        expect(isinstance(value, str) and value.strip(), "startup args must be non-empty strings")
        expect("\0" not in value and "\n" not in value and "\r" not in value, "startup args contain control characters")
        values.append(value.strip())
    return values


def find_arg_value(args: list[str], *names: str) -> str | None:
    for index, arg in enumerate(args):
        if arg in names and index + 1 < len(args):
            return args[index + 1]
    for arg in args:
        for name in names:
            prefix = f"{name}="
            if arg.startswith(prefix):
                return arg[len(prefix) :]
    return None


def contains_arg(args: list[str], *names: str) -> bool:
    return any(arg in names for arg in args)


def ngl_value(args: list[str]) -> int | None:
    value = find_arg_value(args, "-ngl", "--n-gpu-layers")
    if value is None:
        return None
    try:
        return int(value)
    except ValueError as error:
        raise ApePackageError(f"GPU layer count must be an integer, got {value!r}") from error


def validate_release_archive(entry: Any, *, base_dir: Path | None) -> tuple[str, PathSource | None, int]:
    expect(isinstance(entry, dict), "release archive entries must be objects")
    platform = entry.get("platform")
    expect(platform in REQUIRED_PLATFORMS, f"release archive platform must be one of {sorted(REQUIRED_PLATFORMS)}")
    expected_format = ROCM_RELEASE_FORMAT_BY_PLATFORM[platform]
    actual_format = entry.get("format")
    expect(actual_format == expected_format, f"{platform} release format must be {expected_format}")
    name = entry.get("name")
    expect(isinstance(name, str) and name.strip(), f"{platform} release name must be non-empty")
    expect(Path(name).name == name and "/" not in name and "\\" not in name, f"{platform} release name must be a plain file name")
    expect(name.endswith(ROCM_RELEASE_EXT_BY_PLATFORM[platform]), f"{platform} release name must end with {ROCM_RELEASE_EXT_BY_PLATFORM[platform]}")
    payload_path = normalized_payload_path(entry.get("payload_path"), label=f"{platform} payload_path")
    expect(payload_path.startswith("payload/releases/"), f"{platform} release must live under payload/releases/")
    digest = normalize_sha256(entry.get("sha256"), f"{platform} release sha256")
    size = entry.get("size")
    expect(isinstance(size, int) and size > 0, f"{platform} release size must be positive")
    source_path = source_path_from(entry.get("source_path"), base_dir=base_dir, label=f"{platform} source_path")
    source: PathSource | None = None
    if source_path is not None:
        expect(source_path.is_file(), f"{platform} release source does not exist: {source_path}")
        actual_size = source_path.stat().st_size
        expect(actual_size == size, f"{platform} release size mismatch: manifest={size} actual={actual_size}")
        actual_digest = sha256_file(source_path)
        expect(actual_digest == digest, f"{platform} release sha256 mismatch: manifest={digest} actual={actual_digest}")
        source = PathSource(payload_path, source_path, digest, size)
    return platform, source, size


def validate_model(model: Any, *, base_dir: Path | None) -> tuple[PathSource | None, int]:
    expect(isinstance(model, dict), "bootstrap_model must be an object")
    expect(model.get("kind") == "llamafile", "bootstrap_model.kind must be llamafile")
    expect(model.get("embedded") is True, "bootstrap model must be embedded in the APE payload")
    name = model.get("name")
    expect(isinstance(name, str) and name.strip(), "bootstrap_model.name must be non-empty")
    llamafile_name = name.endswith(".llamafile") or name.endswith(".llamafile.exe")
    expect(Path(name).name == name and llamafile_name, "bootstrap_model.name must be a llamafile file name")
    family = str(model.get("family", "")).strip().lower()
    expect(family == DEFAULT_MODEL_FAMILY, "bootstrap_model.family must be qwen")
    parameters = str(model.get("parameters", "")).strip().lower()
    expect(
        bool(re.fullmatch(r"0[.-]?8b", parameters)),
        "bootstrap_model.parameters must be 0.8B",
    )
    expect("qwen" in name.lower(), "bootstrap_model.name must identify Qwen")
    expect("0.8b" in name.lower() or "0-8b" in name.lower(), "bootstrap_model.name must identify the 0.8B model")
    payload_path = normalized_payload_path(model.get("payload_path"), label="bootstrap_model.payload_path")
    expect(payload_path.startswith("payload/bootstrap/"), "bootstrap model must live under payload/bootstrap/")
    digest = normalize_sha256(model.get("sha256"), "bootstrap_model.sha256")
    size = model.get("size")
    expect(isinstance(size, int) and size > 0, "bootstrap_model.size must be positive")
    source_path = source_path_from(model.get("source_path"), base_dir=base_dir, label="bootstrap_model.source_path")
    source: PathSource | None = None
    if source_path is not None:
        expect(source_path.is_file(), f"bootstrap model source does not exist: {source_path}")
        actual_size = source_path.stat().st_size
        expect(actual_size == size, f"bootstrap model size mismatch: manifest={size} actual={actual_size}")
        actual_digest = sha256_file(source_path)
        expect(actual_digest == digest, f"bootstrap model sha256 mismatch: manifest={digest} actual={actual_digest}")
        source = PathSource(payload_path, source_path, digest, size)
    return source, size


def validate_bootstrap_gpu_backends(backends: Any, *, base_dir: Path | None) -> list[PathSource]:
    expect(isinstance(backends, list) and backends, "bootstrap_gpu_backends must be a non-empty list")
    seen: set[str] = set()
    sources: list[PathSource] = []
    for backend in backends:
        expect(isinstance(backend, dict), "bootstrap_gpu_backends entries must be objects")
        platform = backend.get("platform")
        expect(platform in REQUIRED_ROCM_BACKENDS, f"unsupported bootstrap GPU backend platform: {platform}")
        expect(platform not in seen, f"duplicate bootstrap GPU backend for {platform}")
        seen.add(platform)
        expected_name = REQUIRED_ROCM_BACKENDS[str(platform)]
        name = backend.get("name")
        expect(name == expected_name, f"{platform} bootstrap GPU backend must be {expected_name}")
        payload_path = normalized_payload_path(backend.get("payload_path"), label=f"{platform} backend payload_path")
        expect(
            payload_path == f"payload/bootstrap/{expected_name}",
            f"{platform} bootstrap GPU backend must live beside the llamafile",
        )
        digest = normalize_sha256(backend.get("sha256"), f"{platform} backend sha256")
        size = backend.get("size")
        expect(isinstance(size, int) and size > 0, f"{platform} backend size must be positive")
        source_path = source_path_from(backend.get("source_path"), base_dir=base_dir, label=f"{platform} backend source_path")
        if source_path is not None:
            expect(source_path.is_file(), f"{platform} backend source does not exist: {source_path}")
            actual_size = source_path.stat().st_size
            expect(actual_size == size, f"{platform} backend size mismatch: manifest={size} actual={actual_size}")
            actual_digest = sha256_file(source_path)
            expect(actual_digest == digest, f"{platform} backend sha256 mismatch: manifest={digest} actual={actual_digest}")
            sources.append(PathSource(payload_path, source_path, digest, size))
    missing = sorted(set(REQUIRED_ROCM_BACKENDS) - seen)
    expect(not missing, f"bootstrap_gpu_backends missing platforms: {', '.join(missing)}")
    return sources


def validate_bootstrap_runtime_dependencies(deps: Any, *, base_dir: Path | None) -> tuple[list[PathSource], list[int]]:
    expect(
        isinstance(deps, list) and deps,
        "bootstrap_runtime_dependencies must be a non-empty list",
    )
    seen_payloads: set[str] = set()
    seen_platforms: set[str] = set()
    sources: list[PathSource] = []
    sizes: list[int] = []
    for dep in deps:
        expect(isinstance(dep, dict), "bootstrap_runtime_dependencies entries must be objects")
        platform = dep.get("platform")
        expect(platform in REQUIRED_PLATFORMS, f"unsupported bootstrap runtime dependency platform: {platform}")
        seen_platforms.add(str(platform))
        name = dep.get("name")
        expect(isinstance(name, str) and name.strip(), f"{platform} runtime dependency name must be non-empty")
        expect(Path(name).name == name and "/" not in name and "\\" not in name, f"{platform} runtime dependency name must be a plain file name")
        payload_path = normalized_payload_path(dep.get("payload_path"), label=f"{platform} runtime dependency payload_path")
        expect(
            payload_path == f"payload/bootstrap/{name}",
            f"{platform} runtime dependency must live beside the bootstrap llamafile",
        )
        expect(payload_path not in seen_payloads, f"duplicate bootstrap runtime dependency payload: {payload_path}")
        seen_payloads.add(payload_path)
        digest = normalize_sha256(dep.get("sha256"), f"{platform} runtime dependency sha256")
        size = dep.get("size")
        expect(isinstance(size, int) and size > 0, f"{platform} runtime dependency size must be positive")
        sizes.append(size)
        source_path = source_path_from(dep.get("source_path"), base_dir=base_dir, label=f"{platform} runtime dependency source_path")
        if source_path is not None:
            expect(source_path.is_file(), f"{platform} runtime dependency source does not exist: {source_path}")
            actual_size = source_path.stat().st_size
            expect(actual_size == size, f"{platform} runtime dependency size mismatch: manifest={size} actual={actual_size}")
            actual_digest = sha256_file(source_path)
            expect(actual_digest == digest, f"{platform} runtime dependency sha256 mismatch: manifest={digest} actual={actual_digest}")
            sources.append(PathSource(payload_path, source_path, digest, size))
    missing = sorted(REQUIRED_PLATFORMS - seen_platforms)
    expect(not missing, f"bootstrap_runtime_dependencies missing platforms: {', '.join(missing)}")
    linux_missing = sorted(REQUIRED_LINUX_RUNTIME_DEPENDENCIES - seen_payloads_by_name(deps, "linux-amd64"))
    expect(
        not linux_missing,
        f"bootstrap_runtime_dependencies missing required Linux files: {', '.join(linux_missing)}",
    )
    return sources, sizes


def seen_payloads_by_name(entries: list[Any], platform: str) -> set[str]:
    names: set[str] = set()
    for entry in entries:
        if isinstance(entry, dict) and entry.get("platform") == platform:
            name = entry.get("name")
            if isinstance(name, str):
                names.add(name)
    return names


def validate_startup(startup: Any, *, model_payload_path: str) -> list[str]:
    expect(isinstance(startup, dict), "startup must be an object")
    expect(startup.get("mode") == "server", "startup.mode must be server")
    host = startup.get("host")
    expect(isinstance(host, str) and is_loopback_host(host), "startup.host must be loopback")
    expect(startup.get("device_policy") == "gpu_required", "startup.device_policy must be gpu_required")
    expect(startup.get("gpu_vendor") == "amd", "startup.gpu_vendor must be amd")
    expect(startup.get("allow_cpu_fallback") is False, "startup.allow_cpu_fallback must be false")
    expect(startup.get("fallback_policy") == "fail_loudly", "startup.fallback_policy must be fail_loudly")
    port = startup.get("port")
    expect(isinstance(port, int) and 1 <= port <= 65535, "startup.port must be a TCP port")
    tool_calling = startup.get("tool_calling")
    expect(isinstance(tool_calling, dict), "startup.tool_calling must be an object")
    expect(tool_calling.get("openai_compatible") is True, "startup.tool_calling.openai_compatible must be true")
    expect(tool_calling.get("jinja") is True, "startup.tool_calling.jinja must be true")
    proofs = startup.get("gpu_proof_required")
    expect(isinstance(proofs, list) and proofs, "startup.gpu_proof_required must be a non-empty list")
    proof_set = {str(proof) for proof in proofs}
    expect(
        REQUIRED_GPU_PROOFS.issubset(proof_set),
        f"startup.gpu_proof_required must include {sorted(REQUIRED_GPU_PROOFS)}",
    )

    args = arg_values(startup.get("args", []))
    expect(contains_arg(args, "--server"), "startup args must include --server")
    expect(contains_arg(args, "--jinja"), "startup args must include --jinja")
    model_arg = find_arg_value(args, "-m", "--model")
    expect(model_arg is None, "startup runs the embedded .llamafile directly; it must not pass a separate CPU/GGUF model")
    host_arg = find_arg_value(args, "--host")
    expect(host_arg == host, "startup --host must match startup.host")
    gpu_arg = find_arg_value(args, "--gpu")
    expect(gpu_arg == "amd", "startup args must force --gpu amd")
    layers = ngl_value(args)
    expect(layers is not None and layers >= 999, "startup args must request maximum GPU offload with -ngl/--n-gpu-layers >= 999")
    log_level = find_arg_value(args, "-lv", "--verbosity", "--log-verbosity")
    expect(log_level == "0", "startup args must keep llama.cpp foreground logs quiet with -lv 0")
    lowered = [arg.lower() for arg in args]
    expect("--gpu" not in lowered or "--gpu disable" not in " ".join(lowered), "startup args must not disable GPU")
    expect(not any(token in lowered for token in FORBIDDEN_ARG_TOKENS), "startup args must not include CPU fallback tokens")
    expect(layers != 0, "startup args must not request -ngl 0")
    return [
        "startup loopback server ok",
        "startup OpenAI/Jinja tool mode ok",
        "startup AMD GPU-required args ok",
    ]


def validate_rocm_bootstrap_command(command: Any, *, model_payload_path: str, host: str, port: int) -> None:
    expect(isinstance(command, list), "rocm_bootstrap_command must be an argv list")
    args = arg_values(command)
    expect(args[:2] == ["bootstrap", "assistant"], "rocm_bootstrap_command must call bootstrap assistant")
    expect(find_arg_value(args, "--llamafile") == f"{{extract_root}}/{model_payload_path}", "rocm_bootstrap_command must pass the extracted model path")
    expect(find_arg_value(args, "--host") == host, "rocm_bootstrap_command --host must match startup.host")
    expect(find_arg_value(args, "--port") == str(port), "rocm_bootstrap_command --port must match startup.port")
    expect(find_arg_value(args, "--device") == "gpu_required", "rocm_bootstrap_command must pass --device gpu_required")
    lowered = [arg.lower() for arg in args]
    expect("--allow-cpu-fallback" not in lowered, "rocm_bootstrap_command must not allow CPU fallback")
    expect("--json" not in lowered and "--validate-only" not in lowered, "rocm_bootstrap_command must start the assistant, not validation-only output")


def validate_post_therock_setup(value: Any) -> list[str]:
    expect(isinstance(value, dict), "post_therock_setup must be an object")
    self_install = value.get("self_install_cli")
    expect(isinstance(self_install, dict), "post_therock_setup.self_install_cli must be an object")
    expect(self_install.get("enabled") is True, "self_install_cli.enabled must be true")
    expect(
        self_install.get("user_selected_target_required") is True,
        "self_install_cli must require a user-selected install folder",
    )
    expect(
        self_install.get("target_placeholder") == "{user_selected_cli_install_dir}",
        "self_install_cli.target_placeholder must be {user_selected_cli_install_dir}",
    )

    install_command = arg_values(self_install.get("install_command", []))
    expect(
        install_command == [
            "bootstrap",
            "install-cli",
            "--target",
            "{user_selected_cli_install_dir}",
        ],
        "self_install_cli.install_command must install to the user-selected target",
    )
    expect("--add-to-path" not in install_command, "self_install_cli.install_command must not silently change PATH")

    path_prompt = self_install.get("path_prompt")
    expect(isinstance(path_prompt, dict), "self_install_cli.path_prompt must be an object")
    expect(path_prompt.get("after_install") is True, "PATH prompt must happen after self-install")
    expect(
        isinstance(path_prompt.get("copy"), str) and "PATH" in path_prompt["copy"],
        "PATH prompt must use simple user-facing copy",
    )
    path_command = arg_values(path_prompt.get("add_to_path_command", []))
    expect(
        path_command == [
            "bootstrap",
            "install-cli",
            "--target",
            "{user_selected_cli_install_dir}",
            "--add-to-path",
        ],
        "path_prompt.add_to_path_command must be explicit and target the selected folder",
    )
    return ["post-TheRock CLI self-install handoff ok"]


def validate_size_budget(
    manifest: dict[str, Any],
    release_sizes: list[int],
    model_size: int,
    backend_sizes: list[int],
    runtime_dependency_sizes: list[int],
) -> None:
    ape = manifest.get("ape", {})
    expect(isinstance(ape, dict), "ape must be an object")
    cap = ape.get("max_windows_executable_size", DEFAULT_WINDOWS_SIZE_CAP)
    expect(isinstance(cap, int) and cap > 0, "ape.max_windows_executable_size must be positive")
    launcher_overhead = ape.get("launcher_overhead_estimate", 16 * 1024 * 1024)
    expect(isinstance(launcher_overhead, int) and launcher_overhead >= 0, "ape.launcher_overhead_estimate must be non-negative")
    total = (
        model_size
        + sum(backend_sizes)
        + sum(runtime_dependency_sizes)
        + sum(release_sizes)
        + launcher_overhead
    )
    expect(total < cap, f"APE payload estimate {total} bytes exceeds Windows executable cap {cap}")


def validate_manifest(manifest: dict[str, Any], *, base_dir: Path | None = None) -> tuple[list[str], list[PathSource]]:
    expect(manifest.get("schema") == SCHEMA, f"manifest schema must be {SCHEMA}")
    expect(isinstance(manifest.get("version"), str) and manifest["version"].strip(), "manifest.version must be non-empty")
    ape = manifest.get("ape")
    expect(isinstance(ape, dict), "ape must be an object")
    expect(ape.get("kind") == "cosmopolitan_ape", "ape.kind must be cosmopolitan_ape")
    expect(ape.get("target") == "universal-amd64", "ape.target must be universal-amd64")
    expect(ape.get("payload_layout") == "zipaligned-uncompressed", "ape.payload_layout must be zipaligned-uncompressed")
    expect(ape.get("extracts_release_archive") is True, "ape.extracts_release_archive must be true")

    release_archives = manifest.get("release_archives")
    expect(isinstance(release_archives, list) and release_archives, "release_archives must be a non-empty list")
    platforms: set[str] = set()
    sources: list[PathSource] = []
    release_sizes: list[int] = []
    for entry in release_archives:
        platform, source, size = validate_release_archive(entry, base_dir=base_dir)
        expect(platform not in platforms, f"duplicate release archive for {platform}")
        platforms.add(platform)
        if source is not None:
            sources.append(source)
        release_sizes.append(size)
    expect(REQUIRED_PLATFORMS.issubset(platforms), f"release_archives must include {sorted(REQUIRED_PLATFORMS)}")

    model_source, model_size = validate_model(manifest.get("bootstrap_model"), base_dir=base_dir)
    if model_source is not None:
        sources.append(model_source)
    backend_sources = validate_bootstrap_gpu_backends(manifest.get("bootstrap_gpu_backends"), base_dir=base_dir)
    sources.extend(backend_sources)
    backend_sizes = [
        int(backend["size"])
        for backend in manifest["bootstrap_gpu_backends"]
        if isinstance(backend, dict)
    ]
    runtime_sources, runtime_dependency_sizes = validate_bootstrap_runtime_dependencies(
        manifest.get("bootstrap_runtime_dependencies"),
        base_dir=base_dir,
    )
    sources.extend(runtime_sources)
    model_payload_path = normalized_payload_path(
        manifest["bootstrap_model"]["payload_path"],
        label="bootstrap_model.payload_path",
    )

    startup_messages = validate_startup(manifest.get("startup"), model_payload_path=model_payload_path)
    startup = manifest["startup"]
    validate_rocm_bootstrap_command(
        manifest.get("rocm_bootstrap_command"),
        model_payload_path=model_payload_path,
        host=startup["host"],
        port=startup["port"],
    )
    post_setup_messages = validate_post_therock_setup(manifest.get("post_therock_setup"))
    validate_size_budget(manifest, release_sizes, model_size, backend_sizes, runtime_dependency_sizes)
    return [
        "manifest schema ok",
        "universal APE target ok",
        "Windows and Linux release payloads ok",
        "embedded Qwen 0.8B llamafile ok",
        "embedded ROCm llamafile backends ok",
        "embedded ROCm runtime dependencies ok",
        *startup_messages,
        "bootstrap rocm command ok",
        *post_setup_messages,
        "Windows size budget ok",
    ], sources


def sanitized_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    clean = copy.deepcopy(manifest)
    for entry in clean.get("release_archives", []):
        if isinstance(entry, dict):
            entry.pop("source_path", None)
    if isinstance(clean.get("bootstrap_model"), dict):
        clean["bootstrap_model"].pop("source_path", None)
    for entry in clean.get("bootstrap_gpu_backends", []):
        if isinstance(entry, dict):
            entry.pop("source_path", None)
    for entry in clean.get("bootstrap_runtime_dependencies", []):
        if isinstance(entry, dict):
            entry.pop("source_path", None)
    return clean


def validate_zip_payload(path: Path) -> set[str]:
    expect(path.is_file(), f"staged payload does not exist: {path}")
    seen: set[str] = set()
    try:
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                normalized = normalized_payload_path(info.filename, label="staged zip entry")
                expect(normalized not in seen, f"staged zip contains duplicate entry: {normalized}")
                expect(info.compress_type == zipfile.ZIP_STORED, f"staged zip entry must be uncompressed for zipalign: {normalized}")
                seen.add(normalized)
    except zipfile.BadZipFile as error:
        raise ApePackageError(f"staged payload is not a zip archive: {error}") from error
    return seen


def validate_staged_zip(path: Path, manifest: dict[str, Any]) -> list[str]:
    seen = validate_zip_payload(path)
    expected = {DEFAULT_MANIFEST_PATH}
    for entry in manifest["release_archives"]:
        expected.add(normalized_payload_path(entry["payload_path"], label="release payload_path"))
    expected.add(normalized_payload_path(manifest["bootstrap_model"]["payload_path"], label="bootstrap_model.payload_path"))
    for entry in manifest["bootstrap_gpu_backends"]:
        expected.add(normalized_payload_path(entry["payload_path"], label="backend payload_path"))
    for entry in manifest["bootstrap_runtime_dependencies"]:
        expected.add(normalized_payload_path(entry["payload_path"], label="runtime dependency payload_path"))
    missing = sorted(expected - seen)
    expect(not missing, f"staged zip is missing payload entries: {', '.join(missing)}")
    with zipfile.ZipFile(path) as archive:
        embedded_manifest = json.loads(archive.read(DEFAULT_MANIFEST_PATH).decode("utf-8"))
    validate_manifest(embedded_manifest)
    return ["staged uncompressed ZIP payload ok", "embedded manifest ok"]


def write_staged_zip(manifest: dict[str, Any], sources: list[PathSource], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    source_by_payload = {source.payload_path: source for source in sources}
    clean = sanitized_manifest(manifest)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(DEFAULT_MANIFEST_PATH, json.dumps(clean, indent=2, sort_keys=True) + "\n")
        for entry in manifest["release_archives"]:
            payload_path = normalized_payload_path(entry["payload_path"], label="release payload_path")
            source = source_by_payload.get(payload_path)
            expect(source is not None, f"no source_path available for {payload_path}")
            archive.write(source.source_path, payload_path)
        model_path = normalized_payload_path(manifest["bootstrap_model"]["payload_path"], label="bootstrap_model.payload_path")
        source = source_by_payload.get(model_path)
        expect(source is not None, f"no source_path available for {model_path}")
        archive.write(source.source_path, model_path)
        for entry in manifest["bootstrap_gpu_backends"]:
            backend_path = normalized_payload_path(entry["payload_path"], label="backend payload_path")
            source = source_by_payload.get(backend_path)
            expect(source is not None, f"no source_path available for {backend_path}")
            archive.write(source.source_path, backend_path)
        for entry in manifest["bootstrap_runtime_dependencies"]:
            runtime_path = normalized_payload_path(entry["payload_path"], label="runtime dependency payload_path")
            source = source_by_payload.get(runtime_path)
            expect(source is not None, f"no source_path available for {runtime_path}")
            archive.write(source.source_path, runtime_path)


def release_entry(platform: str, path: Path) -> dict[str, Any]:
    return {
        "platform": platform,
        "name": path.name,
        "format": ROCM_RELEASE_FORMAT_BY_PLATFORM[platform],
        "payload_path": f"payload/releases/{path.name}",
        "source_path": str(path),
        "sha256": sha256_file(path),
        "size": path.stat().st_size,
    }


def model_entry(path: Path) -> dict[str, Any]:
    return {
        "kind": "llamafile",
        "name": path.name,
        "family": DEFAULT_MODEL_FAMILY,
        "parameters": DEFAULT_MODEL_PARAMETERS,
        "quantization": "Q8_0",
        "payload_path": f"payload/bootstrap/{path.name}",
        "source_path": str(path),
        "embedded": True,
        "sha256": sha256_file(path),
        "size": path.stat().st_size,
    }


def backend_entry(platform: str, path: Path) -> dict[str, Any]:
    expected_name = REQUIRED_ROCM_BACKENDS[platform]
    expect(path.name == expected_name, f"{platform} backend file must be named {expected_name}")
    return {
        "platform": platform,
        "name": path.name,
        "payload_path": f"payload/bootstrap/{path.name}",
        "source_path": str(path),
        "sha256": sha256_file(path),
        "size": path.stat().st_size,
    }


def runtime_dependency_entry(platform: str, path: Path) -> dict[str, Any]:
    expect(path.name == Path(path.name).name, f"{platform} runtime dependency must be a plain file name")
    return {
        "platform": platform,
        "name": path.name,
        "payload_path": f"payload/bootstrap/{path.name}",
        "source_path": str(path),
        "sha256": sha256_file(path),
        "size": path.stat().st_size,
    }


def build_manifest(
    *,
    version: str,
    windows_release: Path,
    linux_release: Path,
    model: Path,
    windows_rocm_backend: Path,
    linux_rocm_backend: Path,
    windows_runtime_dependency: list[Path],
    linux_runtime_dependency: list[Path],
    port: int,
) -> dict[str, Any]:
    model_payload_path = f"payload/bootstrap/{model.name}"
    host = "127.0.0.1"
    return {
        "schema": SCHEMA,
        "version": version,
        "ape": {
            "kind": "cosmopolitan_ape",
            "target": "universal-amd64",
            "entrypoint": "rocm-cli-ape-launcher",
            "payload_layout": "zipaligned-uncompressed",
            "extracts_release_archive": True,
            "launcher_overhead_estimate": 16 * 1024 * 1024,
            "max_windows_executable_size": DEFAULT_WINDOWS_SIZE_CAP,
        },
        "release_archives": [
            release_entry("windows-amd64", windows_release),
            release_entry("linux-amd64", linux_release),
        ],
        "bootstrap_model": model_entry(model),
        "bootstrap_gpu_backends": [
            backend_entry("windows-amd64", windows_rocm_backend),
            backend_entry("linux-amd64", linux_rocm_backend),
        ],
        "bootstrap_runtime_dependencies": [
            *[runtime_dependency_entry("windows-amd64", path) for path in windows_runtime_dependency],
            *[runtime_dependency_entry("linux-amd64", path) for path in linux_runtime_dependency],
        ],
        "startup": {
            "mode": "server",
            "host": host,
            "port": port,
            "device_policy": "gpu_required",
            "gpu_vendor": "amd",
            "allow_cpu_fallback": False,
            "fallback_policy": "fail_loudly",
            "tool_calling": {
                "openai_compatible": True,
                "jinja": True,
            },
            "gpu_proof_required": sorted(REQUIRED_GPU_PROOFS),
            "args": [
                "--server",
                "--host",
                host,
                "--port",
                str(port),
                "--jinja",
                "--gpu",
                "amd",
                "-ngl",
                "999",
                "-lv",
                "0",
            ],
        },
        "rocm_bootstrap_command": [
            "bootstrap",
            "assistant",
            "--llamafile",
            f"{{extract_root}}/{model_payload_path}",
            "--host",
            host,
            "--port",
            str(port),
            "--device",
            "gpu_required",
        ],
        "post_therock_setup": {
            "self_install_cli": {
                "enabled": True,
                "user_selected_target_required": True,
                "target_placeholder": "{user_selected_cli_install_dir}",
                "install_command": [
                    "bootstrap",
                    "install-cli",
                    "--target",
                    "{user_selected_cli_install_dir}",
                ],
                "path_prompt": {
                    "after_install": True,
                    "copy": "Add this folder to PATH so new terminals can run rocm.",
                    "add_to_path_command": [
                        "bootstrap",
                        "install-cli",
                        "--target",
                        "{user_selected_cli_install_dir}",
                        "--add-to-path",
                    ],
                },
            },
        },
    }


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ApePackageError(f"failed to parse manifest {path}: {error}") from error
    expect(isinstance(value, dict), f"manifest must be a JSON object: {path}")
    return value


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def add_tar_member(package: tarfile.TarFile, name: str, data: bytes, mode: int = 0o644) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = mode
    package.addfile(info, io.BytesIO(data))


def create_fake_linux_release(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as package:
        root = "rocm-cli-test-linux-amd64"
        root_info = tarfile.TarInfo(root)
        root_info.type = tarfile.DIRTYPE
        root_info.mode = 0o755
        package.addfile(root_info)
        add_tar_member(package, f"{root}/bin/rocm", b"fake rocm\n", mode=0o755)
        add_tar_member(package, f"{root}/README.md", b"fake readme\n")


def create_fake_windows_release(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as package:
        package.writestr("rocm-cli-test-windows-amd64/bin/rocm.exe", "fake rocm\n")
        package.writestr("rocm-cli-test-windows-amd64/README.md", "fake readme\n")


def create_fake_model(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake qwen 0.8b llamafile payload\n")


def create_fake_backend(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"fake ROCm backend {path.name}\n".encode("utf-8"))


def create_fake_runtime_dependency(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"fake ROCm runtime dependency {path.name}\n".encode("utf-8"))


def expect_rejected(label: str, func) -> None:
    try:
        func()
    except ApePackageError:
        print(f"APE bootstrap self-test: {label} rejected as expected")
        return
    raise ApePackageError(f"{label} unexpectedly passed")


def run_self_test(root: Path) -> None:
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    try:
        windows_release = root / "inputs" / "rocm-cli-v0.0.0-test-windows-amd64.zip"
        linux_release = root / "inputs" / "rocm-cli-v0.0.0-test-linux-amd64.tar.gz"
        model = root / "inputs" / DEFAULT_MODEL_NAME
        windows_backend = root / "inputs" / REQUIRED_ROCM_BACKENDS["windows-amd64"]
        linux_backend = root / "inputs" / REQUIRED_ROCM_BACKENDS["linux-amd64"]
        windows_runtime = root / "inputs" / "amdhip64_7.dll"
        linux_runtime = root / "inputs" / "libamdhip64.so.7"
        linux_ape_loader = root / "inputs" / "ape-x86_64.elf"
        create_fake_windows_release(windows_release)
        create_fake_linux_release(linux_release)
        create_fake_model(model)
        create_fake_backend(windows_backend)
        create_fake_backend(linux_backend)
        create_fake_runtime_dependency(windows_runtime)
        create_fake_runtime_dependency(linux_runtime)
        create_fake_runtime_dependency(linux_ape_loader)

        manifest = build_manifest(
            version="0.0.0-test",
            windows_release=windows_release,
            linux_release=linux_release,
            model=model,
            windows_rocm_backend=windows_backend,
            linux_rocm_backend=linux_backend,
            windows_runtime_dependency=[windows_runtime],
            linux_runtime_dependency=[linux_runtime, linux_ape_loader],
            port=DEFAULT_PORT,
        )
        messages, sources = validate_manifest(manifest)
        expect("embedded Qwen 0.8B llamafile ok" in messages, "valid manifest did not validate model")
        expect("embedded ROCm llamafile backends ok" in messages, "valid manifest did not validate ROCm backends")
        expect("embedded ROCm runtime dependencies ok" in messages, "valid manifest did not validate ROCm runtime dependencies")
        expect("post-TheRock CLI self-install handoff ok" in messages, "valid manifest did not validate CLI self-install handoff")
        print("APE bootstrap self-test: valid P0 manifest accepted")

        model_exe = root / "inputs" / f"{DEFAULT_MODEL_NAME}.exe"
        create_fake_model(model_exe)
        exe_manifest = build_manifest(
            version="0.0.0-test",
            windows_release=windows_release,
            linux_release=linux_release,
            model=model_exe,
            windows_rocm_backend=windows_backend,
            linux_rocm_backend=linux_backend,
            windows_runtime_dependency=[windows_runtime],
            linux_runtime_dependency=[linux_runtime, linux_ape_loader],
            port=DEFAULT_PORT,
        )
        validate_manifest(exe_manifest)
        print("APE bootstrap self-test: Windows .llamafile.exe model accepted")

        manifest_path = root / "ape-bootstrap.json"
        write_manifest(manifest_path, manifest)
        staged = root / "rocm-cli-ape-bootstrap-fixture.zip"
        write_staged_zip(manifest, sources, staged)
        validate_staged_zip(staged, manifest)
        print("APE bootstrap self-test: staged payload accepted")

        bad_host = copy.deepcopy(manifest)
        bad_host["startup"]["host"] = "0.0.0.0"
        bad_host["startup"]["args"][bad_host["startup"]["args"].index("127.0.0.1")] = "0.0.0.0"
        expect_rejected("public bind", lambda: validate_manifest(bad_host))

        bad_cpu = copy.deepcopy(manifest)
        bad_cpu["startup"]["allow_cpu_fallback"] = True
        expect_rejected("CPU fallback flag", lambda: validate_manifest(bad_cpu))

        bad_ngl = copy.deepcopy(manifest)
        bad_ngl["startup"]["args"][bad_ngl["startup"]["args"].index("999")] = "0"
        expect_rejected("zero GPU layers", lambda: validate_manifest(bad_ngl))

        bad_gpu = copy.deepcopy(manifest)
        gpu_index = bad_gpu["startup"]["args"].index("amd")
        bad_gpu["startup"]["args"][gpu_index] = "disable"
        expect_rejected("disabled GPU", lambda: validate_manifest(bad_gpu))

        bad_model = copy.deepcopy(manifest)
        bad_model["bootstrap_model"]["name"] = "TinyLlama-1.1B-Q8_0.llamafile"
        bad_model["bootstrap_model"]["family"] = "tinyllama"
        bad_model["bootstrap_model"]["parameters"] = "1.1B"
        expect_rejected("non-Qwen model", lambda: validate_manifest(bad_model))

        bad_size = copy.deepcopy(manifest)
        bad_size["ape"]["max_windows_executable_size"] = 1024
        expect_rejected("Windows size cap", lambda: validate_manifest(bad_size))

        bad_platforms = copy.deepcopy(manifest)
        bad_platforms["release_archives"] = [bad_platforms["release_archives"][0]]
        expect_rejected("missing Linux release payload", lambda: validate_manifest(bad_platforms))

        bad_runtime_platform = copy.deepcopy(manifest)
        bad_runtime_platform["bootstrap_runtime_dependencies"] = [
            dep
            for dep in bad_runtime_platform["bootstrap_runtime_dependencies"]
            if dep["platform"] == "windows-amd64"
        ]
        expect_rejected("missing Linux runtime dependency", lambda: validate_manifest(bad_runtime_platform))

        bad_linux_loader = copy.deepcopy(manifest)
        bad_linux_loader["bootstrap_runtime_dependencies"] = [
            dep
            for dep in bad_linux_loader["bootstrap_runtime_dependencies"]
            if dep.get("name") != "ape-x86_64.elf"
        ]
        expect_rejected("missing Linux APE loader", lambda: validate_manifest(bad_linux_loader))

        bad_silent_path = copy.deepcopy(manifest)
        bad_silent_path["post_therock_setup"]["self_install_cli"]["install_command"].append("--add-to-path")
        expect_rejected("silent PATH update", lambda: validate_manifest(bad_silent_path))
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("APE bootstrap self-test: ok")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="Create a manifest from local release/model payloads.")
    plan.add_argument("--version", required=True)
    plan.add_argument("--windows-release", type=Path, required=True)
    plan.add_argument("--linux-release", type=Path, required=True)
    plan.add_argument("--model", type=Path, required=True)
    plan.add_argument("--windows-rocm-backend", type=Path, required=True)
    plan.add_argument("--linux-rocm-backend", type=Path, required=True)
    plan.add_argument("--windows-runtime-dependency", type=Path, action="append", required=True)
    plan.add_argument("--linux-runtime-dependency", type=Path, action="append", required=True)
    plan.add_argument("--port", type=int, default=DEFAULT_PORT)
    plan.add_argument("--output", type=Path, required=True)

    validate = subparsers.add_parser("validate", help="Validate an APE bootstrap manifest.")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--base-dir", type=Path)

    stage = subparsers.add_parser("stage", help="Create an uncompressed payload ZIP for zipalign/APE embedding.")
    stage.add_argument("--manifest", type=Path, required=True)
    stage.add_argument("--output", type=Path, required=True)
    stage.add_argument("--base-dir", type=Path)

    self_test = subparsers.add_parser("self-test", help="Run offline contract self-tests.")
    self_test.add_argument("--root", type=Path, default=default_self_test_root())

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "self-test":
            run_self_test(args.root)
            return 0
        if args.command == "plan":
            manifest = build_manifest(
                version=args.version,
                windows_release=args.windows_release.resolve(),
                linux_release=args.linux_release.resolve(),
                model=args.model.resolve(),
                windows_rocm_backend=args.windows_rocm_backend.resolve(),
                linux_rocm_backend=args.linux_rocm_backend.resolve(),
                windows_runtime_dependency=[path.resolve() for path in args.windows_runtime_dependency],
                linux_runtime_dependency=[path.resolve() for path in args.linux_runtime_dependency],
                port=args.port,
            )
            validate_manifest(manifest)
            write_manifest(args.output, manifest)
            print(f"APE bootstrap package: wrote manifest {args.output}")
            return 0
        manifest = load_manifest(args.manifest)
        base_dir = args.base_dir or args.manifest.parent
        messages, sources = validate_manifest(manifest, base_dir=base_dir)
        if args.command == "validate":
            for message in messages:
                print(f"APE bootstrap package: {message}")
            print("APE bootstrap package: ok")
            return 0
        if args.command == "stage":
            write_staged_zip(manifest, sources, args.output)
            for message in validate_staged_zip(args.output, manifest):
                print(f"APE bootstrap package: {message}")
            print(f"APE bootstrap package: wrote staged payload {args.output}")
            return 0
    except (ApePackageError, OSError) as error:
        fail(str(error))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
