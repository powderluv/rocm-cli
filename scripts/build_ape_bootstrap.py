#!/usr/bin/env python3
"""Build and test the rocm-cli APE bootstrap launcher.

The builder consumes the manifest validated by `ape_bootstrap_package.py`,
expands the platform release archives into an uncompressed ZIP payload, compiles
the C launcher with `cosmocc` or a local C compiler, and appends the payload to
the launcher executable.

Use `--compiler` or `ROCM_CLI_APE_CC` for the compiler. Production should pass
Cosmopolitan's `cosmocc`; the self-test uses a normal POSIX C compiler so the
extract/delegate contract stays testable without a 400+ MB toolchain download.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import posixpath
import shutil
import stat
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import ape_bootstrap_package as ape_contract  # noqa: E402


DEFAULT_LAUNCHER_SOURCE = REPO_ROOT / "launcher" / "ape_bootstrap_launcher.c"
DEFAULT_CC_CANDIDATES = ("cc", "gcc", "clang")
PLATFORM_ROOT = "payload/platform"
MANIFEST_PAYLOAD_PATH = ape_contract.DEFAULT_MANIFEST_PATH
FAKE_ROCM_SOURCE = r"""
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    const char *log = getenv("ROCM_CLI_APE_FAKE_LOG");
    FILE *file;
    int i;
    if (!log || !log[0]) {
        fputs("missing ROCM_CLI_APE_FAKE_LOG\n", stderr);
        return 2;
    }
    file = fopen(log, "ab");
    if (!file) {
        perror("failed to open fake rocm log");
        return 3;
    }
    for (i = 1; i < argc; ++i) {
        fprintf(file, "%s\n", argv[i]);
    }
    if (fclose(file) != 0) {
        perror("failed to close fake rocm log");
        return 4;
    }
    return 0;
}
"""


def default_self_test_root() -> Path:
    return REPO_ROOT / ".rocm-work" / "tests" / f"ape-bootstrap-builder-{os.getpid()}"


class ApeBuildError(Exception):
    """The APE launcher could not be built or tested."""


def fail(message: str) -> None:
    print(f"APE bootstrap builder failed: {message}", file=sys.stderr)
    raise SystemExit(1)


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise ApeBuildError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_zip_name(name: str) -> str:
    normalized = posixpath.normpath(name.replace("\\", "/"))
    expect(normalized not in {"", "."}, f"archive contains empty path: {name!r}")
    expect(not normalized.startswith("/") and not name.startswith("\\"), f"archive contains absolute path: {name}")
    parts = normalized.split("/")
    expect(all(part not in {"", ".", ".."} for part in parts), f"archive contains unsafe path: {name}")
    expect(":" not in parts[0], f"archive contains drive-qualified path: {name}")
    return normalized


def top_level(paths: list[str], archive: Path) -> str:
    values = {path.split("/", 1)[0] for path in paths if path}
    expect(len(values) == 1, f"{archive.name} must contain exactly one top-level folder")
    return next(iter(values))


def zip_info_for_payload(name: str, *, executable: bool = False) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name)
    mode = 0o755 if executable else 0o644
    info.external_attr = (stat.S_IFREG | mode) << 16
    info.compress_type = zipfile.ZIP_STORED
    return info


def write_payload_entry(payload: zipfile.ZipFile, name: str, data: bytes, *, executable: bool = False) -> None:
    safe = ape_contract.normalized_payload_path(name, label="builder payload entry")
    payload.writestr(zip_info_for_payload(safe, executable=executable), data)


def release_entries(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    for entry in manifest["release_archives"]:
        entries[entry["platform"]] = entry
    return entries


def strip_archive_root(path: str, root: str) -> str | None:
    if path == root:
        return None
    prefix = f"{root}/"
    expect(path.startswith(prefix), f"archive path escaped top-level root: {path}")
    relative = path[len(prefix) :]
    return relative or None


def add_zip_release_tree(payload: zipfile.ZipFile, archive: Path, platform: str) -> None:
    with zipfile.ZipFile(archive) as package:
        infos = package.infolist()
        names = [normalized_zip_name(info.filename) for info in infos]
        root = top_level(names, archive)
        for info, safe_name in zip(infos, names):
            relative = strip_archive_root(safe_name, root)
            if relative is None or info.is_dir():
                continue
            target = f"{PLATFORM_ROOT}/{platform}/{relative}"
            mode = (info.external_attr >> 16) & 0o777
            executable = bool(mode & 0o111) or relative.startswith("bin/")
            write_payload_entry(payload, target, package.read(info), executable=executable)


def add_tar_release_tree(payload: zipfile.ZipFile, archive: Path, platform: str) -> None:
    with tarfile.open(archive, "r:gz") as package:
        members = package.getmembers()
        names = [normalized_zip_name(member.name) for member in members]
        root = top_level(names, archive)
        for member, safe_name in zip(members, names):
            relative = strip_archive_root(safe_name, root)
            if relative is None or member.isdir():
                continue
            expect(member.isfile(), f"{archive.name} contains unsupported entry: {member.name}")
            source = package.extractfile(member)
            expect(source is not None, f"failed to read {member.name}")
            target = f"{PLATFORM_ROOT}/{platform}/{relative}"
            executable = bool(stat.S_IMODE(member.mode) & 0o111) or relative.startswith("bin/")
            write_payload_entry(payload, target, source.read(), executable=executable)


def create_expanded_payload(manifest: dict[str, Any], *, base_dir: Path, output: Path) -> None:
    messages, _sources = ape_contract.validate_manifest(manifest, base_dir=base_dir)
    expect("Windows and Linux release payloads ok" in messages, "manifest did not validate release payloads")
    output.parent.mkdir(parents=True, exist_ok=True)
    clean_manifest = ape_contract.sanitized_manifest(manifest)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as payload:
        write_payload_entry(
            payload,
            MANIFEST_PAYLOAD_PATH,
            json.dumps(clean_manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n",
        )
        for entry in manifest["release_archives"]:
            archive = ape_contract.source_path_from(entry.get("source_path"), base_dir=base_dir, label="release source")
            expect(archive is not None and archive.is_file(), f"missing release source for {entry['platform']}")
            write_payload_entry(payload, entry["payload_path"], archive.read_bytes())
            if entry["format"] == "zip":
                add_zip_release_tree(payload, archive, entry["platform"])
            elif entry["format"] == "tar.gz":
                add_tar_release_tree(payload, archive, entry["platform"])
            else:
                raise ApeBuildError(f"unsupported release format: {entry['format']}")
        model = manifest["bootstrap_model"]
        model_path = ape_contract.source_path_from(model.get("source_path"), base_dir=base_dir, label="model source")
        expect(model_path is not None and model_path.is_file(), "missing bootstrap model source")
        write_payload_entry(payload, model["payload_path"], model_path.read_bytes(), executable=True)
        for backend in manifest["bootstrap_gpu_backends"]:
            backend_path = ape_contract.source_path_from(
                backend.get("source_path"),
                base_dir=base_dir,
                label=f"{backend['platform']} backend source",
            )
            expect(backend_path is not None and backend_path.is_file(), f"missing bootstrap GPU backend source for {backend['platform']}")
            write_payload_entry(payload, backend["payload_path"], backend_path.read_bytes(), executable=True)
        for dep in manifest["bootstrap_runtime_dependencies"]:
            dep_path = ape_contract.source_path_from(
                dep.get("source_path"),
                base_dir=base_dir,
                label=f"{dep['platform']} runtime dependency source",
            )
            expect(dep_path is not None and dep_path.is_file(), f"missing bootstrap runtime dependency source for {dep['platform']}: {dep['name']}")
            write_payload_entry(payload, dep["payload_path"], dep_path.read_bytes(), executable=True)
    validate_expanded_payload(output, manifest)


def create_universal_payload(*, windows_release: Path, linux_release: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as payload:
        add_zip_release_tree(payload, windows_release, "windows-amd64")
        add_tar_release_tree(payload, linux_release, "linux-amd64")
    validate_universal_payload(output)


def validate_universal_payload(payload: Path) -> None:
    seen = ape_contract.validate_zip_payload(payload)
    expected = {
        f"{PLATFORM_ROOT}/windows-amd64/bin/rocm.exe",
        f"{PLATFORM_ROOT}/windows-amd64/bin/rocmd.exe",
        f"{PLATFORM_ROOT}/linux-amd64/bin/rocm",
        f"{PLATFORM_ROOT}/linux-amd64/bin/rocmd",
    }
    missing = sorted(expected - set(seen))
    expect(not missing, f"universal payload is missing required entries: {missing}")


def validate_expanded_payload(payload: Path, manifest: dict[str, Any]) -> list[str]:
    seen = ape_contract.validate_zip_payload(payload)
    expected = {MANIFEST_PAYLOAD_PATH, manifest["bootstrap_model"]["payload_path"]}
    expected.update(entry["payload_path"] for entry in manifest["bootstrap_gpu_backends"])
    expected.update(entry["payload_path"] for entry in manifest["bootstrap_runtime_dependencies"])
    for entry in manifest["release_archives"]:
        expected.add(entry["payload_path"])
        platform = entry["platform"]
        binary = "rocm.exe" if platform == "windows-amd64" else "rocm"
        expected.add(f"{PLATFORM_ROOT}/{platform}/bin/{binary}")
    missing = sorted(expected - seen)
    expect(not missing, f"expanded payload is missing required entries: {', '.join(missing)}")
    return ["expanded platform payload ok"]


def compiler_from_args(value: str | None) -> str:
    if value:
        return value
    configured = os.environ.get("ROCM_CLI_APE_CC")
    if configured:
        return configured
    if os.name == "nt":
        for candidate in windows_rocm_clang_candidates():
            if candidate.is_file():
                return str(candidate)
    for candidate in DEFAULT_CC_CANDIDATES:
        found = shutil.which(candidate)
        if found:
            return found
    raise ApeBuildError("no C compiler found; set --compiler or ROCM_CLI_APE_CC")


def windows_rocm_clang_candidates() -> list[Path]:
    roots: list[Path] = []
    for env_name in ("ROCM_CLI_THEROCK_VENV", "VIRTUAL_ENV"):
        value = os.environ.get(env_name)
        if value:
            roots.append(Path(value))
    roots.append(Path("D:/jam/venv"))
    candidates: list[Path] = []
    for root in roots:
        candidates.append(root / "Lib" / "site-packages" / "_rocm_sdk_core" / "lib" / "llvm" / "bin" / "clang.exe")
        candidates.append(root / "Lib" / "site-packages" / "_rocm_sdk_devel" / "lib" / "llvm" / "bin" / "clang.exe")
    configured = os.environ.get("ROCM_CLI_THEROCK_CLANG")
    if configured:
        candidates.insert(0, Path(configured))
    return candidates


def is_cosmopolitan_compiler(compiler: str) -> bool:
    name = Path(compiler).name.lower()
    return "cosmo" in name or "ape" in name


def validate_build_compiler(compiler: str, *, allow_local_compiler: bool) -> list[str]:
    if is_cosmopolitan_compiler(compiler):
        return ["compiler looked like cosmocc/APE"]
    if allow_local_compiler:
        return ["compiler was not cosmocc; output is for local launcher testing only"]
    raise ApeBuildError(
        "production APE builds require Cosmopolitan's cosmocc; "
        "pass --allow-local-compiler only for local launcher extraction tests"
    )


def is_wsl() -> bool:
    if os.name != "posix":
        return False
    if os.environ.get("WSL_INTEROP") or os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        return "microsoft" in Path("/proc/sys/kernel/osrelease").read_text(encoding="utf-8").lower()
    except OSError:
        return False


def ape_loader_for_compiler(compiler: str) -> Path | None:
    compiler_path = Path(compiler)
    for candidate in ape_loader_candidates(compiler_path):
        if candidate.is_file():
            return candidate
    return None


def ape_loader_candidates(compiler_path: Path) -> list[Path]:
    return [
        compiler_path.parent / "ape-x86_64.elf",
        compiler_path.parent.parent / "bin" / "ape-x86_64.elf",
    ]


def invocation_prefix_for_output(compiler: str) -> list[str]:
    if os.name == "nt" or not is_cosmopolitan_compiler(compiler):
        return []
    loader = ape_loader_for_compiler(compiler)
    if loader is None and is_wsl():
        candidates = ", ".join(str(path) for path in ape_loader_candidates(Path(compiler)))
        raise ApeBuildError(
            f"WSL APE execution requires ape-x86_64.elf beside cosmocc; checked: {candidates}"
        )
    if loader is None:
        return []
    return [str(loader)]


def compile_launcher(
    *,
    compiler: str,
    source: Path,
    output: Path,
    manifest: dict[str, Any],
    extra_flags: list[str],
) -> None:
    model_payload = manifest["bootstrap_model"]["payload_path"]
    port = str(manifest["startup"]["port"])
    args = [
        compiler,
        "-O2",
        "-D_CRT_SECURE_NO_WARNINGS",
        f"-DROCM_CLI_APE_VERSION=\"{manifest['version']}\"",
        f"-DROCM_CLI_APE_MODEL_PAYLOAD=\"{model_payload}\"",
        f"-DROCM_CLI_APE_BOOTSTRAP_PORT=\"{port}\"",
        *extra_flags,
        "-o",
        str(output),
        str(source),
    ]
    completed = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0:
        raise ApeBuildError(f"compiler failed with exit {completed.returncode}:\n{completed.stdout}")
    if os.name != "nt":
        output.chmod(output.stat().st_mode | stat.S_IXUSR)


def compile_universal_launcher(
    *,
    compiler: str,
    source: Path,
    output: Path,
    version: str,
    extra_flags: list[str],
) -> None:
    args = [
        compiler,
        "-O2",
        "-D_CRT_SECURE_NO_WARNINGS",
        f"-DROCM_CLI_APE_VERSION=\"{version}\"",
        *extra_flags,
        "-o",
        str(output),
        str(source),
    ]
    completed = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0:
        raise ApeBuildError(f"compiler failed with exit {completed.returncode}:\n{completed.stdout}")
    if os.name != "nt":
        output.chmod(output.stat().st_mode | stat.S_IXUSR)


def compile_fake_rocm(*, compiler: str, source: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    args = [compiler, "-O2", "-D_CRT_SECURE_NO_WARNINGS", "-o", str(output), str(source)]
    completed = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0:
        raise ApeBuildError(f"fake rocm compiler failed with exit {completed.returncode}:\n{completed.stdout}")
    if os.name != "nt":
        output.chmod(output.stat().st_mode | stat.S_IXUSR)


def append_payload(*, launcher: Path, payload: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        handle.write(launcher.read_bytes())
        handle.write(payload.read_bytes())
    if os.name != "nt":
        output.chmod(output.stat().st_mode | stat.S_IXUSR)


def extensionless_wsl_alias_path(output: Path) -> Path | None:
    if not is_wsl() or output.suffix.lower() != ".exe":
        return None
    return output.with_suffix("")


def create_extensionless_wsl_alias(output: Path) -> Path | None:
    alias = extensionless_wsl_alias_path(output)
    if alias is None:
        return None
    if alias.exists() or alias.is_symlink():
        if alias.is_dir():
            raise ApeBuildError(f"cannot replace WSL extensionless alias directory: {alias}")
        alias.unlink()
    try:
        os.link(output, alias)
    except OSError:
        shutil.copy2(output, alias)
    alias.chmod(alias.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return alias


def build_launcher(
    *,
    manifest_path: Path,
    output: Path,
    compiler: str,
    source: Path,
    base_dir: Path | None,
    work_dir: Path,
    extra_flags: list[str],
) -> Path:
    manifest = ape_contract.load_manifest(manifest_path)
    base = base_dir or manifest_path.parent
    ape_contract.validate_manifest(manifest, base_dir=base)
    work_dir.mkdir(parents=True, exist_ok=True)
    payload = work_dir / "ape-bootstrap-payload-expanded.zip"
    compiled = work_dir / ("ape-bootstrap-launcher-compiled.exe" if os.name == "nt" else "ape-bootstrap-launcher-compiled")
    create_expanded_payload(manifest, base_dir=base, output=payload)
    compile_launcher(
        compiler=compiler,
        source=source,
        output=compiled,
        manifest=manifest,
        extra_flags=extra_flags,
    )
    append_payload(launcher=compiled, payload=payload, output=output)
    create_extensionless_wsl_alias(output)
    return output


def build_universal_launcher(
    *,
    windows_release: Path,
    linux_release: Path,
    output: Path,
    compiler: str,
    source: Path,
    work_dir: Path,
    version: str,
    extra_flags: list[str],
) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    payload = work_dir / "rocm-universal-payload-expanded.zip"
    compiled = work_dir / ("rocm-universal-launcher-compiled.exe" if os.name == "nt" else "rocm-universal-launcher-compiled")
    create_universal_payload(
        windows_release=windows_release,
        linux_release=linux_release,
        output=payload,
    )
    compile_universal_launcher(
        compiler=compiler,
        source=source,
        output=compiled,
        version=version,
        extra_flags=extra_flags,
    )
    append_payload(launcher=compiled, payload=payload, output=output)
    create_extensionless_wsl_alias(output)
    return output


def create_fake_windows_release(path: Path, fake_rocm: Path | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rocm_bytes = fake_rocm.read_bytes() if fake_rocm is not None else b"fake windows rocm\n"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as package:
        package.writestr(zip_info_for_payload("rocm-cli-test-windows-amd64/bin/rocm.exe", executable=True), rocm_bytes)
        package.writestr(zip_info_for_payload("rocm-cli-test-windows-amd64/bin/rocmd.exe", executable=True), rocm_bytes)
        package.writestr("rocm-cli-test-windows-amd64/README.md", "fake readme\n")


def add_tar_member(package: tarfile.TarFile, name: str, data: bytes, mode: int = 0o644) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = mode
    package.addfile(info, io.BytesIO(data))


def create_fake_linux_release(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    script = b"""#!/bin/sh
set -eu
log="${ROCM_CLI_APE_FAKE_LOG:?missing ROCM_CLI_APE_FAKE_LOG}"
if [ "$#" -gt 0 ]; then
  printf '%s\n' "$@" >> "$log"
else
  : > "$log"
fi
exit 0
"""
    with tarfile.open(path, "w:gz") as package:
        root = "rocm-cli-test-linux-amd64"
        root_info = tarfile.TarInfo(root)
        root_info.type = tarfile.DIRTYPE
        root_info.mode = 0o755
        package.addfile(root_info)
        add_tar_member(package, f"{root}/bin/rocm", script, mode=0o755)
        add_tar_member(package, f"{root}/bin/rocmd", script, mode=0o755)
        add_tar_member(package, f"{root}/README.md", b"fake readme\n")


def create_fake_model(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake qwen 0.8b llamafile payload\n")


def create_fake_backend(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"fake ROCm backend {path.name}\n".encode("utf-8"))


def create_fake_runtime_dependency(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"fake ROCm runtime dependency {path.name}\n".encode("utf-8"))


def run_self_test(
    root: Path,
    compiler: str | None,
    *,
    keep: bool = False,
    windows_fake_rocm: Path | None = None,
) -> None:
    root = root.resolve()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    try:
        cc = compiler_from_args(compiler)
        inputs = root / "inputs"
        tools = root / "tools"
        windows_release = inputs / "rocm-cli-v0.0.0-test-windows-amd64.zip"
        linux_release = inputs / "rocm-cli-v0.0.0-test-linux-amd64.tar.gz"
        model = inputs / ape_contract.DEFAULT_MODEL_NAME
        windows_backend = inputs / ape_contract.REQUIRED_ROCM_BACKENDS["windows-amd64"]
        linux_backend = inputs / ape_contract.REQUIRED_ROCM_BACKENDS["linux-amd64"]
        windows_runtime = inputs / "amdhip64_7.dll"
        linux_runtime = inputs / "libamdhip64.so.7"
        linux_ape_loader = inputs / "ape-x86_64.elf"
        fake_rocm_exe: Path | None = windows_fake_rocm
        if fake_rocm_exe is not None:
            fake_rocm_exe = fake_rocm_exe.resolve()
            expect(fake_rocm_exe.is_file(), f"provided Windows fake rocm does not exist: {fake_rocm_exe}")
        elif os.name == "nt":
            tools.mkdir(parents=True, exist_ok=True)
            fake_rocm_source = tools / "fake_rocm.c"
            fake_rocm_source.write_text(FAKE_ROCM_SOURCE, encoding="utf-8")
            fake_rocm_exe = tools / "fake_rocm.exe"
            compile_fake_rocm(compiler=cc, source=fake_rocm_source, output=fake_rocm_exe)
        create_fake_windows_release(windows_release, fake_rocm=fake_rocm_exe)
        create_fake_linux_release(linux_release)
        create_fake_model(model)
        create_fake_backend(windows_backend)
        create_fake_backend(linux_backend)
        create_fake_runtime_dependency(windows_runtime)
        create_fake_runtime_dependency(linux_runtime)
        create_fake_runtime_dependency(linux_ape_loader)
        manifest = ape_contract.build_manifest(
            version="0.0.0-test",
            windows_release=windows_release,
            linux_release=linux_release,
            model=model,
            windows_rocm_backend=windows_backend,
            linux_rocm_backend=linux_backend,
            windows_runtime_dependency=[windows_runtime],
            linux_runtime_dependency=[linux_runtime, linux_ape_loader],
            port=ape_contract.DEFAULT_PORT,
        )
        manifest_path = root / "ape-bootstrap.json"
        ape_contract.write_manifest(manifest_path, manifest)

        output = root / ("rocm-bootstrap-ape-test.exe" if os.name == "nt" or is_cosmopolitan_compiler(cc) else "rocm-bootstrap-ape-test")
        build_launcher(
            manifest_path=manifest_path,
            output=output,
            compiler=cc,
            source=DEFAULT_LAUNCHER_SOURCE,
            base_dir=None,
            work_dir=root / "build",
            extra_flags=[],
        )
        print(f"APE bootstrap builder self-test: built launcher with {Path(cc).name}")

        extract_root = root / "extract-root"
        fake_log = root / "fake-rocm-argv.log"
        env = os.environ.copy()
        env["ROCM_CLI_APE_ROOT"] = str(extract_root)
        env["ROCM_CLI_APE_FAKE_LOG"] = str(fake_log)
        run_prefix = invocation_prefix_for_output(cc)
        completed = subprocess.run(
            [*run_prefix, str(output), "--", "version", "--json"],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        expect(completed.returncode == 0, f"delegated rocm args failed:\n{completed.stdout}")
        expect(
            fake_log.read_text(encoding="utf-8").splitlines() == ["version", "--json"],
            "delegated rocm argv did not match",
        )
        expect(
            (extract_root / "payload" / "platform" / ("windows-amd64" if os.name == "nt" else "linux-amd64") / "bin" / ("rocm.exe" if os.name == "nt" else "rocm")).is_file(),
            "linux rocm binary was not extracted",
        )
        print("APE bootstrap builder self-test: delegated argv accepted")

        alias = extensionless_wsl_alias_path(output)
        if alias is not None:
            expect(alias.is_file(), f"WSL extensionless alias was not created: {alias}")
            fake_log.unlink()
            completed = subprocess.run(
                [*run_prefix, str(alias), "--", "version", "--json"],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            expect(completed.returncode == 0, f"extensionless WSL alias failed:\n{completed.stdout}")
            expect(
                fake_log.read_text(encoding="utf-8").splitlines() == ["version", "--json"],
                "extensionless WSL alias did not delegate argv correctly",
            )
            print("APE bootstrap builder self-test: extensionless WSL alias accepted")

        fake_log.unlink()
        completed = subprocess.run(
            [*run_prefix, str(output)],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        expect(completed.returncode == 0, f"default bootstrap delegation failed:\n{completed.stdout}")
        expect(
            "ROCm CLI Setup Assistant" in completed.stdout,
            f"default bootstrap did not show startup UI:\n{completed.stdout}",
        )
        expect(
            "Starting ROCm CLI setup." in completed.stdout,
            f"default launch did not show setup startup status:\n{completed.stdout}",
        )
        lines = fake_log.read_text(encoding="utf-8").splitlines()
        expect(lines == [], f"default launch should delegate to rocm with no extra args, got: {lines}")
        print("APE bootstrap builder self-test: default rocm launch accepted")

        universal_output = root / ("rocm-universal-ape-test.exe" if os.name == "nt" or is_cosmopolitan_compiler(cc) else "rocm-universal-ape-test")
        build_universal_launcher(
            windows_release=windows_release,
            linux_release=linux_release,
            output=universal_output,
            compiler=cc,
            source=DEFAULT_LAUNCHER_SOURCE,
            work_dir=root / "build-universal",
            version="0.0.0-test-universal",
            extra_flags=[],
        )
        fake_log.unlink()
        env["ROCM_CLI_APE_ROOT"] = str(root / "extract-universal-root")
        completed = subprocess.run(
            [*run_prefix, str(universal_output), "--", "version", "--json"],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        expect(completed.returncode == 0, f"universal delegated rocm args failed:\n{completed.stdout}")
        expect(
            fake_log.read_text(encoding="utf-8").splitlines() == ["version", "--json"],
            "universal delegated rocm argv did not match",
        )
        expect(
            (Path(env["ROCM_CLI_APE_ROOT"]) / "payload" / "platform" / ("windows-amd64" if os.name == "nt" else "linux-amd64") / "bin" / ("rocm.exe" if os.name == "nt" else "rocm")).is_file(),
            "universal platform rocm binary was not extracted",
        )
        print("APE bootstrap builder self-test: universal launcher delegated argv accepted")

        env["ROCM_CLI_APE_ROOT"] = str(extract_root)
        completed = subprocess.run(
            [*run_prefix, str(output), "--ape-extract-only"],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        expect(completed.returncode == 0, f"extract-only failed:\n{completed.stdout}")
        expect(completed.stdout.strip() == str(extract_root), "extract-only did not print extraction root")
        print("APE bootstrap builder self-test: extract-only accepted")

        if is_cosmopolitan_compiler(cc):
            print("APE bootstrap builder self-test: compiler looked like cosmocc/APE")
        else:
            try:
                validate_build_compiler(cc, allow_local_compiler=False)
            except ApeBuildError:
                print("APE bootstrap builder self-test: production non-cosmocc rejection accepted")
            else:
                raise ApeBuildError("production compiler validation unexpectedly accepted local compiler")
    finally:
        if keep:
            print(f"APE bootstrap builder self-test: kept fixture root {root}")
        else:
            shutil.rmtree(root, ignore_errors=True)
    print("APE bootstrap builder self-test: ok")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build a launcher and append the expanded payload ZIP.")
    build.add_argument("--manifest", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--compiler", help="C compiler path. Use cosmocc for production APE builds.")
    build.add_argument("--source", type=Path, default=DEFAULT_LAUNCHER_SOURCE)
    build.add_argument("--base-dir", type=Path)
    build.add_argument("--work-dir", type=Path, default=REPO_ROOT / ".rocm-work" / "ape-builder")
    build.add_argument("--cflag", action="append", default=[], help="Extra C compiler flag. Repeat as needed.")
    build.add_argument(
        "--allow-local-compiler",
        action="store_true",
        help="Allow a non-cosmocc compiler for local launcher extraction tests only.",
    )

    universal = subparsers.add_parser("build-universal", help="Build the current universal rocm launcher without bootstrap model payloads.")
    universal.add_argument("--windows-release", type=Path, required=True)
    universal.add_argument("--linux-release", type=Path, required=True)
    universal.add_argument("--output", type=Path, required=True)
    universal.add_argument("--compiler", help="C compiler path. Use cosmocc for production APE builds.")
    universal.add_argument("--source", type=Path, default=DEFAULT_LAUNCHER_SOURCE)
    universal.add_argument("--work-dir", type=Path, default=REPO_ROOT / ".rocm-work" / "ape-builder")
    universal.add_argument("--version", default="0.2.0-universal")
    universal.add_argument("--cflag", action="append", default=[], help="Extra C compiler flag. Repeat as needed.")
    universal.add_argument(
        "--allow-local-compiler",
        action="store_true",
        help="Allow a non-cosmocc compiler for local launcher extraction tests only.",
    )

    stage = subparsers.add_parser("stage-expanded", help="Stage the expanded uncompressed payload ZIP only.")
    stage.add_argument("--manifest", type=Path, required=True)
    stage.add_argument("--output", type=Path, required=True)
    stage.add_argument("--base-dir", type=Path)

    self_test = subparsers.add_parser("self-test", help="Run offline builder self-tests.")
    self_test.add_argument("--root", type=Path, default=default_self_test_root())
    self_test.add_argument("--compiler", help="C compiler path for the self-test.")
    self_test.add_argument("--keep", action="store_true", help="Keep the generated fixture root for cross-platform smoke tests.")
    self_test.add_argument("--windows-fake-rocm", type=Path, help="Optional Windows rocm.exe fixture to embed in the Windows release archive.")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "self-test":
            run_self_test(args.root, args.compiler, keep=args.keep, windows_fake_rocm=args.windows_fake_rocm)
            return 0
        if args.command == "stage-expanded":
            manifest = ape_contract.load_manifest(args.manifest)
            create_expanded_payload(
                manifest,
                base_dir=args.base_dir or args.manifest.parent,
                output=args.output,
            )
            for message in validate_expanded_payload(args.output, manifest):
                print(f"APE bootstrap builder: {message}")
            print(f"APE bootstrap builder: wrote expanded payload {args.output}")
            return 0
        compiler = compiler_from_args(args.compiler)
        compiler_messages = validate_build_compiler(
            compiler,
            allow_local_compiler=args.allow_local_compiler,
        )
        if args.command == "build-universal":
            output = build_universal_launcher(
                windows_release=args.windows_release.resolve(),
                linux_release=args.linux_release.resolve(),
                output=args.output.resolve(),
                compiler=compiler,
                source=args.source.resolve(),
                work_dir=args.work_dir.resolve(),
                version=args.version,
                extra_flags=args.cflag,
            )
            print(f"APE bootstrap builder: wrote universal launcher {output}")
            alias = extensionless_wsl_alias_path(output)
            if alias is not None and alias.exists():
                print(f"APE bootstrap builder: wrote WSL alias {alias}")
            print(f"APE bootstrap builder: sha256 {sha256_file(output)}")
            for message in compiler_messages:
                print(f"APE bootstrap builder: {message}")
            return 0
        output = build_launcher(
            manifest_path=args.manifest,
            output=args.output,
            compiler=compiler,
            source=args.source,
            base_dir=args.base_dir,
            work_dir=args.work_dir,
            extra_flags=args.cflag,
        )
        print(f"APE bootstrap builder: wrote launcher {output}")
        alias = extensionless_wsl_alias_path(output)
        if alias is not None and alias.exists():
            print(f"APE bootstrap builder: wrote WSL alias {alias}")
        print(f"APE bootstrap builder: sha256 {sha256_file(output)}")
        for message in compiler_messages:
            print(f"APE bootstrap builder: {message}")
        return 0
    except (ApeBuildError, ape_contract.ApePackageError, OSError) as error:
        fail(str(error))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
