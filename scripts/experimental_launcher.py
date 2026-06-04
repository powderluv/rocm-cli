#!/usr/bin/env python3
"""Experimental self-extracting rocm-cli launcher spike.

This is not a production release path. It is a small, testable packaging spike
for Idea 2 in docs/future-bootstrap-packaging.md:

- accept an existing rocm-cli release archive, or an archive embedded into a
  generated copy of this script;
- verify the archive SHA-256 before extraction;
- extract into .rocm/launcher/<version>/<platform>/;
- reuse an already extracted matching archive;
- delegate argv directly to the extracted bin/rocm executable.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import platform
import posixpath
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence


SCHEMA = "rocm-cli-launcher-spike/v1"
PAYLOAD_BEGIN = "# __ROCM_CLI_LAUNCHER_SPIKE_PAYLOAD_V1__"
PAYLOAD_END = "# __ROCM_CLI_LAUNCHER_SPIKE_PAYLOAD_END__"
ACTIVATION_MANIFEST = ".rocm-launcher-spike.json"
DEFAULT_SELF_TEST_ROOT = Path("target") / "launcher-spike-self-test"
SUPPORTED_FORMATS = ("tar.gz", "zip")


class LauncherError(Exception):
    """The experimental launcher could not safely continue."""


@dataclass(frozen=True)
class ArchivePayload:
    manifest: dict
    archive_bytes: bytes
    archive_name: str


@dataclass(frozen=True)
class LaunchResult:
    target_dir: Path
    rocm_path: Path
    reused: bool
    exit_code: int


Runner = Callable[[Sequence[str]], object]


def fail(message: str) -> None:
    print(f"rocm-cli launcher spike: {message}", file=sys.stderr)
    raise SystemExit(1)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize_sha256(value: str) -> str:
    digest = value.strip().lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise LauncherError(f"invalid sha256 digest in launcher manifest: {value!r}")
    return digest


def parse_sha256_sidecar(path: Path) -> str:
    sidecar = Path(f"{path}.sha256")
    if not sidecar.is_file():
        raise LauncherError(f"missing checksum sidecar: {sidecar}")
    lines = [line.strip() for line in sidecar.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise LauncherError(f"checksum sidecar is empty: {sidecar}")
    return normalize_sha256(lines[0].split()[0])


def host_platform_id() -> str:
    if sys.platform.startswith("win"):
        os_name = "windows"
    elif sys.platform.startswith("linux"):
        os_name = "linux"
    else:
        raise LauncherError(f"unsupported launcher host OS for spike: {sys.platform}")

    machine = platform.machine().lower()
    if machine in {"x86_64", "amd64"}:
        arch = "amd64"
    else:
        raise LauncherError(f"unsupported launcher host architecture for spike: {machine}")
    return f"{os_name}-{arch}"


def default_rocm_dir() -> Path:
    configured = os.environ.get("ROCM_CLI_LAUNCHER_ROCM_DIR") or os.environ.get("ROCM_CLI_CONFIG_DIR")
    if configured:
        return Path(configured).expanduser()

    home = os.environ.get("USERPROFILE") or os.environ.get("HOME")
    if not home:
        raise LauncherError("unable to determine home directory for .rocm launcher cache")
    return Path(home).expanduser() / ".rocm"


def archive_format(name: str, manifest: dict) -> str:
    configured = manifest.get("archive", {}).get("format")
    if configured:
        if configured not in SUPPORTED_FORMATS:
            raise LauncherError(f"unsupported archive format in launcher manifest: {configured}")
        return configured

    lower_name = name.lower()
    if lower_name.endswith(".tar.gz") or lower_name.endswith(".tgz"):
        return "tar.gz"
    if lower_name.endswith(".zip"):
        return "zip"
    raise LauncherError(f"unsupported release archive extension: {name}")


def manifest_archive(manifest: dict) -> dict:
    archive = manifest.get("archive")
    if not isinstance(archive, dict):
        raise LauncherError("launcher manifest must contain an archive object")
    return archive


def manifest_version(manifest: dict) -> str:
    version = manifest.get("version")
    if not isinstance(version, str) or not version.strip():
        raise LauncherError("launcher manifest must contain a non-empty version")
    return version.strip()


def manifest_platform(manifest: dict) -> str:
    value = manifest.get("platform")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return host_platform_id()


def expected_archive_sha256(manifest: dict, archive_path: Path | None) -> str:
    archive = manifest_archive(manifest)
    digest = archive.get("sha256")
    if isinstance(digest, str) and digest.strip():
        return normalize_sha256(digest)
    if archive_path is not None:
        return parse_sha256_sidecar(archive_path)
    raise LauncherError("launcher manifest must include archive.sha256 for embedded archives")


def load_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise LauncherError(f"failed to parse JSON manifest {path}: {error}") from error
    if not isinstance(data, dict):
        raise LauncherError(f"JSON manifest must be an object: {path}")
    return data


def load_external_payload(args: argparse.Namespace) -> ArchivePayload:
    manifest_path = Path(args.manifest).resolve() if args.manifest else None
    manifest = load_json(manifest_path) if manifest_path else {"schema": SCHEMA, "archive": {}}
    archive = manifest.setdefault("archive", {})
    if not isinstance(archive, dict):
        raise LauncherError("launcher manifest archive field must be an object")

    if args.version:
        manifest["version"] = args.version
    if args.platform:
        manifest["platform"] = args.platform
    if args.sha256:
        archive["sha256"] = args.sha256

    archive_path: Path | None = None
    if args.archive:
        archive_path = Path(args.archive).resolve()
    elif isinstance(archive.get("path"), str) and archive["path"].strip():
        base = manifest_path.parent if manifest_path else Path.cwd()
        archive_path = (base / archive["path"]).resolve()

    if archive_path is None:
        raise LauncherError("external launcher mode requires --archive or manifest archive.path")
    if not archive_path.is_file():
        raise LauncherError(f"release archive not found: {archive_path}")

    archive_bytes = archive_path.read_bytes()
    archive.setdefault("name", archive_path.name)
    archive.setdefault("format", archive_format(archive_path.name, manifest))
    archive["sha256"] = expected_archive_sha256(manifest, archive_path)

    return ArchivePayload(
        manifest=manifest,
        archive_bytes=archive_bytes,
        archive_name=str(archive.get("name") or archive_path.name),
    )


def strip_existing_embedded_payload(source: bytes) -> bytes:
    begin = PAYLOAD_BEGIN.encode("ascii")
    lines = source.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.rstrip(b"\r\n") == begin:
            return b"".join(lines[:index]).rstrip() + b"\n"
    return source.rstrip() + b"\n"


def embedded_payload_lines(script_path: Path) -> list[bytes] | None:
    lines = script_path.read_bytes().splitlines()
    begin = PAYLOAD_BEGIN.encode("ascii")
    end = PAYLOAD_END.encode("ascii")
    for index, line in enumerate(lines):
        if line == begin:
            payload_lines: list[bytes] = []
            for payload_line in lines[index + 1 :]:
                if payload_line == end:
                    return payload_lines
                if not payload_line.startswith(b"# "):
                    raise LauncherError(f"malformed embedded launcher payload in {script_path}")
                payload_lines.append(payload_line[2:])
            raise LauncherError(f"unterminated embedded launcher payload in {script_path}")
    return None


def load_embedded_payload(script_path: Path) -> ArchivePayload:
    payload_lines = embedded_payload_lines(script_path)
    if payload_lines is None:
        raise LauncherError(
            "no embedded payload found; pass --archive/--manifest or build a generated launcher"
        )
    encoded = b"".join(payload_lines)
    try:
        payload = json.loads(base64.b64decode(encoded).decode("utf-8"))
    except (ValueError, json.JSONDecodeError) as error:
        raise LauncherError(f"failed to decode embedded launcher payload: {error}") from error
    if not isinstance(payload, dict):
        raise LauncherError("embedded launcher payload must decode to an object")

    manifest = payload.get("manifest")
    archive_b64 = payload.get("archive_b64")
    if not isinstance(manifest, dict) or not isinstance(archive_b64, str):
        raise LauncherError("embedded launcher payload must include manifest and archive_b64")
    try:
        archive_bytes = base64.b64decode(archive_b64)
    except ValueError as error:
        raise LauncherError(f"failed to decode embedded archive bytes: {error}") from error

    archive = manifest_archive(manifest)
    archive_name = str(archive.get("name") or "embedded-release-archive")
    return ArchivePayload(manifest=manifest, archive_bytes=archive_bytes, archive_name=archive_name)


def payload_from_args(args: argparse.Namespace, script_path: Path) -> ArchivePayload:
    if args.archive or args.manifest:
        return load_external_payload(args)
    return load_embedded_payload(script_path)


def normalized_archive_name(name: str) -> str:
    normalized = posixpath.normpath(name.replace("\\", "/"))
    if normalized in {"", "."}:
        raise LauncherError(f"archive contains an empty path entry: {name!r}")
    if normalized.startswith("/") or name.startswith("\\"):
        raise LauncherError(f"archive contains an absolute path entry: {name!r}")
    parts = normalized.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise LauncherError(f"archive contains an unsafe path entry: {name!r}")
    if ":" in parts[0]:
        raise LauncherError(f"archive contains a drive-qualified path entry: {name!r}")
    return normalized


def top_level_name(paths: Iterable[str]) -> str:
    top_levels = {path.split("/", 1)[0] for path in paths}
    if len(top_levels) != 1:
        joined = ", ".join(sorted(top_levels))
        raise LauncherError(f"release archive must contain exactly one top-level directory: {joined}")
    return next(iter(top_levels))


def stripped_bundle_path(path: str, top_level: str) -> str | None:
    if path == top_level:
        return None
    prefix = f"{top_level}/"
    if not path.startswith(prefix):
        raise LauncherError(f"archive path escaped top-level directory: {path}")
    stripped = path[len(prefix) :]
    return stripped or None


def write_file(path: Path, data: bytes, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    if mode is not None:
        path.chmod(mode)


def extract_tar_gz(archive_bytes: bytes, destination: Path) -> None:
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as package:
        members = package.getmembers()
        safe_names = [normalized_archive_name(member.name) for member in members]
        top = top_level_name(safe_names)
        for member, safe_name in zip(members, safe_names):
            if member.issym() or member.islnk():
                raise LauncherError(f"archive contains unsupported link entry: {member.name}")
            relative = stripped_bundle_path(safe_name, top)
            if relative is None:
                continue
            target = destination / Path(*relative.split("/"))
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                extracted = package.extractfile(member)
                if extracted is None:
                    raise LauncherError(f"failed to read archive member: {member.name}")
                mode = stat.S_IMODE(member.mode) or None
                write_file(target, extracted.read(), mode=mode)
            else:
                raise LauncherError(f"archive contains unsupported entry: {member.name}")


def zip_entry_is_symlink(info: zipfile.ZipInfo) -> bool:
    return ((info.external_attr >> 16) & 0o170000) == stat.S_IFLNK


def extract_zip(archive_bytes: bytes, destination: Path) -> None:
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as package:
        infos = package.infolist()
        if any(zip_entry_is_symlink(info) for info in infos):
            raise LauncherError("zip archive contains unsupported symlink entries")
        safe_names = [normalized_archive_name(info.filename) for info in infos]
        top = top_level_name(safe_names)
        for info, safe_name in zip(infos, safe_names):
            relative = stripped_bundle_path(safe_name, top)
            if relative is None:
                continue
            target = destination / Path(*relative.split("/"))
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            mode = stat.S_IMODE(info.external_attr >> 16) or None
            write_file(target, package.read(info), mode=mode)


def rocm_binary_name(platform_id: str) -> str:
    return "rocm.exe" if platform_id.startswith("windows-") else "rocm"


def rocm_binary_path(target_dir: Path, platform_id: str) -> Path:
    return target_dir / "bin" / rocm_binary_name(platform_id)


def target_dir_for(rocm_dir: Path, version: str, platform_id: str) -> Path:
    return rocm_dir / "launcher" / version / platform_id


def activation_marker_data(manifest: dict, archive_sha256: str, archive_name: str) -> dict:
    return {
        "schema": SCHEMA,
        "version": manifest_version(manifest),
        "platform": manifest_platform(manifest),
        "archive": {
            "name": archive_name,
            "sha256": archive_sha256,
            "format": archive_format(archive_name, manifest),
        },
    }


def read_activation_marker(target_dir: Path) -> dict | None:
    path = target_dir / ACTIVATION_MANIFEST
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def activation_ready(target_dir: Path, manifest: dict, archive_sha256: str, archive_name: str) -> bool:
    marker = read_activation_marker(target_dir)
    if marker != activation_marker_data(manifest, archive_sha256, archive_name):
        return False
    return rocm_binary_path(target_dir, manifest_platform(manifest)).is_file()


def write_activation_marker(target_dir: Path, manifest: dict, archive_sha256: str, archive_name: str) -> None:
    marker = activation_marker_data(manifest, archive_sha256, archive_name)
    (target_dir / ACTIVATION_MANIFEST).write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def replace_with_staging(staging_dir: Path, target_dir: Path) -> None:
    if target_dir.exists():
        shutil.rmtree(target_dir)
    staging_dir.rename(target_dir)


def extract_release_archive(payload: ArchivePayload, target_dir: Path) -> None:
    archive_kind = archive_format(payload.archive_name, payload.manifest)
    parent = target_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{target_dir.name}.tmp-", dir=str(parent)))
    try:
        if archive_kind == "tar.gz":
            extract_tar_gz(payload.archive_bytes, staging_dir)
        elif archive_kind == "zip":
            extract_zip(payload.archive_bytes, staging_dir)
        else:
            raise LauncherError(f"unsupported archive format: {archive_kind}")

        rocm_path = rocm_binary_path(staging_dir, manifest_platform(payload.manifest))
        if not rocm_path.is_file():
            expected = f"bin/{rocm_binary_name(manifest_platform(payload.manifest))}"
            raise LauncherError(f"release archive did not contain {expected}")
        replace_with_staging(staging_dir, target_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise


def verify_archive_hash(payload: ArchivePayload, archive_path: Path | None = None) -> str:
    expected = expected_archive_sha256(payload.manifest, archive_path)
    actual = sha256_bytes(payload.archive_bytes)
    if expected != actual:
        raise LauncherError(
            f"archive sha256 verification failed for {payload.archive_name}: "
            f"expected {expected}, got {actual}"
        )
    return actual


def runner_exit_code(result: object) -> int:
    if isinstance(result, int):
        return result
    return int(getattr(result, "returncode"))


def subprocess_runner(argv: Sequence[str]) -> subprocess.CompletedProcess:
    return subprocess.run(list(argv), check=False)


def run_launcher(
    payload: ArchivePayload,
    *,
    rocm_dir: Path,
    rocm_args: Sequence[str],
    runner: Runner = subprocess_runner,
) -> LaunchResult:
    archive_sha256 = verify_archive_hash(payload)
    version = manifest_version(payload.manifest)
    platform_id = manifest_platform(payload.manifest)
    target_dir = target_dir_for(rocm_dir, version, platform_id)

    reused = activation_ready(target_dir, payload.manifest, archive_sha256, payload.archive_name)
    if not reused:
        extract_release_archive(payload, target_dir)
        write_activation_marker(target_dir, payload.manifest, archive_sha256, payload.archive_name)

    rocm_path = rocm_binary_path(target_dir, platform_id)
    if not rocm_path.is_file():
        raise LauncherError(f"extracted rocm binary is missing: {rocm_path}")

    result = runner([str(rocm_path), *rocm_args])
    return LaunchResult(
        target_dir=target_dir,
        rocm_path=rocm_path,
        reused=reused,
        exit_code=runner_exit_code(result),
    )


def launcher_manifest(
    *,
    version: str,
    platform_id: str,
    archive_name: str,
    archive_bytes: bytes,
    archive_kind: str,
) -> dict:
    return {
        "schema": SCHEMA,
        "experimental": True,
        "version": version,
        "platform": platform_id,
        "archive": {
            "name": archive_name,
            "format": archive_kind,
            "sha256": sha256_bytes(archive_bytes),
            "size": len(archive_bytes),
        },
    }


def write_embedded_launcher(
    *,
    source_path: Path,
    archive_path: Path,
    output_path: Path,
    version: str,
    platform_id: str,
) -> dict:
    archive_bytes = archive_path.read_bytes()
    manifest = launcher_manifest(
        version=version,
        platform_id=platform_id,
        archive_name=archive_path.name,
        archive_bytes=archive_bytes,
        archive_kind=archive_format(archive_path.name, {"archive": {}}),
    )
    payload = {
        "manifest": manifest,
        "archive_b64": base64.b64encode(archive_bytes).decode("ascii"),
    }
    encoded_payload = base64.b64encode(json.dumps(payload, sort_keys=True).encode("utf-8")).decode("ascii")
    source = strip_existing_embedded_payload(source_path.read_bytes())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        handle.write(source)
        handle.write(PAYLOAD_BEGIN.encode("ascii") + b"\n")
        for line in textwrap.wrap(encoded_payload, width=76):
            handle.write(b"# " + line.encode("ascii") + b"\n")
        handle.write(PAYLOAD_END.encode("ascii") + b"\n")
    if os.name != "nt":
        output_path.chmod(output_path.stat().st_mode | stat.S_IXUSR)
    return manifest


def add_tar_member(package: tarfile.TarFile, name: str, data: bytes, mode: int = 0o644) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = mode
    package.addfile(info, io.BytesIO(data))


def create_fake_release_archive(path: Path, *, platform_id: str, root_name: str) -> None:
    rocm_name = rocm_binary_name(platform_id)
    required = {
        f"{root_name}/bin/{rocm_name}": b"fake rocm binary\n",
        f"{root_name}/README.md": b"fake readme\n",
        f"{root_name}/LICENSE": b"fake license\n",
    }
    if path.name.endswith(".zip"):
        with zipfile.ZipFile(path, "w") as package:
            package.writestr(f"{root_name}/", "")
            package.writestr(f"{root_name}/bin/", "")
            for name, data in required.items():
                package.writestr(name, data)
        return

    with tarfile.open(path, "w:gz") as package:
        root_info = tarfile.TarInfo(root_name)
        root_info.type = tarfile.DIRTYPE
        root_info.mode = 0o755
        package.addfile(root_info)
        bin_info = tarfile.TarInfo(f"{root_name}/bin")
        bin_info.type = tarfile.DIRTYPE
        bin_info.mode = 0o755
        package.addfile(bin_info)
        for name, data in required.items():
            mode = 0o755 if name.endswith(f"/{rocm_name}") else 0o644
            add_tar_member(package, name, data, mode=mode)


def self_test_manifest(path: Path, *, version: str, platform_id: str) -> dict:
    archive_bytes = path.read_bytes()
    return launcher_manifest(
        version=version,
        platform_id=platform_id,
        archive_name=path.name,
        archive_bytes=archive_bytes,
        archive_kind=archive_format(path.name, {"archive": {}}),
    )


def expect_launcher_error(label: str, func: Callable[[], object]) -> None:
    try:
        func()
    except LauncherError:
        print(f"launcher spike self-test: {label} rejected as expected")
        return
    raise LauncherError(f"{label} unexpectedly passed")


def run_self_test(root: Path) -> None:
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    try:
        archive = root / "rocm-cli-test-linux-amd64.tar.gz"
        create_fake_release_archive(archive, platform_id="linux-amd64", root_name="rocm-cli-test-linux-amd64")
        manifest = self_test_manifest(archive, version="self-test-1", platform_id="linux-amd64")
        payload = ArchivePayload(manifest=manifest, archive_bytes=archive.read_bytes(), archive_name=archive.name)

        bad_manifest = json.loads(json.dumps(manifest))
        bad_manifest["archive"]["sha256"] = "0" * 64
        bad_payload = ArchivePayload(
            manifest=bad_manifest,
            archive_bytes=archive.read_bytes(),
            archive_name=archive.name,
        )
        expect_launcher_error(
            "bad archive hash",
            lambda: run_launcher(
                bad_payload,
                rocm_dir=root / "bad-home" / ".rocm",
                rocm_args=["version"],
                runner=lambda argv: 0,
            ),
        )

        calls: list[Sequence[str]] = []

        def recording_runner(argv: Sequence[str]) -> int:
            calls.append(list(argv))
            return 7

        rocm_dir = root / "home" / ".rocm"
        first = run_launcher(
            payload,
            rocm_dir=rocm_dir,
            rocm_args=["version", "--json"],
            runner=recording_runner,
        )
        if first.reused:
            raise LauncherError("first extraction should not report reuse")
        if first.exit_code != 7:
            raise LauncherError(f"delegated exit code was not propagated: {first.exit_code}")
        if Path(calls[-1][0]) != first.rocm_path:
            raise LauncherError(f"delegation target mismatch: {calls[-1][0]} != {first.rocm_path}")
        if list(calls[-1][1:]) != ["version", "--json"]:
            raise LauncherError(f"delegation args mismatch: {calls[-1]}")
        print("launcher spike self-test: delegated argv accepted")

        sentinel = first.target_dir / "reuse-sentinel.txt"
        sentinel.write_text("kept\n", encoding="utf-8")
        second = run_launcher(
            payload,
            rocm_dir=rocm_dir,
            rocm_args=["doctor"],
            runner=recording_runner,
        )
        if not second.reused:
            raise LauncherError("second run should reuse the existing extraction")
        if not sentinel.is_file():
            raise LauncherError("reused extraction was unexpectedly rewritten")
        print("launcher spike self-test: extraction reuse accepted")

        zip_archive = root / "rocm-cli-test-windows-amd64.zip"
        create_fake_release_archive(
            zip_archive,
            platform_id="windows-amd64",
            root_name="rocm-cli-test-windows-amd64",
        )
        zip_manifest = self_test_manifest(
            zip_archive,
            version="self-test-zip",
            platform_id="windows-amd64",
        )
        zip_result = run_launcher(
            ArchivePayload(zip_manifest, zip_archive.read_bytes(), zip_archive.name),
            rocm_dir=root / "zip-home" / ".rocm",
            rocm_args=["engines", "list"],
            runner=recording_runner,
        )
        if zip_result.rocm_path.name != "rocm.exe":
            raise LauncherError(f"windows launcher path should use rocm.exe: {zip_result.rocm_path}")
        print("launcher spike self-test: zip archive extraction accepted")

        embedded_output = root / "rocm-cli-linux-amd64-launcher.py"
        embedded_manifest = write_embedded_launcher(
            source_path=Path(__file__).resolve(),
            archive_path=archive,
            output_path=embedded_output,
            version="self-test-embedded",
            platform_id="linux-amd64",
        )
        embedded_payload = load_embedded_payload(embedded_output)
        if embedded_payload.manifest != embedded_manifest:
            raise LauncherError("embedded manifest did not round-trip")
        embedded_result = run_launcher(
            embedded_payload,
            rocm_dir=root / "embedded-home" / ".rocm",
            rocm_args=["version"],
            runner=recording_runner,
        )
        if not embedded_result.rocm_path.is_file():
            raise LauncherError("embedded payload did not extract bin/rocm")
        print("launcher spike self-test: embedded single-file payload accepted")
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("launcher spike self-test: ok")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    command_names = {"run", "build", "self-test"}
    if not argv or argv[0] not in command_names:
        argv = ["run", *argv]

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run from an embedded or external release archive.")
    run_parser.add_argument("--manifest", type=Path, help="Launcher JSON manifest for an external archive.")
    run_parser.add_argument("--archive", type=Path, help="Existing rocm-cli release archive to verify and extract.")
    run_parser.add_argument("--sha256", help="Expected archive SHA-256. Defaults to manifest or .sha256 sidecar.")
    run_parser.add_argument("--version", help="Version directory under .rocm/launcher for external archive mode.")
    run_parser.add_argument("--platform", help="Platform directory such as linux-amd64 or windows-amd64.")
    run_parser.add_argument("--rocm-dir", type=Path, help="Override the .rocm directory used by the launcher spike.")
    run_parser.add_argument("rocm_args", nargs=argparse.REMAINDER, help="Arguments delegated to bin/rocm.")

    build_parser = subparsers.add_parser("build", help="Build a generated single-file launcher script.")
    build_parser.add_argument("--archive", type=Path, required=True, help="Existing rocm-cli release archive to embed.")
    build_parser.add_argument("--output", type=Path, required=True, help="Generated launcher script path.")
    build_parser.add_argument("--version", required=True, help="Version directory under .rocm/launcher.")
    build_parser.add_argument("--platform", default=None, help="Platform directory. Defaults to this host.")

    self_test_parser = subparsers.add_parser("self-test", help="Run focused launcher spike self-tests.")
    self_test_parser.add_argument("--root", type=Path, default=DEFAULT_SELF_TEST_ROOT)

    return parser.parse_args(list(argv))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if args.command == "self-test":
            run_self_test(args.root)
            return 0

        if args.command == "build":
            platform_id = args.platform or host_platform_id()
            manifest = write_embedded_launcher(
                source_path=Path(__file__).resolve(),
                archive_path=args.archive.resolve(),
                output_path=args.output.resolve(),
                version=args.version,
                platform_id=platform_id,
            )
            print("rocm-cli launcher spike: generated experimental launcher")
            print(f"  output: {args.output}")
            print(f"  version: {manifest_version(manifest)}")
            print(f"  platform: {manifest_platform(manifest)}")
            print(f"  archive_sha256: {manifest_archive(manifest)['sha256']}")
            return 0

        payload = payload_from_args(args, Path(__file__).resolve())
        rocm_dir = args.rocm_dir if args.rocm_dir else default_rocm_dir()
        rocm_args = list(args.rocm_args)
        if rocm_args and rocm_args[0] == "--":
            rocm_args = rocm_args[1:]
        print("rocm-cli launcher spike: experimental path")
        print(f"  version: {manifest_version(payload.manifest)}")
        print(f"  platform: {manifest_platform(payload.manifest)}")
        print(f"  launcher_dir: {target_dir_for(rocm_dir, manifest_version(payload.manifest), manifest_platform(payload.manifest))}")
        result = run_launcher(payload, rocm_dir=rocm_dir, rocm_args=rocm_args)
        if result.reused:
            print("  extraction: reused")
        else:
            print("  extraction: created")
        return result.exit_code
    except LauncherError as error:
        fail(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
