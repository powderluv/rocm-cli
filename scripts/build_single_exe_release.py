#!/usr/bin/env python3
"""Build rocm-cli release artifacts.

The active single-file release is a universal APE launcher named `rocm.exe`.
It embeds the Windows and Linux rocm-cli payloads and delegates to the matching
platform binary at runtime. It does not embed a bootstrap assistant, model,
runtime sidecar, or vendored Codex binary. Running it with no arguments opens
the first-time setup wizard when setup has not been completed yet.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import os
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

WINDOWS_BINARIES = [
    "rocm.exe",
    "rocmd.exe",
    "rocm-engine-pytorch.exe",
    "rocm-engine-llama-cpp.exe",
    "rocm-engine-lemonade.exe",
    "rocm-engine-atom.exe",
    "rocm-engine-vllm.exe",
    "rocm-engine-sglang.exe",
]
LINUX_BINARIES = [name[:-4] for name in WINDOWS_BINARIES]
PLATFORMS = {"windows-amd64", "linux-amd64"}
BACKEND_NAME_BY_PLATFORM = {
    "windows-amd64": "ggml-rocm.dll",
    "linux-amd64": "ggml-rocm.so",
}
RUNTIME_DIR_BY_PLATFORM = {
    "windows-amd64": "windows-runtime",
    "linux-amd64": "linux-runtime",
}


class ReleaseBuildError(Exception):
    """The release artifact could not be built."""


def fail(message: str) -> None:
    print(f"single-exe release build failed: {message}", file=sys.stderr)
    raise SystemExit(1)


def run(args: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(args))
    completed = subprocess.run(args, cwd=cwd, env=env, text=True, check=False)
    if completed.returncode != 0:
        raise ReleaseBuildError(f"command failed with exit {completed.returncode}: {' '.join(args)}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_sha256(path: Path) -> None:
    digest = sha256_file(path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n",
        encoding="ascii",
    )


def current_platform() -> str:
    if os.name == "nt":
        return "windows-amd64"
    if sys.platform.startswith("linux"):
        return "linux-amd64"
    raise ReleaseBuildError(f"unsupported build host: {sys.platform}")


def cargo_build_release(repo_root: Path, jobs: int | None) -> None:
    env = os.environ.copy()
    if jobs is not None:
        env["CARGO_BUILD_JOBS"] = str(jobs)
    run(["cargo", "build", "--release", "--workspace", "--bins"], cwd=repo_root, env=env)


def cargo_build_rocm_release(repo_root: Path, jobs: int | None) -> None:
    env = os.environ.copy()
    if jobs is not None:
        env["CARGO_BUILD_JOBS"] = str(jobs)
    run(["cargo", "build", "--release", "-p", "rocm", "--bin", "rocm"], cwd=repo_root, env=env)


def strip_tool_for(platform: str) -> str | None:
    candidates: list[str] = []
    configured = os.environ.get("ROCM_CLI_STRIP")
    if configured:
        candidates.append(configured)
    if platform == "windows-amd64":
        candidates.extend(
            [
                r"D:\jam\venv\Lib\site-packages\_rocm_sdk_core\lib\llvm\bin\llvm-strip.exe",
                r"D:\jam\venv\Lib\site-packages\_rocm_sdk_devel\lib\llvm\bin\llvm-strip.exe",
                "llvm-strip",
            ]
        )
    else:
        candidates.extend(["llvm-strip", "strip"])
    for candidate in candidates:
        found = shutil.which(candidate) if not Path(candidate).is_file() else candidate
        if found:
            return str(found)
    return None


def maybe_strip(path: Path, *, platform: str, strip_debug_only: bool = False) -> None:
    tool = strip_tool_for(platform)
    if tool is None:
        return
    flag = "--strip-debug" if strip_debug_only else "--strip-all"
    completed = subprocess.run(
        [tool, flag, str(path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if completed.returncode != 0 and not strip_debug_only:
        subprocess.run(
            [tool, "--strip-debug", str(path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )


def copy_required(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise ReleaseBuildError(f"required file not found: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def platform_binaries(platform: str) -> list[str]:
    if platform == "windows-amd64":
        return WINDOWS_BINARIES
    if platform == "linux-amd64":
        return LINUX_BINARIES
    raise ReleaseBuildError(f"unsupported platform: {platform}")


def standalone_binary(platform: str) -> str:
    if platform == "windows-amd64":
        return "rocm.exe"
    if platform == "linux-amd64":
        return "rocm"
    raise ReleaseBuildError(f"unsupported platform: {platform}")


def platform_install_script(platform: str) -> str:
    return "install.ps1" if platform == "windows-amd64" else "install.sh"


def create_zip(root: Path, archive: Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as package:
        for path in sorted(root.rglob("*")):
            if path.is_dir():
                continue
            package.write(path, path.relative_to(root.parent).as_posix())


def reset_tar_info(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    return info


def create_tar_gz(root: Path, archive: Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    with archive.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w") as package:
                package.add(root, arcname=root.name, filter=reset_tar_info)


def assert_no_codex(archive: Path) -> None:
    names: list[str]
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as package:
            names = package.namelist()
    elif archive.name.endswith(".tar.gz"):
        with tarfile.open(archive, "r:gz") as package:
            names = package.getnames()
    else:
        raise ReleaseBuildError(f"unsupported archive type: {archive}")
    offenders = [name for name in names if "rocm-codex" in name or "/codex" in name]
    if offenders:
        raise ReleaseBuildError(f"Codex binary leaked into release archive: {offenders[:3]}")


def stage_platform_release(
    *,
    repo_root: Path,
    output_dir: Path,
    version: str,
    platform: str,
    skip_cargo_build: bool,
    strip_binaries: bool,
    jobs: int | None,
) -> Path:
    if platform not in PLATFORMS:
        raise ReleaseBuildError(f"platform must be one of {sorted(PLATFORMS)}")
    if not skip_cargo_build:
        cargo_build_release(repo_root, jobs)

    dist_name = f"rocm-cli-v{version}-{platform}"
    staging_root = output_dir / "staging" / platform / dist_name
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True)
    bin_dir = staging_root / "bin"
    bin_dir.mkdir()

    profile_dir = repo_root / "target" / "release"
    for binary in platform_binaries(platform):
        destination = bin_dir / binary
        copy_required(profile_dir / binary, destination)
        if strip_binaries:
            maybe_strip(destination, platform=platform)

    for name in ["README.md", "LICENSE", platform_install_script(platform)]:
        copy_required(repo_root / name, staging_root / name)

    archive = output_dir / (f"{dist_name}.zip" if platform == "windows-amd64" else f"{dist_name}.tar.gz")
    if archive.exists():
        archive.unlink()
    if platform == "windows-amd64":
        create_zip(staging_root, archive)
    else:
        create_tar_gz(staging_root, archive)
    assert_no_codex(archive)
    write_sha256(archive)
    print(f"wrote {archive}")
    print(f"sha256 {sha256_file(archive)}")
    return archive


def build_standalone_release(
    *,
    repo_root: Path,
    output_dir: Path,
    output: Path | None,
    platform: str,
    skip_cargo_build: bool,
    strip_binary: bool,
    write_digest: bool,
    jobs: int | None,
) -> Path:
    if platform not in PLATFORMS:
        raise ReleaseBuildError(f"platform must be one of {sorted(PLATFORMS)}")
    if platform != current_platform() and not skip_cargo_build:
        raise ReleaseBuildError(
            f"standalone builds target the current host by default ({current_platform()}); "
            f"run this command on {platform} or pass --skip-cargo-build after staging the binary"
        )
    if not skip_cargo_build:
        cargo_build_rocm_release(repo_root, jobs)

    binary = standalone_binary(platform)
    source = repo_root / "target" / "release" / binary
    destination = output.resolve() if output is not None else output_dir.resolve() / binary
    copy_required(source, destination)
    if strip_binary:
        maybe_strip(destination, platform=platform)
    if platform == "linux-amd64":
        destination.chmod(destination.stat().st_mode | stat.S_IXUSR)
    if write_digest:
        write_sha256(destination)
    print(f"wrote {destination}")
    if write_digest:
        print(f"sha256 {sha256_file(destination)}")
    return destination


def sorted_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise ReleaseBuildError(f"directory not found: {directory}")
    return sorted(path for path in directory.iterdir() if path.is_file())


def stage_runtime(
    *,
    output_dir: Path,
    platform: str,
    rocm_backend: Path,
    runtime_dir: Path,
    strip_debug: bool,
) -> None:
    if platform not in PLATFORMS:
        raise ReleaseBuildError(f"platform must be one of {sorted(PLATFORMS)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    backend_name = BACKEND_NAME_BY_PLATFORM[platform]
    backend_target = output_dir / backend_name
    copy_required(rocm_backend, backend_target)
    if strip_debug:
        maybe_strip(backend_target, platform=platform, strip_debug_only=True)

    runtime_target = output_dir / RUNTIME_DIR_BY_PLATFORM[platform]
    if runtime_target.exists():
        shutil.rmtree(runtime_target)
    runtime_target.mkdir(parents=True)
    for source in sorted_files(runtime_dir):
        target = runtime_target / source.name
        copy_required(source, target)
        if strip_debug and (source.suffix.lower() in {".dll", ".so"} or ".so." in source.name):
            maybe_strip(target, platform=platform, strip_debug_only=True)
    print(f"wrote {backend_target}")
    print(f"wrote {runtime_target}")


def build_ape(args: argparse.Namespace) -> Path:
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = args.manifest.resolve()
    runtime_args: list[str] = []
    for path in sorted_files(args.windows_runtime_dir.resolve()):
        runtime_args.extend(["--windows-runtime-dependency", str(path)])
    for path in sorted_files(args.linux_runtime_dir.resolve()):
        runtime_args.extend(["--linux-runtime-dependency", str(path)])

    plan_args = [
        sys.executable,
        str(SCRIPT_DIR / "ape_bootstrap_package.py"),
        "plan",
        "--version",
        args.version,
        "--windows-release",
        str(args.windows_release.resolve()),
        "--linux-release",
        str(args.linux_release.resolve()),
        "--model",
        str(args.model.resolve()),
        "--windows-rocm-backend",
        str(args.windows_rocm_backend.resolve()),
        "--linux-rocm-backend",
        str(args.linux_rocm_backend.resolve()),
        *runtime_args,
        "--output",
        str(manifest),
    ]
    run(plan_args, cwd=REPO_ROOT)
    run(
        [
            sys.executable,
            str(SCRIPT_DIR / "ape_bootstrap_package.py"),
            "validate",
            "--manifest",
            str(manifest),
        ],
        cwd=REPO_ROOT,
    )
    run(
        [
            sys.executable,
            str(SCRIPT_DIR / "build_ape_bootstrap.py"),
            "build",
            "--manifest",
            str(manifest),
            "--output",
            str(output),
            "--compiler",
            str(args.compiler),
            "--work-dir",
            str(args.work_dir.resolve()),
        ],
        cwd=REPO_ROOT,
    )
    write_sha256(output)
    print(f"wrote {output}")
    print(f"sha256 {sha256_file(output)}")
    return output


def archive_path_for(output_dir: Path, *, version: str, platform: str) -> Path:
    suffix = ".zip" if platform == "windows-amd64" else ".tar.gz"
    return output_dir / f"rocm-cli-v{version}-{platform}{suffix}"


def build_universal(args: argparse.Namespace) -> Path:
    archive_dir = args.archive_dir.resolve()
    windows_release = (args.windows_release or archive_path_for(archive_dir, version=args.version, platform="windows-amd64")).resolve()
    linux_release = (args.linux_release or archive_path_for(archive_dir, version=args.version, platform="linux-amd64")).resolve()
    if not windows_release.is_file():
        raise ReleaseBuildError(
            f"missing Windows release archive: {windows_release}; "
            "run stage-platform on Windows first or pass --windows-release"
        )
    if not linux_release.is_file():
        raise ReleaseBuildError(
            f"missing Linux release archive: {linux_release}; "
            "run stage-platform on Linux/WSL first or pass --linux-release"
        )
    output = (args.output or (args.output_dir / "rocm.exe")).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(SCRIPT_DIR / "build_ape_bootstrap.py"),
        "build-universal",
        "--windows-release",
        str(windows_release),
        "--linux-release",
        str(linux_release),
        "--output",
        str(output),
        "--compiler",
        str(args.compiler),
        "--work-dir",
        str(args.work_dir.resolve()),
        "--version",
        args.universal_version,
    ]
    run(command, cwd=REPO_ROOT)
    write_sha256(output)
    print(f"wrote {output}")
    print(f"sha256 {sha256_file(output)}")
    return output


def run_self_test(root: Path) -> None:
    if root.exists():
        shutil.rmtree(root)
    fake_repo = root / "repo"
    fake_out = root / "out"
    try:
        for platform in sorted(PLATFORMS):
            profile = fake_repo / "target" / "release"
            profile.mkdir(parents=True, exist_ok=True)
            for binary in platform_binaries(platform):
                path = profile / binary
                path.write_bytes(b"fake release binary\n")
                if platform == "linux-amd64":
                    path.chmod(path.stat().st_mode | stat.S_IXUSR)
            for name in ["README.md", "LICENSE", "install.ps1", "install.sh"]:
                (fake_repo / name).write_text(f"fake {name}\n", encoding="utf-8")
            standalone = build_standalone_release(
                repo_root=fake_repo,
                output_dir=fake_out / "standalone" / platform,
                output=None,
                platform=platform,
                skip_cargo_build=True,
                strip_binary=False,
                write_digest=False,
                jobs=None,
            )
            assert standalone.name == standalone_binary(platform)
            archive = stage_platform_release(
                repo_root=fake_repo,
                output_dir=fake_out,
                version="0.0.0-test",
                platform=platform,
                skip_cargo_build=True,
                strip_binaries=False,
                jobs=None,
            )
            assert_no_codex(archive)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("single-exe release builder self-test: ok")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    standalone = subparsers.add_parser("standalone", help="Build the standalone rocm-cli binary.")
    standalone.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    standalone.add_argument("--output-dir", type=Path, default=REPO_ROOT / ".rocm-work" / "standalone-release")
    standalone.add_argument("--output", type=Path)
    standalone.add_argument("--platform", choices=sorted(PLATFORMS), default=current_platform())
    standalone.add_argument("--skip-cargo-build", action="store_true")
    standalone.add_argument("--no-strip", action="store_true")
    standalone.add_argument("--write-sha256", action="store_true")
    standalone.add_argument("--jobs", type=int, default=96)

    stage = subparsers.add_parser("stage-platform", help="Build and archive the current platform payload.")
    stage.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    stage.add_argument("--output-dir", type=Path, default=REPO_ROOT / ".rocm-work" / "ape-min-release")
    stage.add_argument("--version", default="0.2.0")
    stage.add_argument("--platform", choices=sorted(PLATFORMS), default=current_platform())
    stage.add_argument("--skip-cargo-build", action="store_true")
    stage.add_argument("--no-strip", action="store_true")
    stage.add_argument("--jobs", type=int, default=96)

    runtime = subparsers.add_parser("stage-runtime", help="Copy and strip ROCm runtime sidecars.")
    runtime.add_argument("--output-dir", type=Path, default=REPO_ROOT / ".rocm-work" / "ape-min-release")
    runtime.add_argument("--platform", choices=sorted(PLATFORMS), default=current_platform())
    runtime.add_argument("--rocm-backend", type=Path, required=True)
    runtime.add_argument("--runtime-dir", type=Path, required=True)
    runtime.add_argument("--no-strip", action="store_true")

    ape = subparsers.add_parser("build-ape", help="Build the final universal APE from staged inputs.")
    ape.add_argument("--version", default="0.2.0-release-min-20260604")
    ape.add_argument("--manifest", type=Path, required=True)
    ape.add_argument("--output", type=Path, required=True)
    ape.add_argument("--windows-release", type=Path, required=True)
    ape.add_argument("--linux-release", type=Path, required=True)
    ape.add_argument("--model", type=Path, required=True)
    ape.add_argument("--windows-rocm-backend", type=Path, required=True)
    ape.add_argument("--linux-rocm-backend", type=Path, required=True)
    ape.add_argument("--windows-runtime-dir", type=Path, required=True)
    ape.add_argument("--linux-runtime-dir", type=Path, required=True)
    ape.add_argument("--compiler", required=True)
    ape.add_argument("--work-dir", type=Path, default=REPO_ROOT / ".rocm-work" / "ape-min-release" / "builder")

    universal = subparsers.add_parser("universal", help="Build the universal single-exe rocm launcher from staged platform archives.")
    universal.add_argument("--version", default="0.2.0", help="Release archive version to consume.")
    universal.add_argument("--universal-version", default="0.2.0-universal", help="Version key used by the launcher extraction cache.")
    universal.add_argument("--archive-dir", type=Path, default=REPO_ROOT / ".rocm-work" / "ape-min-release")
    universal.add_argument("--output-dir", type=Path, default=REPO_ROOT / ".rocm-work" / "standalone-release")
    universal.add_argument("--output", type=Path)
    universal.add_argument("--windows-release", type=Path)
    universal.add_argument("--linux-release", type=Path)
    universal.add_argument("--compiler", required=True)
    universal.add_argument("--work-dir", type=Path, default=REPO_ROOT / ".rocm-work" / "universal-release" / "builder")

    self_test = subparsers.add_parser("self-test", help="Run offline archive policy tests.")
    self_test.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT / ".rocm-work" / "tests" / f"single-exe-release-builder-{os.getpid()}",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "standalone":
            build_standalone_release(
                repo_root=args.repo_root.resolve(),
                output_dir=args.output_dir.resolve(),
                output=args.output,
                platform=args.platform,
                skip_cargo_build=args.skip_cargo_build,
                strip_binary=not args.no_strip,
                write_digest=args.write_sha256,
                jobs=args.jobs,
            )
            return 0
        if args.command == "stage-platform":
            stage_platform_release(
                repo_root=args.repo_root.resolve(),
                output_dir=args.output_dir.resolve(),
                version=args.version,
                platform=args.platform,
                skip_cargo_build=args.skip_cargo_build,
                strip_binaries=not args.no_strip,
                jobs=args.jobs,
            )
            return 0
        if args.command == "stage-runtime":
            stage_runtime(
                output_dir=args.output_dir.resolve(),
                platform=args.platform,
                rocm_backend=args.rocm_backend.resolve(),
                runtime_dir=args.runtime_dir.resolve(),
                strip_debug=not args.no_strip,
            )
            return 0
        if args.command == "build-ape":
            build_ape(args)
            return 0
        if args.command == "universal":
            build_universal(args)
            return 0
        if args.command == "self-test":
            run_self_test(args.root.resolve())
            return 0
    except (ReleaseBuildError, OSError, subprocess.SubprocessError) as error:
        fail(str(error))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
