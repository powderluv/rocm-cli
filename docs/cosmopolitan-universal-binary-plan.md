# Cosmopolitan Universal Binary Plan

This document defines what "single universal binary" means for rocm-cli and
separates it from the self-extracting APE launcher spike.

## Definitions

- **Platform-native standalone**: the Rust `rocm` or `rocm.exe` binary built for
  one OS. This is the active release artifact today.
- **Self-extracting APE launcher**: a small Cosmopolitan C executable with a ZIP
  payload appended. It extracts a Windows or Linux Rust release payload and then
  delegates to that native binary. This is a compatibility spike, not the final
  universal binary.
- **True no-extract APE**: the actual rocm-cli program is compiled and linked as
  one Cosmopolitan executable. The executable can run on supported OSes without
  unpacking separate `rocm.exe`/`rocm` binaries first.

The target product requirement is the third shape.

## Upstream Facts

- Cosmopolitan Libc is for C/C++ programs and reconfigures GCC/Clang to emit an
  Actually Portable Executable that runs across supported OSes without a VM:
  https://github.com/jart/cosmopolitan
- APE loading maps the executable into memory. If the system APE loader is not
  installed, the embedded loader may be copied to a small `.ape` helper path,
  but the normal APE model is not "extract this app into OS-specific payloads":
  https://justine.lol/apeloader/
- Rust targets are selected with `--target`, and built-in targets are visible
  through `rustc --print target-list`:
  https://doc.rust-lang.org/rustc/targets/index.html
- Rust custom targets exist, but the Rust book says the target JSON/build-std
  path is unstable and must pin the compiler:
  https://doc.rust-lang.org/rustc/targets/custom.html
- Independent Rust/Cosmopolitan experiments exist, including `std` experiments,
  but they are not an official Rust target and are not a drop-in guarantee for
  this repo:
  https://ahgamut.github.io/2022/07/27/ape-rust-example/

Local probe on 2026-06-05:

```text
rustc 1.96.0 does not report a built-in target containing cosmo, cosmopolitan,
or ape.
```

## Current Repo Reality

The repo currently has these runnable release paths:

- `scripts/build_single_exe_release.py standalone`
  - builds/copies the platform-native Rust `rocm` or `rocm.exe`;
  - no extraction;
  - not universal across OSes.
- `scripts/build_single_exe_release.py universal`
  - builds a Cosmopolitan C wrapper;
  - appends Windows and Linux platform release archives;
  - extracts the matching payload to a cache and delegates to that binary;
  - one file to distribute, but not a true no-extract rocm-cli binary.

The self-extracting launcher remains useful only as a compatibility fallback or
as an APE behavior test bed. It must not be presented as the final answer.

## Implementation Tracks

### Track A: Native Cosmopolitan Bootstrap/Core

Build a small C/C++ Cosmopolitan program that owns the first-run setup flow:

1. Detect OS and AMD GPU basics.
2. If AMD runtime/driver is missing, show simple driver guidance.
3. Provide the cross-platform folder picker.
4. Download or locate Python 3.12.10 when Python is missing.
5. Install TheRock wheels into the selected folder using rocm-cli's current
   package-selection rules.
6. Write the normal `~/.rocm` JSON config and runtime registry.

This track can produce a real no-extract APE sooner because Cosmopolitan's
supported language path is C/C++. It will initially be a bootstrap/core program,
not the full Rust TUI.

### Track B: Rust-To-Cosmopolitan Feasibility

Prove whether the existing Rust CLI can become a true APE:

1. Pin a Rust toolchain.
2. Create a custom target JSON for `x86_64-unknown-cosmo` or equivalent.
3. Build a tiny `std` Rust program with `cosmocc`.
4. Build a tiny subset crate from this workspace.
5. Attempt `rocm` with minimal features.
6. Record every required patch to dependencies, `std`, linker args, and ABI
   assumptions.

Acceptance for this track is not "hello world". It must either build a useful
slice of rocm-cli as one APE or produce a clear blocker list.

### Track C: Collapse Multi-Binary Runtime Shape

A true no-extract rocm-cli cannot depend on sibling engine adapter executables
such as `rocm-engine-pytorch`, `rocm-engine-llama-cpp`, or `rocmd` unless those
are also compiled into the same executable or replaced by in-process modules.

Work items:

1. Inventory every spawned rocm-cli helper binary.
2. Decide which helpers must become in-process modules.
3. Keep external Python environments, TheRock wheels, ComfyUI, models, and
   ROCm libraries as managed installed content. Those are runtime payloads the
   user asks rocm-cli to install, not part of the CLI executable itself.

## Acceptance Criteria

A true universal rocm-cli binary is accepted only when:

1. One produced file runs on Windows, WSL/Linux, and native Linux without
   extracting separate `rocm.exe` or `rocm` binaries.
2. The first-run setup/bootstrap UI opens directly.
3. The folder picker works cross-platform.
4. Python 3.12.10 can be installed or selected if missing.
5. TheRock installs into the user-selected folder.
6. Read-only system/GPU checks do not require approval.
7. Mutating install/uninstall/config operations still require clear approval.
8. The executable does not claim GPU readiness unless the AMD runtime needed by
   the selected workflow is actually available.

## Automation

Run the current feasibility probe:

```bash
python scripts/cosmopolitan_feasibility.py self-test
python scripts/cosmopolitan_feasibility.py probe
```

When a Cosmopolitan compiler is available:

```bash
python scripts/cosmopolitan_feasibility.py probe \
  --compiler .rocm-work/tools/cosmocc-wsl-elf/bin/cosmocc \
  --compile-c
```

The probe is intentionally small. It should fail loudly if repo wording drifts
back to calling the self-extracting wrapper a true universal binary.
