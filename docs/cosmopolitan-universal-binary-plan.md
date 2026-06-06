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
rustc 1.98.0-nightly (2026-06-04) does not report a built-in target containing
cosmo, cosmopolitan, or ape.
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

As of 2026-06-05, the repo also has a true Rust/Cosmopolitan feasibility
builder:

- `scripts/setup-cosmocc.sh`
  - downloads `cosmocc` under `.rocm-work/tools/cosmocc`;
  - creates a WSL-safe ELF-converted toolchain under
    `.rocm-work/tools/cosmocc-wsl-elf`.
- `scripts/rust_cosmopolitan_spike.py install-toolchain`
  - installs a workspace-local nightly Rust toolchain plus `rust-src` under
    `.rocm-work/tools`.
- `scripts/rust_cosmopolitan_spike.py build-rocm --release --clean --jobs 96`
  - builds the real Rust `rocm` binary as a no-extract Cosmopolitan APE;
  - writes `.rocm-work/tests/rust-cosmopolitan/rocm-rust-cosmo-release.exe`.

The Rust APE currently uses a Linux-shaped custom Rust target so `std` can link
against Cosmopolitan. Product code must use `rocm_core::RuntimePlatform` for
runtime OS decisions; direct `cfg!(windows)` and `std::env::consts::OS` are
compile-target facts and are not enough for a universal APE.

Current Rust `std::process` also compiles Linux pidfd support for this custom
target. Rust's pidfd code references the public POSIX `waitid` symbol. The
downloaded Cosmopolitan toolchain exposes the lower-level `sys_waitid` syscall
entry but not the public libc wrapper, so `scripts/rust_cosmopolitan_spike.py`
now patches the workspace-local Cosmopolitan source tree under `.rocm-work` and
archives a real `libc/proc/waitid.c` implementation into `libcosmo.a` before
building rocm-cli. The linker wrapper does not inject rocm-cli-local syscall
objects or compatibility aliases.

Validated clean rebuild on 2026-06-05:

- `.rocm-work` was deleted, then recreated by the setup/build scripts above.
- Revalidated after removing the old linker-injected waitid object path:
  `libcosmo.a` now exports `waitid` from the Cosmopolitan source-level patch,
  and the generated linker wrapper contains no `ROCM_CLI_RUST_COSMO_WAITID_OBJECT`,
  no `ROCM_CLI_RUST_COSMO_COMPAT_OBJECT`, and no `--allow-multiple-definition`.
- Native Windows smoke:
  - `version` prints `rocm 0.2.0`;
  - `doctor` reports `os: windows`, AMD Ryzen Threadripper PRO 9995WX,
    Radeon RX 9070 XT, `detected_gfx_target: gfx1201`,
    `compatible_therock_family: gfx120X-all`, and AMD display driver
    `32.0.23033.1002`.
- WSL smoke through the Cosmopolitan APE loader reports `os: linux`,
  `wsl: true`, `driver_policy: wsl_rocdxg`, `driver_status:
  wsl_rocdxg_ready`, and `detected_gfx_target: gfx1201`.
- WSL smoke through `sh <same APE> doctor` reports the Linux path and fails if
  the output reports `os: windows`.

WSL caveat: WSLInterop registers an `MZ` binfmt handler that can intercept
direct `./rocm` execution before rocm-cli starts, even when the file has no
`.exe` extension. Use `sh ./rocm ...` or the bundled `ape-x86_64.elf ./rocm
...` launch path on WSL to force the Linux runtime path without changing the
single file.

Current release shape: first-party engine adapters and the small `rocmd`
service/tool helper surface are built into `rocm`. The universal APE no longer
requires sibling `rocm-engine-*` or `rocmd` files for doctor/setup, bridge
snapshots, service list/stop/restart helpers, foreground serving, or managed
serving launch. External Python environments, TheRock wheels, ComfyUI, ROCm
libraries, and models are still installed on disk as user-managed runtime
content.

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

1. Done: pin a workspace-local nightly Rust toolchain.
2. Done: create a custom Linux-shaped `x86_64-unknown-linux-cosmo` target JSON.
3. Done: build a tiny Rust `std` APE with `cosmocc`.
4. Done: build the real `rocm` binary as one APE.
5. Done: smoke `version` and `doctor` on native Windows and WSL.
6. Done: no-arg first-time setup TUI opens from the APE on Windows in a PTY.
7. Remaining: graduate the spike script into the release pipeline.

Acceptance for this track is no longer "hello world". The current artifact runs
useful rocm-cli doctor/setup code as one APE; the remaining blocker list is now
about release hardening and helper binaries, not Rust/Cosmopolitan feasibility.

### Track C: Collapse Multi-Binary Runtime Shape

A true no-extract rocm-cli cannot depend on sibling engine adapter executables
such as `rocm-engine-pytorch`, `rocm-engine-llama-cpp`, or `rocmd` unless those
are also compiled into the same executable or replaced by in-process modules.

Work items:

1. Done: inventory every spawned rocm-cli helper binary.
2. Done: first-party engines are linked into `rocm` and exposed through hidden
   internal `__engine-stdio` and `__engine-serve-http` routes.
3. Done: the `rocmd` status/bridge/sandbox service helper surface used by the
   TUI is implemented inside `rocm`; `rocm_core::daemon_binary_path()` now
   resolves to the current executable for same-file helper invocations.
4. Keep external Python environments, TheRock wheels, ComfyUI, models, and
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

Build the current Rust APE from a clean workspace-local tool cache:

```bash
rm -rf .rocm-work
scripts/setup-cosmocc.sh
python3 scripts/rust_cosmopolitan_spike.py install-toolchain
python3 scripts/rust_cosmopolitan_spike.py build-rocm --release --clean --jobs 96
python3 scripts/rust_cosmopolitan_spike.py smoke-wsl-linux-path --release
```

The output is:

```text
.rocm-work/tests/rust-cosmopolitan/rocm-rust-cosmo-release.exe
```
