# Bootstrap Packaging Validation

This page defines the current acceptance harness for historical pre-TheRock
bootstrap assistant and self-extracting APE launcher spikes. The default path is
offline and fixture-backed: it must not download llamafiles, GGUF weights,
TheRock wheels, ROCm packages, or driver artifacts.

For the current true no-extract Cosmopolitan binary requirement, see
`docs/cosmopolitan-universal-binary-plan.md`. The APE launcher described here
extracts platform-native payloads and is not the final universal binary target.

## Offline CI Harness

Run:

```bash
python scripts/bootstrap_packaging_validation.py --self-test
```

The harness uses `tests/fixtures/bootstrap_packaging/fake_bootstrap_matrix.json`
and validates:

- read-only bootstrap tool calls run without approval
- mutating install/start/stop/config calls become approval-required decisions
- unsupported shell, PowerShell, package-manager, pipe, and `--yes` flows are
  rejected
- bootstrap server startup rejects public binds and CPU fallback, and requires
  the fake `--jinja` tool-call mode
- a no-supported-GPU doctor fixture hard-blocks mutating install/start actions
  before approval
- TheRock prompts for build date, custom folder, and exact version preserve the
  expected argv-style install request
- fake launcher payload extraction verifies the embedded SHA-256, rejects
  unsafe archive paths, and reuses an already extracted version

The fake server implements only loopback `/health` and
`/v1/chat/completions` with OpenAI-style tool calls. It proves the validation
contract around tool routing, not model quality.

## APE Bootstrap Package Contract

Run:

```bash
python scripts/ape_bootstrap_package.py self-test
```

This separate offline harness validates the historical self-extracting APE
launcher contract:

- one AMD64 APE launcher payload carries both Windows and Linux rocm-cli release
  archives and extracts the matching platform payload
- the bootstrap assistant model is an embedded Qwen 0.8B-class `.llamafile`
- the bootstrap payload also includes AMD GPU helper sidecars for both
  platforms: `ggml-rocm.dll` and `ggml-rocm.so`
- Linux/WSL bootstrap payloads include `ape-x86_64.elf` beside the model so an
  embedded APE llamafile can run without global `binfmt_misc` changes
- startup is loopback-only, OpenAI/Jinja-capable, AMD GPU-required, and asks for
  maximum GPU offload with quiet foreground logs (`-lv 0`)
- after TheRock succeeds, the payload advertises a post-setup handoff that
  installs the extracted `rocm` CLI into a user-selected folder, then asks
  separately before adding that folder to PATH
- CPU fallback, public bind, disabled GPU, `-ngl 0`, wrong model family/size,
  missing platform payloads, silent PATH mutation, and oversize Windows payloads
  are rejected
- staged payload ZIP entries are uncompressed so the later APE/zipalign builder
  can preserve the embedding contract

The hidden implementation command is:

```bash
rocm bootstrap install-cli --target <user-selected-folder>
rocm bootstrap install-cli --target <user-selected-folder> --add-to-path
```

The first command copies the extracted platform `bin` payload into the selected
folder and writes `.rocm-cli-manifest`. The second form is only for the user's
explicit "yes, add it to PATH" choice; the bootstrap flow must not include
`--add-to-path` in the initial self-install action.

## APE Launcher Builder

Run the offline launcher builder self-tests:

```bash
python scripts/build_ape_bootstrap.py self-test
```

On Windows, pass the TheRock/ROCm LLVM clang from a managed environment instead
of Strawberry GCC:

```powershell
python scripts\build_ape_bootstrap.py self-test `
  --compiler D:\jam\venv\Lib\site-packages\_rocm_sdk_core\lib\llvm\bin\clang.exe
```

On WSL, prepare and use the workspace-local Cosmopolitan compiler:

```bash
scripts/setup-cosmocc.sh
python3 scripts/build_ape_bootstrap.py self-test \
  --compiler .rocm-work/tools/cosmocc-wsl-elf/bin/cosmocc
```

The WSL setup script keeps the raw `cosmocc` download under
`.rocm-work/tools/cosmocc` and creates `.rocm-work/tools/cosmocc-wsl-elf` by
assimilating APE helper binaries to ELF. That avoids global `binfmt_misc`
changes and prevents WSLInterop from treating nested APE compiler helpers as
Windows programs.

If WSL does not have `unzip`, the setup script uses Python's zip reader and
restores the executable bits from the archive before running `assimilate`.

Production `build` now requires a compiler that looks like Cosmopolitan
`cosmocc`/APE. Local `clang`/`gcc` builds are still allowed for `self-test`, and
can be used for extraction-only development builds only with
`--allow-local-compiler`.

## Standalone Single-Exe Release Builder

Use `scripts/build_single_exe_release.py standalone` for the current repeatable
release artifact. The output is the `rocm`/`rocm.exe` binary itself: no
self-extraction, no embedded Qwen/llamafile, no ROCm sidecars, and no vendored
Codex binary. This artifact is platform-native, not Cosmopolitan-universal.
Running it with no arguments opens rocm-cli; on first run, the dedicated setup
wizard appears automatically.

Typical flow:

```bash
python scripts/build_single_exe_release.py standalone
```

On Windows this writes `.rocm-work/standalone-release/rocm.exe`; on Linux it
writes `.rocm-work/standalone-release/rocm`.

The older APE commands below are historical self-extracting launcher tooling
and are not the active release path or the final no-extract Cosmopolitan target:

```bash
python scripts/build_single_exe_release.py stage-platform --platform windows-amd64
python scripts/build_single_exe_release.py stage-runtime --platform windows-amd64 \
  --rocm-backend path/to/ggml-rocm.dll --runtime-dir path/to/windows-runtime

python3 scripts/build_single_exe_release.py stage-platform --platform linux-amd64
python3 scripts/build_single_exe_release.py stage-runtime --platform linux-amd64 \
  --rocm-backend path/to/ggml-rocm.so --runtime-dir path/to/linux-runtime

python3 scripts/build_single_exe_release.py build-ape \
  --manifest .rocm-work/ape-min-release/ape-bootstrap-release-min.json \
  --output .rocm-work/ape-min-release/output/rocm-universal-bootstrap-release.exe \
  --windows-release .rocm-work/ape-min-release/rocm-cli-v0.2.0-windows-amd64.zip \
  --linux-release .rocm-work/ape-min-release/rocm-cli-v0.2.0-linux-amd64.tar.gz \
  --model .rocm-work/ape-real-inputs/Qwen3.5-0.8B-Q8_0.llamafile.exe \
  --windows-rocm-backend .rocm-work/ape-min-release/ggml-rocm.dll \
  --linux-rocm-backend .rocm-work/ape-min-release/ggml-rocm.so \
  --windows-runtime-dir .rocm-work/ape-min-release/windows-runtime \
  --linux-runtime-dir .rocm-work/ape-min-release/linux-runtime \
  --compiler .rocm-work/tools/cosmocc-wsl-elf/bin/cosmocc
```

Print the remaining live matrix without running downloads:

```bash
python scripts/bootstrap_packaging_validation.py --print-live-matrix
```

## Existing Focused Checks

These production tests remain useful companion coverage for the same contract:

```bash
cargo test -p rocmd direct_mcp_call_guard_blocks_mutation_without_explicit_ack
cargo test -p rocmd rocm_command_helper_allows_only_read_only_rocm_commands
cargo test -p rocmd launch_server_rejects_public_bind_without_ack
cargo test -p rocmd supervise_defaults_to_gpu_required_without_cpu_fallback
cargo test -p rocmd install_sdk_forwards_requested_build_date_and_rejects_conflict
cargo test -p rocm --bin rocm assistant_tui_setup_prompts_open_review_cards
cargo test -p rocm --bin rocm chat_tools_mutating_call_opens_cli_approval
```

## Live GPU Matrix

These checks still require real hardware/artifacts and should stay out of
normal CI until dedicated hosts are available.

| Target | Required live validation |
| --- | --- |
| Native Windows | Run the pinned GPU-capable Qwen llamafile with CPU fallback disabled, verify GPU helper use from logs/perf counters, confirm loopback-only bind and OpenAI tools with `--jinja`, keep driver guidance on AMD's official flow, and confirm the bundled Windows llamafile stays below 4 GB. |
| Native Linux | Run the same pinned model/artifact on a supported AMD GPU, verify HIP/GPU execution, exercise Linux driver plan/dry-run/approval rails, and reject raw `sudo`, package-manager, shell, and `--yes` suggestions. |
| WSL | Treat WSL as distinct: verify ROCDXG readiness, Windows driver prerequisite guidance, WSLInterop/APE behavior, loopback serving, and no native DKMS install path. |
| No supported AMD GPU | On a fixture or host with no supported AMD GPU, confirm `rocm doctor` produces the hard error before any mutating install/start action can be proposed or approved. |
| Launcher | With a real signed release archive, verify embedded hash before extraction, preserve dist verifier expectations, run `bin/rocm version`, `bin/rocm doctor`, and `bin/rocm engines list`, then verify reuse, uninstall, and upgrade behavior on Windows, Linux, and WSL. |
| After TheRock setup | Confirm the managed local assistant path remains GPU-required and that no CPU-backed bootstrap helper satisfies normal `rocm serve` or `rocm chat --tools --provider local`. |

Record the exact llamafile release, file size, SHA-256, embedded model,
quantization, chat template, `ggml-rocm.dll`/`ggml-rocm.so` SHA-256 values,
GPU helper behavior, and platform logs in the test run notes for any live pass.

### Native Windows Real Smoke

When the real Qwen llamafile, ROCm backend sidecar, and matching ROCm runtime
libraries are available, run the non-mock smoke:

```powershell
python scripts\bootstrap_real_gpu_smoke.py --skip-build `
  --llamafile .rocm-work\real-bootstrap\Qwen3.5-0.8B-Q8_0.llamafile.exe `
  --rocm-backend .rocm-work\real-bootstrap\ggml-rocm-therock-gfx1201-patched.dll `
  --runtime-dir D:\jam\venv\Lib\site-packages\_rocm_sdk_core\bin `
  --runtime-dir D:\jam\venv\Lib\site-packages\_rocm_sdk_libraries_gfx120X_all\bin
```

The script stages the artifacts under `.rocm-work/tests`, runs
`rocm bootstrap assistant --smoke-stop-after-ready`, requires ROCm GPU evidence
in the saved log, calls the OpenAI chat endpoint, and verifies the child server
stops cleanly. It rejects CPU fallback, missing ROCm backend support, and
invalid GPU code objects.

Current Windows live pass:

```text
model: Qwen3.5-0.8B-Q8_0.llamafile.exe
model sha256: 052D8C0D6EF9809B3BA0DE6BBDBDC92864A9411B13EF76BB974D7E42E00AB6D1
sidecar: ggml-rocm-therock-gfx1201-patched.dll
sidecar sha256: 054293646DB2E5CD7704A624E662649CE77D80BE52C0F2F499D626EA43E5CDE5
GPU: AMD Radeon RX 9070 XT / gfx1201
result: bootstrap real GPU smoke passed; chat response was OK; service stopped
```

Important sidecar note: Mozilla's published `ggml-rocm.dll` loaded but failed
on this GPU because it lacked `gfx1201` code objects. A rebuilt sidecar also
crashed until the Mozilla `llama.cpp.patches` GGML backend ABI patches were
applied. Do not build a Windows llamafile GPU sidecar from unpatched upstream
GGML sources.

## Real P0 Manifest Inputs

When building a real debug or release APE payload, the manifest generation must
pass the downloaded/pinned ROCm sidecars explicitly:

```bash
python scripts/ape_bootstrap_package.py plan \
  --version 0.2.0 \
  --windows-release path/to/rocm-cli-windows-amd64.zip \
  --linux-release path/to/rocm-cli-linux-amd64.tar.gz \
  --model path/to/Qwen3.5-0.8B-Q8_0.llamafile.exe \
  --windows-rocm-backend path/to/ggml-rocm.dll \
  --linux-rocm-backend path/to/ggml-rocm.so \
  --windows-runtime-dependency path/to/amdhip64_7.dll \
  --linux-runtime-dependency path/to/ape-x86_64.elf \
  --output path/to/ape-bootstrap-manifest.json
```

The no-argument launcher path should extract the payload and run
`rocm bootstrap assistant ...` so an interactive terminal opens the embedded
Qwen assistant immediately. The user then chats with the assistant to choose the
TheRock install folder and approve any `rocm install sdk --prefix PATH`
command.

## 2026-06-04 Self-Extracting APE Validation Notes

The historical self-extracting APE proof used one canonical `cosmocc`-built
artifact:

```text
.rocm-work/ape-min-release/output/rocm-universal-bootstrap-release.exe
sha256 fefd783f68395a8eaec2fe9212a247ce5849601099a6b95b6d9c0f4e67ac8c39
```

Do not treat a clang-built Windows launcher and a WSL-built launcher as two
deliverables. Clang is only useful for tiny local launcher self-tests. The
deliverable is the one `cosmocc` APE file above, built once and run unchanged.
When the build runs inside WSL and the requested output ends in `.exe`, the
builder also creates an extensionless hard-link/copy beside it, for example:

```text
.rocm-work/ape-min-release/output/rocm-universal-bootstrap-release
```

Use that extensionless name in WSL-facing instructions.

Windows real smoke test:

```powershell
$ape = ".rocm-work\ape-min-release\output\rocm-universal-bootstrap-release.exe"
$env:ROCM_CLI_APE_ROOT = ".rocm-work\tests\ape-real-windows-root-final"
$extract = & $ape --ape-extract-only
$model = Join-Path $env:ROCM_CLI_APE_ROOT "payload\bootstrap\Qwen3.5-0.8B-Q8_0.llamafile.exe"
& $ape -- bootstrap assistant --llamafile $model --port <free-port> `
  --smoke-stop-after-ready --smoke-prompt "Reply with exactly OK."
```

WSL with normal WSLInterop enabled must use the extensionless artifact name and
the shell trampoline so WSL does not hand the APE's MZ/PE header to Windows:

```bash
ape=/home/jam/rocm-cli-wsl-bootstrap/.rocm-work/ape-min-release/output/rocm-universal-bootstrap-release
root=/home/jam/rocm-cli-wsl-bootstrap/.rocm-work/tests/ape-real-wsl-root-final
export ROCM_CLI_APE_SELF="$ape"
export ROCM_CLI_APE_ROOT="$root"
extract_root=$(sh "$ape" --ape-extract-only)
model="$extract_root/payload/bootstrap/Qwen3.5-0.8B-Q8_0.llamafile.exe"
sh "$ape" -- bootstrap assistant --llamafile "$model" --port <free-port> \
  --smoke-stop-after-ready --smoke-prompt "Reply with exactly OK."
```

Renaming the WSL artifact does not by itself bypass WSLInterop, because WSL
routes by the executable header, not just the `.exe` extension. Direct `./rocm`
execution also works when the WSLInterop binfmt entry is disabled. For
validation only, disable it, run the direct test from ext4, then restart WSL to
restore normal behavior:

```bash
sudo sh -c 'echo -1 > /proc/sys/fs/binfmt_misc/WSLInterop'
ROCM_CLI_APE_ROOT=~/rocm-extract ./rocm -- version
```

Then from Windows:

```powershell
wsl --shutdown
```

Native Linux should follow the direct APE path because it does not have WSL's
Windows interop binfmt handler. Still validate this on a native Linux machine
before release.

Current live pass:

```text
Windows: passed; response was OK; foreground stayed clean.
WSL: passed via `sh <ape>`; response was OK; foreground stayed clean.
Native Linux: still requires a physical-host live validation pass.
```
