# P0 Bootstrap And Single-File Packaging Spike

Status update, 2026-06-05: this embedded-llamafile/APE bootstrap direction is
superseded for the active `jam/updates` branch. Bootstrap now uses deterministic
first-time setup UI, and the active single-file release artifact is the
standalone `rocm`/`rocm.exe` binary itself. Keep the notes below as historical
research unless the user explicitly reopens the embedded assistant path.

This note captures the active P0 spike on the `jam/updates_exe_ape` worktree.
The goal is a Cosmopolitan/APE-inspired single-exe bootstrap artifact that can
start a tiny GPU-required Qwen assistant before TheRock is installed, open a
conversation-first TUI immediately, then use approval-gated `rocm` commands to
help with setup.

The spike has two coupled requirements:

1. Embed a pinned Qwen 0.8B-class bootstrap model, currently targeting
   `Qwen3.5-0.8B-Q8_0.llamafile`, plus the platform ROCm sidecar libraries
   needed by llamafile GPU offload.
2. Carry the normal rocm-cli release payloads inside one universal AMD64 APE
   launcher, then extract and delegate to the platform's real `bin/rocm`.

The existing signed zip/tar release bundles remain intact until the APE path
proves equal trust, observability, and live GPU behavior. The spike still does
not weaken the GPU-required serving policy: if AMD GPU execution cannot be
proven, bootstrap assistant startup must fail loudly.

## Current rocm-cli Constraints

- First-run setup already owns the TheRock install decision in the TUI and
  keeps `rocm install sdk` behind an approval card with live foreground output.
- `rocmd` already exposes a structured MCP-style tool surface. Read-only tools
  can run directly; mutating tools such as `install_sdk`, `install_engine`,
  `launch_server`, and `stop_server` require explicit mutation approval.
- The assistant tool contract is argv-style `rocm` commands, not shell text.
  `rocmd` rejects arbitrary or mutating `rocm_command` calls that do not go
  through the approval UI.
- The assistant-first UI should mirror Codex-rs patterns where practical:
  conversation first, bottom composer, compact running status, scrollable
  history, and modal approval/progress cards instead of action-menu-first
  screens.
- Current release packaging is a signed archive containing sibling binaries:
  `rocm`, `rocmd`, engine adapters, vendored `rocm-codex`, README/LICENSE, and
  the platform installer. Installers verify checksums and can verify detached
  signatures before activation.
- TheRock installs are not just one executable. They create managed Python
  environments, localized pip caches, runtime manifests, ROCm SDK libraries,
  PyTorch packages, and engine-specific environments.

## Upstream Facts Checked

- `llamafile` combines `llama.cpp` with Cosmopolitan Libc into a single-file
  executable that runs locally with no installation on many platforms. Mozilla's
  README currently shows `llamafile v0.10.3` as the latest GitHub release on
  June 2, 2026.
  Source: https://github.com/mozilla-ai/llamafile
- Mozilla's docs describe the supported OS set as Linux, macOS, Windows,
  FreeBSD, NetBSD, and OpenBSD. Windows is AMD64-only, requires adding an
  `.exe` extension, and bundled llamafiles above 4 GB do not run on Windows.
  Source: https://mozilla-ai.github.io/llamafile/support/
- Llamafile supports server mode and recommends `--jinja` for agentic uses.
  Source: https://mozilla-ai.github.io/llamafile/running_llamafile/
- Llamafile creation bundles the executable, GGUF weights, and default args
  into an APE file that can also be inspected as ZIP/ZIP64 content.
  Source: https://mozilla-ai.github.io/llamafile/creating_llamafiles/
- Llamafile technical docs call out a key packaging limit: Cosmopolitan's static
  linking model cannot directly statically link GPU support. Llamafile handles
  GPU by compiling or loading platform-specific dynamic helpers at runtime.
  Source: https://mozilla-ai.github.io/llamafile/technical_details/
- Mozilla's quick-start example currently uses
  `Qwen3.5-0.8B-Q8_0.llamafile`, described there as the smallest prebuilt
  llamafile and therefore most likely to work out of the box.
  Source: https://github.com/mozilla-ai/llamafile
- Llamafile's supported-systems docs say AMD/NVIDIA GPU use may require HIP/CUDA
  SDKs and that if GPU support cannot be built or linked, llamafile falls back
  to CPU inference. The product priority for this investigation is the GPU
  path; the bootstrap design must work with CPU fallback disabled. A pinned
  0.8B-class Qwen CPU fallback is only a low-priority contingency if it proves
  tool-safe enough, and it must still not become a serving fallback.
  Source: https://mozilla-ai.github.io/llamafile/support/
- AMD's official driver page is the authoritative Windows driver download
  entry point and also links Linux driver resources.
  Source: https://www.amd.com/en/support/download/drivers.html
- Current ROCm Linux quick-start docs recommend package-manager installation
  and show `amdgpu-dkms` as the kernel driver package after installing the AMD
  package repository metadata.
  Source: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html
- AMD's WSL guidance describes ROCDXG as the Linux-side user-mode bridge to the
  Windows display driver stack; the setup path starts with the Windows Adrenalin
  driver, WSL, then the `librocdxg` quickstart.
  Source: https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/wsl/howto_wsl.html
- `llama-server` has OpenAI-compatible chat/completions/embedding routes and
  function calling/tool use support when launched with `--jinja`; tool support
  quality still depends on model, template, and quantization.
  Sources:
  https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/server/README.md
  and https://raw.githubusercontent.com/ggml-org/llama.cpp/master/docs/function-calling.md
- Cosmopolitan Libc produces Actually Portable Executables for C/C++ programs
  across Linux, macOS, Windows, BSDs, and BIOS without a VM. Its own README
  warns that WSL can be unsafe for APE programs unless WSLInterop/binfmt
  behavior is handled deliberately.
  Source: https://raw.githubusercontent.com/jart/cosmopolitan/master/README.md
- AMD's TheRock release docs describe Python wheels, tarballs, and native
  packages. Wheels can pull compatible ROCm packages automatically for PyTorch,
  while tarballs are raw extracted files and native packages are still described
  as primarily development/testing, with unsigned native package notes in the
  release page.
  Source: https://github.com/ROCm/TheRock/blob/main/RELEASES.md

## Idea 1: Pre-TheRock Qwen Llamafile Assistant

### Feasible Shape

The safest version is a separate bootstrap assistant artifact, not a replacement
for the built-in GPU assistant:

1. User installs the normal rocm-cli bundle or installer first.
2. rocm-cli runs `rocm doctor` or the same deterministic doctor GPU summary
   before launching the helper.
3. If doctor reports absolutely no supported AMD GPU, rocm-cli reports a hard
   error and must not offer install/start actions that imply the machine can run
   ROCm GPU workflows. A later UX spike can decide whether the assistant starts
   only to explain that error, but it cannot propose a mutating install.
4. If setup is incomplete and the host is not ruled out by doctor, opening the
   single exe should start `rocm bootstrap assistant` and enter the
   assistant-first TUI automatically.
5. rocm-cli launches a signed, pinned, GPU-capable 0.8B-class Qwen `llamafile`
   locally with a system prompt limited to setup support. The current upstream
   candidate to validate is `Qwen3.5-0.8B-Q8_0.llamafile`, because Mozilla
   publishes it as the smallest quick-start llamafile, but the acceptance target
   is GPU execution.
6. The bootstrap must still work when CPU fallback is disabled. If the GPU
   llamafile cannot launch because driver/HIP prerequisites are missing, the
   first-run path should fall back to deterministic rocm-cli doctor and driver
   guidance, not to CPU inference. CPU inference is a low-priority contingency
   only after validation proves the pinned model can use the required ROCm tool
   calls reliably enough for setup. The UI must label it as a setup helper, not
   local AI serving.
7. The assistant must ask where the TheRock/ROCm Python folder should live
   before any install action. If the user provides a path, the eventual command
   must preserve it as `rocm install sdk ... --prefix PATH`; if not, the
   guided folder picker should collect it. There is no hidden default folder in
   the assistant path.
8. The model can call only the existing rocm-cli tool facade:
   `doctor`, `model`, `update_check`, `install_sdk_dry_run`, read-only driver
   plans/guidance, WSL ROCDXG status/guidance, and approved install requests.
9. rocm-cli owns validation, approval rendering, execution, and live output.
   The llamafile process never receives shell access and never executes
   commands itself.
10. After TheRock is ready, bootstrap should offer to install ROCm CLI itself
    into a folder the user chooses. The implementation command is
    `rocm bootstrap install-cli --target <folder>`. Only after that succeeds
    should the UI ask whether to add the folder to PATH, using
    `--add-to-path` only when the user explicitly says yes.
11. After TheRock is ready, the embedded Qwen helper may remain registered as
   the general local assistant when it is the ready service. A later UX pass can
   offer the larger managed GPU `qwen` assistant as an upgrade, but the
   bootstrap assistant must remain useful immediately after setup.

This means the bootstrap helper is GPU-first and CPU-independent by design. A
CPU-backed helper can be investigated later only as a temporary setup
contingency. It must not satisfy `rocm serve`, must not replace the managed GPU
`qwen` assistant after setup, and must not hide any broken ROCm GPU path.

### Driver Guidance Boundary

The bootstrap assistant may guide users toward driver readiness, but rocm-cli
must own source selection and command execution:

- Windows: validate the detected driver state and point users to AMD's official
  Windows driver download/install flow. Do not run a Windows driver installer or
  invent direct download URLs unless a later source-policy spike adds signed,
  verified driver metadata.
- Native Linux: use the existing `rocm install driver` plan/dry-run surfaces and
  approval-gated execution where supported. Do not let the model emit raw
  package-manager, `sudo`, or `--yes` commands.
- WSL: guide the ROCDXG path and Windows WSL-capable AMD driver prerequisite.
  Treat WSL driver readiness as ROCDXG status, not as a native Linux DKMS
  driver install.

### Integration Points Already Present

- `rocmd` tool schema:
  - `rocm_command` for read-only argv commands.
  - `install_sdk_dry_run` for preview.
  - `install_sdk` for mutation.
- `build_install_sdk_args` already rejects conflicting `version` and
  `build_date` fields and requires `allow_system_prefix=true` for system paths.
- `ensure_direct_mcp_call_allowed` already classifies `install_sdk` and other
  mutating tools as approval-required.
- TUI chat result handling already turns mutating assistant suggestions into
  approval cards, including TheRock `--build-date` and `--version` installs.

### Open Questions

- GPU bootstrap viability: Windows `gfx1201` now has a real smoke pass with the
  Qwen llamafile, a Mozilla-patched TheRock-built `ggml-rocm.dll`, and staged
  TheRock runtime libraries. Native Linux and WSL still need the same proof.
  The bootstrap must keep a deterministic non-LLM setup path for states where
  the GPU assistant cannot start.
- Model quality: A 0.8B GGUF may be good enough to inspect first and request
  simple rocm-cli actions, but that must be tested, not assumed.
- Model choice: Mozilla's current smallest quick-start llamafile is
  `Qwen3.5-0.8B-Q8_0.llamafile`. Upstream llama.cpp docs list native tool-call
  formats for Qwen 2.5 and generic/Jinja support more broadly, but do not by
  themselves prove this Qwen3.5 0.8B llamafile can use every rocm-cli tool
  reliably. It needs a dedicated validation matrix.
- Windows size limit: any bundled Windows llamafile must stay below 4 GB.
  External GGUF weights avoid the limit but weaken the "single file" UX.
- ROCm sidecar provenance: Mozilla's published Windows `ggml-rocm.dll` is not
  sufficient for this RX 9070 XT because it lacks `gfx1201` code objects. The
  validated Windows path is a rebuilt sidecar from llamafile v0.10.3 with
  Mozilla's `GGML_CALL`/`ms_abi` GGML backend patches applied and
  `--offload-arch=gfx1201`. Linux/WSL sidecars still need pinned build and
  provenance work.
- Trust and update strategy: the llamafile binary and model weights need the
  same trust story as rocm-cli releases: version pin, checksum, detached or
  platform signature, and provenance for model weights and embedded changes.
- Endpoint safety: server mode must bind to loopback only. Upstream examples
  use `--host 0.0.0.0` for network access, but rocm-cli should not expose a
  bootstrap assistant publicly by default.
- CPU bootstrap boundary: CPU inference is low priority and cannot be required
  for the bootstrap feature to work. If it is added later, it can be allowed
  only for the bootstrap assistant with the pinned and validated 0.8B-class Qwen
  model. GPU-required serving remains GPU-required, and broken GPU support must
  still fail loudly.
- Driver guidance boundary: Windows driver help should remain validate/manual
  guidance through AMD's official driver page; Linux driver work should stay on
  the existing rocm-cli driver plan/approval rails; WSL should stay on ROCDXG
  guidance. Automatic driver downloads need a production source policy first.
- No-supported-GPU boundary: if doctor reports no supported AMD GPU, rocm-cli
  must report that directly and must not let the model talk the user into an
  install/start flow that cannot work.
- WSL behavior: Mozilla docs present WSL as useful for files above 4 GB, while
  Cosmopolitan's README warns about WSLInterop/binfmt behavior. Treat WSL as an
  explicit test target before recommending this flow there.
- Licensing: model weights, llama.cpp, llamafile changes, and any embedded
  notice files must be reviewed before redistribution.

### Minimal Validation Spike

No production UX should be added until these pass:

- Download `Qwen3.5-0.8B-Q8_0.llamafile` or build an equivalent 0.8B-class Qwen
  GPU-capable llamafile from a pinned release and record exact file size,
  checksum, embedded model, quantization, chat template, GPU helper behavior,
  and llamafile revision.
- Run on native Windows, native Linux, and WSL from an isolated rocm-cli config.
  Current native Windows pass: `scripts/bootstrap_real_gpu_smoke.py` with
  `Qwen3.5-0.8B-Q8_0.llamafile.exe` SHA-256
  `052D8C0D6EF9809B3BA0DE6BBDBDC92864A9411B13EF76BB974D7E42E00AB6D1` and
  `ggml-rocm-therock-gfx1201-patched.dll` SHA-256
  `054293646DB2E5CD7704A624E662649CE77D80BE52C0F2F499D626EA43E5CDE5`.
- Run with CPU fallback disabled where llamafile allows it, or otherwise verify
  from logs/perf counters that the GPU path is actually used. The feature should
  still complete bootstrap validation without relying on a CPU assistant.
- Run once on a no-supported-GPU fixture or host and confirm `rocm doctor`
  causes a hard "no supported AMD GPU" error before any mutating install/start
  action can be proposed or approved.
- Start server mode on `127.0.0.1` with `--jinja` and verify
  `/v1/chat/completions` accepts OpenAI-style tools.
- Use the existing rocm-cli tool schema with prompts for every supported
  bootstrap command, including:
  - "Is ROCm installed?"
  - "Which AMD GPU is detected?"
  - "Do I need a Windows AMD driver?"
  - "Guide me to the right Windows driver."
  - "Do I need Linux DKMS driver setup?"
  - "Show the Linux driver plan without installing."
  - "What does WSL need for ROCDXG?"
  - "Show me what TheRock install would do."
  - "Install TheRock from build date YYYY-MM-DD."
  - "Install to this custom folder."
  - "Install this exact TheRock version."
  - "Uninstall the managed runtime."
  - "Change the default engine."
  - "Turn telemetry off."
  - "Install/start/show logs for ComfyUI."
  - "Install/start/show logs for llama.cpp."
- Confirm read-only calls run without approval and mutating calls create the
  same approval cards and live-output surfaces as the current local assistant.
- Confirm Windows driver responses point to AMD's official driver flow without
  executing installers, Linux driver responses use the existing driver
  plan/dry-run/approval path, and WSL responses use ROCDXG guidance.
- Confirm the model does not hallucinate unsupported shell, PowerShell, package
  manager, or `--yes` flows when asked adversarially.
- Confirm bad prompts cannot cause shell execution, public bind, `--yes`, or
  unsupported package-manager commands.
- Confirm after TheRock setup that GPU-required assistant serving is still
  required for normal `rocm chat --tools --provider local`.

## P0 APE Packaging Contract

`scripts/ape_bootstrap_package.py` is the executable contract for the current
P0 branch. It validates and stages the payload that a Cosmopolitan/APE builder
must embed.

The required manifest shape is:

- `ape.kind = cosmopolitan_ape`
- `ape.target = universal-amd64`
- Windows and Linux rocm-cli release archives are both embedded under
  `payload/releases/`
- the bootstrap model is an embedded Qwen 0.8B-class `.llamafile` under
  `payload/bootstrap/`
- startup uses `--server`, `--jinja`, `--host 127.0.0.1`, `--gpu amd`, and
  `-ngl` or `--n-gpu-layers` at `999+`
- CPU fallback flags, public binds, `--gpu disable`, and `-ngl 0` are rejected
- the launcher must run `rocm bootstrap assistant` with
  `--device gpu_required`
- GPU proof must include both startup log evidence and a health/backend check
- the total Windows executable estimate must stay below the Windows llamafile
  size cap

Run the offline contract tests with:

```bash
python scripts/ape_bootstrap_package.py self-test
```

Create a manifest from real local artifacts with:

```bash
python scripts/ape_bootstrap_package.py plan \
  --version <version> \
  --windows-release dist/rocm-cli-<version>-windows-amd64.zip \
  --linux-release dist/rocm-cli-<version>-linux-amd64.tar.gz \
  --model artifacts/Qwen3.5-0.8B-Q8_0.llamafile \
  --output .rocm-work/ape/ape-bootstrap.json
```

Stage an uncompressed ZIP payload for a later `zipalign`/APE builder with:

```bash
python scripts/ape_bootstrap_package.py stage \
  --manifest .rocm-work/ape/ape-bootstrap.json \
  --output .rocm-work/ape/rocm-cli-ape-bootstrap-payload.zip
```

This would reduce "download archive, unpack, run installer" friction without
pretending ROCm, HIP libraries, Python wheels, or model artifacts are truly
inside one universal executable.

### Cosmopolitan-Inspired But Not Assumed

Cosmopolitan is attractive because its APE format can carry executable content
and ZIP data in one file. For rocm-cli, this branch treats Cosmopolitan as the
launcher layer, not as a rewrite of the Rust app:

- rocm-cli is Rust plus several Rust engine adapters, not a small C/C++ tool.
- TheRock and HIP libraries are dynamically loaded platform-specific assets.
- Python environments and wheels are intentionally managed on disk.
- Windows Authenticode signing, AV reputation, installer SmartScreen behavior,
  and enterprise policy are different from detached archive signatures.
- A universal APE still extracts native platform binaries before running them,
  especially for Rust binaries and engine adapters.

### Non-Goals

- Do not embed TheRock wheels or managed Python into the rocm-cli executable.
- Do not embed model weights into the normal rocm-cli tar/zip release bundles.
  The APE bootstrap artifact is the exception: it must embed the pinned Qwen
  0.8B bootstrap model so the first-run assistant is available before TheRock.
- Do not make CPU inference a fallback for GPU-required serving. The only CPU
  exception under investigation is a low-priority pinned, validated 0.8B-class
  Qwen pre-TheRock setup contingency; the primary bootstrap path must work
  without it.
- Do not replace release signatures with "because it is one file" trust.
- Do not remove sibling binary support; engine adapters and vendored Codex need
  a stable local path for process launch and diagnostics.

### Minimal Packaging Spike

The useful proof is smaller than a production APE release:

1. Build the existing release archive with current scripts.
2. Build or download the pinned Qwen 0.8B-class llamafile and record exact
   provenance.
3. Use `scripts/ape_bootstrap_package.py plan` and `stage` to create the
   validated uncompressed payload.
4. Build the Cosmopolitan/APE launcher around that payload.
5. On first run, extract into `.rocm/launcher/<version>/<platform>/`.
6. Verify embedded hashes before extraction and preserve the current dist
   verifier expectations for the archive payload.
7. Run `rocm bootstrap assistant` with the embedded model, loopback-only bind,
   OpenAI/Jinja tool mode, AMD GPU-required startup flags, and live GPU proof.
8. Run `bin/rocm version`, `bin/rocm doctor`, and `bin/rocm engines list`.
9. Re-run and confirm it reuses the extracted version instead of rewriting.
10. Test uninstall and upgrade behavior against existing installer manifests.
11. Repeat on Windows, WSL, and native Linux.

### Experimental Prototype Path

`scripts/experimental_launcher.py` is a runnable helper for this spike. It can
use an existing release archive plus manifest, or generate a Python-based
single-file launcher with the archive embedded as a base64 payload. On run, it
verifies the archive SHA-256, extracts into
`.rocm/launcher/<version>/<platform>/`, reuses an activation that matches the
same archive hash, and delegates argv directly to the extracted `bin/rocm`.

This remains experimental packaging code only. It does not replace the signed
zip/tar installers, embed TheRock runtimes or model artifacts, or change any
serving policy.

### Cosmopolitan APE Builder Path

`scripts/setup-cosmocc.sh` downloads the workspace-local Cosmopolitan compiler
under `.rocm-work/tools/cosmocc`. On WSL it also creates
`.rocm-work/tools/cosmocc-wsl-elf`, an ELF-converted copy of the toolchain, so
nested `cosmocc` helper binaries do not get intercepted by WSLInterop. Use the
returned compiler path from that script for production APE builds.

`scripts/build_ape_bootstrap.py` is the current C launcher builder. It expands
validated Windows and Linux release archives into one uncompressed ZIP payload,
compiles `launcher/ape_bootstrap_launcher.c`, appends the payload, verifies
CRC-protected extraction/reuse, and delegates to the extracted platform binary.
The production `build` command requires a compiler that looks like
Cosmopolitan `cosmocc`/APE; non-Cosmopolitan `clang`/`gcc` builds are accepted
only by `self-test` or by an explicit `--allow-local-compiler` development
override.

The launcher must make OS choices at runtime, not with `_WIN32` alone, because
a Cosmopolitan APE is one binary. The prototype uses Cosmopolitan `IsWindows()`
when built by `cosmocc`, chooses `windows-amd64/bin/rocm.exe` on Windows and
`linux-amd64/bin/rocm` on Linux/WSL, and normalizes Windows drive paths before
creating extraction directories.

Current offline proof commands:

```bash
scripts/setup-cosmocc.sh
python3 scripts/build_ape_bootstrap.py self-test \
  --compiler .rocm-work/tools/cosmocc-wsl-elf/bin/cosmocc
```

The WSL setup path is tested on hosts without `unzip`; the Python extraction
fallback restores archive permission bits so `ape-x86_64.elf`, `assimilate`,
and nested compiler helpers stay executable.

The same generated `.exe` has been smoke-tested directly on Windows, through
`sh ./rocm` on WSL with WSLInterop enabled, directly on WSL after temporarily
disabling WSLInterop, and through `ape-x86_64.elf` on WSL. The current debug
artifact embeds real debug rocm-cli platform payloads, the pinned Qwen
llamafile artifact, and both ROCm llamafile sidecars:

```text
.rocm-work/ape-real-inputs/rocm-debug-bootstrap-ape.exe
sha256 5376e256ab2086afa821fcfa50a9f4e512c7ed383566a04bc97108a71e1079fa
```

Production still needs signed release rocm-cli archives, native Linux hardware
validation, Windows signing/AV validation, and live GPU bootstrap proof.

The remaining APE builder work should answer one question: can a C/C++ APE
launcher reliably choose the host platform, read embedded ZIP content, extract
signed rocm-cli binaries plus the embedded Qwen bootstrap model, start
GPU-required assistant mode, and then exec the normal `rocm` binary without WSL,
AV, or signing surprises?

### Vendored Codex Follow-Up

The repo vendors `third_party/openai-codex/codex-rs`. Once the P0 APE path is
stable, evaluate using that vendored Codex implementation to run Codex itself
and to reuse its mature chat, approval, tool-call, and TUI/session primitives
for locally hosted ROCm assistant models. Prefer this over continuing to grow a
bespoke mini-Codex UI if the integration surface is practical.

## Recommendation

- Proceed with the APE branch as an isolated P0 spike.
- Keep the current archive installers as production until the APE path passes
  live Windows, WSL, and native Linux validation.
- Keep the first implementation as a bootstrap helper and launcher, not a
  rewrite of rocm-cli internals.
- Treat a bundled llamafile assistant as acceptable only before TheRock setup,
  only if GPU execution is proven, only if the pinned 0.8B-class Qwen model
  validates across the rocm-cli bootstrap tool suite, and only through
  rocm-cli's existing approval-gated tool executor.
- Keep the current signed archive installers as the production release path
  until the launcher proves at least equal trust, observability, and rollback
  behavior.
