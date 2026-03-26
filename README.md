# rocm-cli

`rocm-cli` is the ROCm AI Command Center CLI for AMD systems.

Install on Linux x86_64:

```bash
curl -fsSL https://raw.githubusercontent.com/powderluv/rocm-cli/main/install.sh | sh
```

Re-running the installer replaces the previous `rocm-cli` binaries from the same install directory.

Install the latest nightly on Linux x86_64:

```bash
curl -fsSL https://raw.githubusercontent.com/powderluv/rocm-cli/main/install.sh | sh -s -- nightly
```

Current repository status:
- Rust workspace scaffold for `rocm`, `rocmd`, shared core types, and engine protocol
- first-party `pytorch` engine crate intended to be the default native Windows local serving path
- placeholder CLI flows for `doctor`, `serve`, `engines`, `install`, and daemon lifecycle
- initial plugin protocol structs shared across the workspace

Planned product shape:
- `rocm` chat-first TUI with Codex-like plan and tool execution views
- TheRock-managed runtime installs with `pip` venvs by default and tarballs as an explicit option
- native automations and watchers with contained sandbox execution
- pluggable serving engines including `pytorch`, `llama.cpp`, `vllm`, `sglang`, and `atom`

Workspace layout:
- `apps/rocm`: main CLI and future TUI
- `apps/rocmd`: local supervisor and daemon entrypoint
- `crates/rocm-core`: app paths, host summary, shared defaults
- `crates/rocm-engine-protocol`: shared engine request and response types
- `engines/pytorch`: first-party PyTorch serving engine

Planning docs:
- `plans/rocm-cli-implementation-plan.md`
- `plans/rocm-cli-pytorch-engine-spec.md`

This is the initial implementation scaffold, not a production release.
