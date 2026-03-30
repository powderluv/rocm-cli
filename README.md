# rocm-cli

`rocm-cli` is the ROCm AI Command Center CLI for AMD systems.

Install on Linux x86_64:

```bash
curl -fsSL https://raw.githubusercontent.com/powderluv/rocm-cli/main/install.sh | sh
```

The installer updates your shell profile to add the install directory to `PATH` when needed.

Re-running the installer replaces the previous `rocm-cli` binaries from the same install directory.

Install the latest nightly on Linux x86_64:

```bash
curl -fsSL https://raw.githubusercontent.com/powderluv/rocm-cli/main/install.sh | sh -s -- nightly
```

Current repository status:
- interactive `rocm` TUI plus non-interactive CLI commands for `doctor`, `serve`, `engines`, `install`, `update`, `automations`, `daemon`, and `uninstall`
- `rocmd` supervisor with managed service supervision and persisted automation/watcher runtime state
- first-party `pytorch` engine with managed venv installs, TheRock PyTorch wheel resolution, and OpenAI-compatible local serving
- TheRock SDK resolver for `pip` and tarball installs, plus managed runtime update checks
- vendored Codex TUI scaffold with a packaged `rocm-codex` binary for experimental interactive launch

Planned product shape:
- `rocm` chat-first TUI with deeper inline tool execution and approvals
- TheRock-managed runtime installs with `pip` venvs by default and tarballs as an explicit option
- native automations and watchers with broader contained sandbox execution
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

This is an early implementation, not a production release.
