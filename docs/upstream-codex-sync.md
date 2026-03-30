# Upstream Codex Sync Notes

## Current Pin

- upstream repo: `https://github.com/openai/codex`
- vendored subtree: `third_party/openai-codex/codex-rs`
- pinned commit: `5037a2d19924f2e49490453ab2a913da938afbe5`
- vendored on: `2026-03-28`

## Imported Files

- `third_party/openai-codex/codex-rs/`
- `third_party/openai-codex/OPENAI-CODEX-LICENSE`
- `third_party/openai-codex/OPENAI-CODEX-NOTICE`

## Current Local Integration State

- upstream Rust workspace vendored into `rocm-cli`
- no source-level modifications inside the vendored tree yet
- the project build/package flow prebuilds the vendored `codex` binary as `rocm-codex`
- `rocm --experimental-codex-tui` launches only the prebuilt shipped binary

Current internal build command:

```bash
./scripts/build-vendored-codex.sh release
```

This preserves the upstream ChatGPT sign-in flow, including the no-key onboarding path, while avoiding runtime compilation during interactive launch.

## Local Patch Categories

The intent is to keep local patches in the vendored tree small and attributable.

Expected patch categories:

- ROCm branding
- provider policy integration
- `rocmd` backend bridge
- ROCm-specific sidebar/status extensions
- disabled upstream features that are irrelevant for `rocm-cli`

## Sync Procedure

1. clone or fetch `openai/codex`
2. pick a target upstream commit SHA
3. replace the vendored `codex-rs` subtree with that snapshot
4. update this file with the new SHA and sync date
5. rebuild the vendored binary with `./scripts/build-vendored-codex.sh release`
6. rerun `rocm --experimental-codex-tui`
7. record any new local patch requirements

## Notes

- `rocm-cli` does not currently add the vendored workspace to its own Cargo workspace.
- the vendored tree is intentionally isolated so root `cargo check` for `rocm-cli` remains focused on native crates.
- a later step may add dedicated CI jobs for the vendored workspace once the bridge and wrapper stabilize.
