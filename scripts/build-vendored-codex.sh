#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 2 ]]; then
  echo "usage: $0 [debug|release] [target-triple]" >&2
  exit 1
fi

PROFILE="${1:-release}"
TARGET_TRIPLE="${2:-}"

case "${PROFILE}" in
  debug|release) ;;
  *)
    echo "invalid profile: ${PROFILE} (expected debug or release)" >&2
    exit 1
    ;;
esac

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "missing required command: $1" >&2
    exit 1
  }
}

need_cmd cargo

if [[ "$(uname -s)" == "Linux" ]]; then
  need_cmd pkg-config
  if ! pkg-config --exists libcap; then
    cat >&2 <<'EOF'
vendored Codex build prerequisites are missing on this Linux host.

Install `pkg-config` plus the `libcap` development package, then rerun the build.
Examples:
  Debian/Ubuntu: sudo apt install pkg-config libcap-dev
  Fedora/RHEL:  sudo dnf install pkgconf-pkg-config libcap-devel
EOF
    exit 1
  fi
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CODEX_MANIFEST="${REPO_ROOT}/third_party/openai-codex/codex-rs/Cargo.toml"

if [[ ! -f "${CODEX_MANIFEST}" ]]; then
  echo "vendored Codex manifest not found: ${CODEX_MANIFEST}" >&2
  exit 1
fi

echo "building vendored Codex TUI"
echo "  manifest: ${CODEX_MANIFEST}"
echo "  profile: ${PROFILE}"
if [[ -n "${TARGET_TRIPLE}" ]]; then
  echo "  target: ${TARGET_TRIPLE}"
fi

BUILD_ARGS=(build --manifest-path "${CODEX_MANIFEST}" -p codex-cli --bin codex)
if [[ "${PROFILE}" == "release" ]]; then
  BUILD_ARGS+=(--release)
fi
if [[ -n "${TARGET_TRIPLE}" ]]; then
  BUILD_ARGS+=(--target "${TARGET_TRIPLE}")
fi

(cd "${REPO_ROOT}" && cargo "${BUILD_ARGS[@]}")

CODEX_TARGET_DIR="${REPO_ROOT}/third_party/openai-codex/codex-rs/target"
ROCM_TARGET_DIR="${REPO_ROOT}/target"

if [[ -n "${TARGET_TRIPLE}" ]]; then
  CODEX_BINARY="${CODEX_TARGET_DIR}/${TARGET_TRIPLE}/${PROFILE}/codex"
  ROCM_PROFILE_DIR="${ROCM_TARGET_DIR}/${TARGET_TRIPLE}/${PROFILE}"
else
  CODEX_BINARY="${CODEX_TARGET_DIR}/${PROFILE}/codex"
  ROCM_PROFILE_DIR="${ROCM_TARGET_DIR}/${PROFILE}"
fi

if [[ ! -x "${CODEX_BINARY}" ]]; then
  echo "vendored Codex binary not found after build: ${CODEX_BINARY}" >&2
  exit 1
fi

mkdir -p "${ROCM_PROFILE_DIR}"
install -m 0755 "${CODEX_BINARY}" "${ROCM_PROFILE_DIR}/rocm-codex"

echo "  installed wrapper binary: ${ROCM_PROFILE_DIR}/rocm-codex"
