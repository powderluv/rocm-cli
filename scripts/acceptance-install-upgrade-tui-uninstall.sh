#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "missing required command: $1" >&2
    exit 1
  }
}

fail() {
  echo "acceptance failed: $*" >&2
  exit 1
}

assert_file() {
  local path="$1"
  [[ -f "${path}" ]] || fail "expected file: ${path}"
}

assert_missing() {
  local path="$1"
  [[ ! -e "${path}" ]] || fail "expected path to be removed: ${path}"
}

need_cmd cargo
need_cmd script
need_cmd mktemp
need_cmd grep

TMP_ROOT="$(mktemp -d)"
cleanup() {
  rm -rf "${TMP_ROOT}"
}
trap cleanup EXIT INT TERM

DIST_NAME="rocm-cli-linux-amd64"
DIST_DIR="${TMP_ROOT}/dist"
INSTALL_DIR="${TMP_ROOT}/install/bin"
XDG_CONFIG_HOME="${TMP_ROOT}/xdg/config"
XDG_DATA_HOME="${TMP_ROOT}/xdg/data"
XDG_CACHE_HOME="${TMP_ROOT}/xdg/cache"
DOWNLOAD_BASE="file://${DIST_DIR}"
TUI_LOG="${TMP_ROOT}/tui.log"
INSTALL_LOG_1="${TMP_ROOT}/install-1.log"
INSTALL_LOG_2="${TMP_ROOT}/install-2.log"
UNINSTALL_LOG="${TMP_ROOT}/uninstall.log"
CONFIG_FILE="${XDG_CONFIG_HOME}/rocm-cli/config.json"

echo "acceptance: build release binaries"
(cd "${REPO_ROOT}" && cargo build --release -p rocm -p rocmd -p rocm-engine-pytorch)

echo "acceptance: package local release bundle"
(cd "${REPO_ROOT}" && ./scripts/package-linux-release.sh "${DIST_NAME}" "${DIST_DIR}")

run_installer() {
  (
    cd "${REPO_ROOT}"
    ROCM_CLI_DOWNLOAD_BASE="${DOWNLOAD_BASE}" \
    ROCM_CLI_INSTALL_DIR="${INSTALL_DIR}" \
    sh ./install.sh release
  )
}

echo "acceptance: first install"
run_installer | tee "${INSTALL_LOG_1}"
assert_file "${INSTALL_DIR}/rocm"
assert_file "${INSTALL_DIR}/rocmd"
assert_file "${INSTALL_DIR}/rocm-engine-pytorch"
assert_file "${INSTALL_DIR}/.rocm-cli-manifest"

echo "acceptance: simulate stale prior install entry and reinstall"
echo "stale" > "${INSTALL_DIR}/rocm-engine-stale"
echo "${INSTALL_DIR}/rocm-engine-stale" >> "${INSTALL_DIR}/.rocm-cli-manifest"
run_installer | tee "${INSTALL_LOG_2}"
assert_missing "${INSTALL_DIR}/rocm-engine-stale"
assert_file "${INSTALL_DIR}/.rocm-cli-manifest"
grep -q "removing previous rocm-cli install" "${INSTALL_LOG_2}" \
  || fail "installer did not report removal of previous install"

echo "acceptance: drive the TUI through a pseudo-terminal"
(
  sleep 1
  printf 'config set-default-engine pytorch\r'
  sleep 1
  printf 'config show\r'
  sleep 1
  printf 'install sdk --channel nightly\r'
  sleep 1
  printf '/quit\r'
) | script -qefc \
  "env XDG_CONFIG_HOME='${XDG_CONFIG_HOME}' XDG_DATA_HOME='${XDG_DATA_HOME}' XDG_CACHE_HOME='${XDG_CACHE_HOME}' '${INSTALL_DIR}/rocm' chat --provider openai" \
  "${TUI_LOG}"

assert_file "${CONFIG_FILE}"
grep -q '"default_engine"[[:space:]]*:[[:space:]]*"pytorch"' "${CONFIG_FILE}" \
  || fail "TUI did not persist the expected default engine"
assert_file "${TUI_LOG}"

echo "acceptance: uninstall from the installed binary"
env \
  XDG_CONFIG_HOME="${XDG_CONFIG_HOME}" \
  XDG_DATA_HOME="${XDG_DATA_HOME}" \
  XDG_CACHE_HOME="${XDG_CACHE_HOME}" \
  "${INSTALL_DIR}/rocm" uninstall --yes | tee "${UNINSTALL_LOG}"

assert_missing "${INSTALL_DIR}/rocm"
assert_missing "${INSTALL_DIR}/rocmd"
assert_missing "${INSTALL_DIR}/rocm-engine-pytorch"
assert_missing "${INSTALL_DIR}/.rocm-cli-manifest"
assert_missing "${XDG_CONFIG_HOME}/rocm-cli"
assert_missing "${XDG_DATA_HOME}/rocm-cli"
assert_missing "${XDG_CACHE_HOME}/rocm-cli"

echo "acceptance: ok"
