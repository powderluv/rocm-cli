#!/bin/sh
set -eu

REPO="${ROCM_CLI_GITHUB_REPO:-powderluv/rocm-cli}"
CHANNEL="${1:-release}"
INSTALL_DIR="${ROCM_CLI_INSTALL_DIR:-$HOME/.local/bin}"

fail() {
  echo "rocm-cli installer: $*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || fail "missing required command: $1"
}

sha256_file() {
  file="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$file" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$file" | awk '{print $1}'
  else
    fail "missing sha256sum or shasum for checksum verification"
  fi
}

fetch() {
  url="$1"
  output="$2"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$url" -o "$output"
  elif command -v wget >/dev/null 2>&1; then
    wget -qO "$output" "$url"
  else
    fail "missing curl or wget"
  fi
}

need_cmd tar
need_cmd mkdir
need_cmd mktemp
need_cmd install

os="$(uname -s)"
arch="$(uname -m)"

case "$os" in
  Linux) platform_os="linux" ;;
  *)
    fail "unsupported OS: $os (installer currently supports Linux x86_64 only)"
    ;;
esac

case "$arch" in
  x86_64|amd64) platform_arch="amd64" ;;
  *)
    fail "unsupported architecture: $arch (installer currently supports Linux x86_64 only)"
    ;;
esac

case "$CHANNEL" in
  nightly)
    asset_base="rocm-cli-nightly-${platform_os}-${platform_arch}.tar.gz"
    release_path="releases/download/nightly"
    ;;
  release)
    asset_base="rocm-cli-${platform_os}-${platform_arch}.tar.gz"
    release_path="releases/latest/download"
    ;;
  *)
    asset_base="rocm-cli-${platform_os}-${platform_arch}.tar.gz"
    release_path="releases/download/${CHANNEL}"
    ;;
esac

download_base="${ROCM_CLI_DOWNLOAD_BASE:-https://github.com/${REPO}/${release_path}}"
archive_url="${download_base}/${asset_base}"
sha_url="${archive_url}.sha256"

tmp_dir="$(mktemp -d)"
cleanup() {
  rm -rf "$tmp_dir"
}
trap cleanup EXIT INT TERM

archive_path="${tmp_dir}/${asset_base}"
sha_path="${archive_path}.sha256"

echo "rocm-cli installer"
echo "  repo: ${REPO}"
echo "  channel: ${CHANNEL}"
echo "  install_dir: ${INSTALL_DIR}"
echo "  download: ${archive_url}"

fetch "$archive_url" "$archive_path"
fetch "$sha_url" "$sha_path"

expected="$(awk '{print $1}' "$sha_path" | head -n1)"
[ -n "$expected" ] || fail "checksum file did not contain a sha256 digest"
actual="$(sha256_file "$archive_path")"
[ "$expected" = "$actual" ] || fail "checksum verification failed"

extract_dir="${tmp_dir}/extract"
mkdir -p "$extract_dir"
tar -xzf "$archive_path" -C "$extract_dir"

bundle_dir="$(find "$extract_dir" -mindepth 1 -maxdepth 1 -type d | head -n1)"
[ -n "$bundle_dir" ] || fail "unable to locate extracted bundle directory"

[ -f "${bundle_dir}/bin/rocm" ] || fail "bundle did not contain bin/rocm"
[ -f "${bundle_dir}/bin/rocmd" ] || fail "bundle did not contain bin/rocmd"
[ -f "${bundle_dir}/bin/rocm-engine-pytorch" ] || fail "bundle did not contain bin/rocm-engine-pytorch"

mkdir -p "$INSTALL_DIR"
for bin_name in rocm rocmd rocm-engine-pytorch; do
  install -m 0755 "${bundle_dir}/bin/${bin_name}" "${INSTALL_DIR}/${bin_name}"
done

echo "installed:"
echo "  ${INSTALL_DIR}/rocm"
echo "  ${INSTALL_DIR}/rocmd"
echo "  ${INSTALL_DIR}/rocm-engine-pytorch"

case ":$PATH:" in
  *:"${INSTALL_DIR}":*)
    ;;
  *)
    echo "note: ${INSTALL_DIR} is not on PATH"
    echo "  add this to your shell profile:"
    echo "  export PATH=\"${INSTALL_DIR}:\$PATH\""
    ;;
esac

echo "run:"
echo "  rocm doctor"
