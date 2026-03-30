#!/bin/sh
set -eu

REPO="${ROCM_CLI_GITHUB_REPO:-powderluv/rocm-cli}"
CHANNEL="${1:-release}"
INSTALL_DIR="${ROCM_CLI_INSTALL_DIR:-$HOME/.local/bin}"
UPDATE_SHELL_PATH="${ROCM_CLI_UPDATE_SHELL_PATH:-1}"

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
    if [ -t 2 ]; then
      curl -fL --progress-bar "$url" -o "$output"
    else
      curl -fsSL "$url" -o "$output"
    fi
  elif command -v wget >/dev/null 2>&1; then
    if [ -t 2 ]; then
      wget --show-progress -O "$output" "$url"
    else
      wget -qO "$output" "$url"
    fi
  else
    fail "missing curl or wget"
  fi
}

need_cmd tar
need_cmd mkdir
need_cmd mktemp
need_cmd install
need_cmd rm
need_cmd grep
need_cmd sed

shell_name() {
  if [ -n "${ROCM_CLI_SHELL_NAME:-}" ]; then
    printf '%s\n' "${ROCM_CLI_SHELL_NAME}"
    return
  fi

  shell_path="${SHELL:-}"
  if [ -z "$shell_path" ]; then
    printf '%s\n' "sh"
    return
  fi
  printf '%s\n' "${shell_path##*/}"
}

profile_path_for_shell() {
  if [ -n "${ROCM_CLI_SHELL_PROFILE:-}" ]; then
    printf '%s\n' "${ROCM_CLI_SHELL_PROFILE}"
    return
  fi

  case "$(shell_name)" in
    bash) printf '%s\n' "$HOME/.bashrc" ;;
    zsh) printf '%s\n' "$HOME/.zshrc" ;;
    fish) printf '%s\n' "$HOME/.config/fish/config.fish" ;;
    ksh) printf '%s\n' "$HOME/.kshrc" ;;
    *) printf '%s\n' "$HOME/.profile" ;;
  esac
}

path_expr_for_profile() {
  path="$1"
  case "$path" in
    "$HOME")
      printf '%s\n' '$HOME'
      ;;
    "$HOME"/*)
      printf '%s\n' "\$HOME/${path#$HOME/}"
      ;;
    *)
      printf '%s\n' "$path"
      ;;
  esac
}

escape_for_double_quotes() {
  printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'
}

profile_has_path_entry() {
  profile="$1"
  path_expr="$2"
  [ -f "$profile" ] || return 1
  grep -F "# >>> rocm-cli path >>>" "$profile" >/dev/null 2>&1 && return 0
  grep -F "$path_expr" "$profile" >/dev/null 2>&1 && return 0
  grep -F "$INSTALL_DIR" "$profile" >/dev/null 2>&1 && return 0
  return 1
}

append_path_snippet() {
  profile="$1"
  shell_kind="$2"
  path_expr="$3"
  escaped_path_expr="$(escape_for_double_quotes "$path_expr")"

  profile_dir="${profile%/*}"
  if [ "$profile_dir" != "$profile" ]; then
    mkdir -p "$profile_dir"
  fi
  [ -f "$profile" ] || : > "$profile"

  if profile_has_path_entry "$profile" "$path_expr"; then
    printf '%s\n' "unchanged:${profile}"
    return 0
  fi

  case "$shell_kind" in
    fish)
      cat >> "$profile" <<EOF

# >>> rocm-cli path >>>
if not contains -- "${escaped_path_expr}" \$PATH
    set -gx PATH "${escaped_path_expr}" \$PATH
end
# <<< rocm-cli path <<<
EOF
      ;;
    *)
      cat >> "$profile" <<EOF

# >>> rocm-cli path >>>
case ":\$PATH:" in
  *:"${escaped_path_expr}":*) ;;
  *) export PATH="${escaped_path_expr}:\$PATH" ;;
esac
# <<< rocm-cli path <<<
EOF
      ;;
  esac

  printf '%s\n' "updated:${profile}"
}

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

manifest_path="${INSTALL_DIR}/.rocm-cli-manifest"

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
[ -f "${bundle_dir}/bin/rocm-codex" ] || fail "bundle did not contain bin/rocm-codex"

mkdir -p "$INSTALL_DIR"

if [ -f "$manifest_path" ]; then
  echo "removing previous rocm-cli install"
  while IFS= read -r installed_path; do
    [ -n "$installed_path" ] || continue
    case "$installed_path" in
      "$INSTALL_DIR"/*)
        rm -f "$installed_path"
        ;;
      *)
        echo "warning: skipping manifest entry outside install dir: $installed_path" >&2
        ;;
    esac
  done < "$manifest_path"
  rm -f "$manifest_path"
fi

manifest_tmp="${tmp_dir}/install-manifest"
: > "$manifest_tmp"
for bin_path in "${bundle_dir}"/bin/*; do
  [ -f "$bin_path" ] || continue
  bin_name="${bin_path##*/}"
  rm -f "${INSTALL_DIR}/${bin_name}"
  install -m 0755 "$bin_path" "${INSTALL_DIR}/${bin_name}"
  echo "${INSTALL_DIR}/${bin_name}" >> "$manifest_tmp"
done
install -m 0644 "$manifest_tmp" "$manifest_path"

echo "installed:"
while IFS= read -r installed_path; do
  [ -n "$installed_path" ] || continue
  echo "  ${installed_path}"
done < "$manifest_path"

case ":$PATH:" in
  *:"${INSTALL_DIR}":*)
    ;;
  *)
    if [ "$UPDATE_SHELL_PATH" = "1" ]; then
      profile_path="$(profile_path_for_shell)"
      path_expr="$(path_expr_for_profile "$INSTALL_DIR")"
      profile_result="$(append_path_snippet "$profile_path" "$(shell_name)" "$path_expr")" || true
      case "$profile_result" in
        updated:*)
          echo "shell path updated:"
          echo "  profile: ${profile_result#updated:}"
          echo "  added: ${path_expr}"
          echo "  restart your shell or run:"
          echo "  export PATH=\"${INSTALL_DIR}:\$PATH\""
          ;;
        unchanged:*)
          echo "shell path already configured:"
          echo "  profile: ${profile_result#unchanged:}"
          ;;
        *)
          echo "note: ${INSTALL_DIR} is not on PATH"
          echo "  add this to your shell profile:"
          echo "  export PATH=\"${INSTALL_DIR}:\$PATH\""
          ;;
      esac
    else
      echo "note: ${INSTALL_DIR} is not on PATH"
      echo "  add this to your shell profile:"
      echo "  export PATH=\"${INSTALL_DIR}:\$PATH\""
    fi
    ;;
esac

echo "run:"
echo "  rocm doctor"
