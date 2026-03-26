#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "usage: $0 <dist-name> [output-dir] [target-triple]" >&2
  exit 1
fi

DIST_NAME="$1"
OUTPUT_DIR="${2:-dist}"
TARGET_TRIPLE="${3:-}"
ROOT_DIR="${OUTPUT_DIR}/${DIST_NAME}"
ARCHIVE_PATH="${OUTPUT_DIR}/${DIST_NAME}.tar.gz"
TAR_PATH="${OUTPUT_DIR}/${DIST_NAME}.tar"
BINARY_DIR="target/release"

if [[ -n "${TARGET_TRIPLE}" ]]; then
  BINARY_DIR="target/${TARGET_TRIPLE}/release"
fi

mkdir -p "${OUTPUT_DIR}"
rm -rf "${ROOT_DIR}"
rm -f "${ARCHIVE_PATH}" "${ARCHIVE_PATH}.sha256" "${TAR_PATH}"
mkdir -p "${ROOT_DIR}/bin"

cp "${BINARY_DIR}/rocm" "${ROOT_DIR}/bin/"
cp "${BINARY_DIR}/rocmd" "${ROOT_DIR}/bin/"
cp "${BINARY_DIR}/rocm-engine-pytorch" "${ROOT_DIR}/bin/"
cp README.md LICENSE install.sh "${ROOT_DIR}/"

(cd "${OUTPUT_DIR}" && tar -cf "${DIST_NAME}.tar" "${DIST_NAME}")
gzip -c "${TAR_PATH}" > "${ARCHIVE_PATH}"
rm -f "${TAR_PATH}"
sha256sum "${ARCHIVE_PATH}" > "${ARCHIVE_PATH}.sha256"

echo "created:"
echo "  ${ARCHIVE_PATH}"
echo "  ${ARCHIVE_PATH}.sha256"
