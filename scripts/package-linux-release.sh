#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 <dist-name> [output-dir]" >&2
  exit 1
fi

DIST_NAME="$1"
OUTPUT_DIR="${2:-dist}"
ROOT_DIR="${OUTPUT_DIR}/${DIST_NAME}"
ARCHIVE_PATH="${OUTPUT_DIR}/${DIST_NAME}.tar.gz"
TAR_PATH="${OUTPUT_DIR}/${DIST_NAME}.tar"

mkdir -p "${OUTPUT_DIR}"
rm -rf "${ROOT_DIR}"
rm -f "${ARCHIVE_PATH}" "${ARCHIVE_PATH}.sha256" "${TAR_PATH}"
mkdir -p "${ROOT_DIR}/bin"

cp target/release/rocm "${ROOT_DIR}/bin/"
cp target/release/rocmd "${ROOT_DIR}/bin/"
cp target/release/rocm-engine-pytorch "${ROOT_DIR}/bin/"
cp README.md LICENSE install.sh "${ROOT_DIR}/"

(cd "${OUTPUT_DIR}" && tar -cf "${DIST_NAME}.tar" "${DIST_NAME}")
gzip -c "${TAR_PATH}" > "${ARCHIVE_PATH}"
rm -f "${TAR_PATH}"
sha256sum "${ARCHIVE_PATH}" > "${ARCHIVE_PATH}.sha256"

echo "created:"
echo "  ${ARCHIVE_PATH}"
echo "  ${ARCHIVE_PATH}.sha256"
