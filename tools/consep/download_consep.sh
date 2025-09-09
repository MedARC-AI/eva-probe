#!/usr/bin/env bash
# Usage: ./download_consep.sh /path/to/data
# Requires ACCESS_KEY_ID and SECRET_ACCESS_KEY in environment variables

set -e

# Load .env if present in the script’s folder
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/.env" ]]; then
  export $(grep -v '^#' "$SCRIPT_DIR/.env" | xargs)
fi

ROOT=${1:-"./data"}
DEST="$ROOT/consep"

echo "[INFO] Preparing CoNSeP dataset under $DEST"
mkdir -p "$DEST"

# Ensure openxlab is installed
pip show openxlab >/dev/null 2>&1 || pip install -U openxlab

# Login non-interactively
if [[ -z "$ACCESS_KEY_ID" || -z "$SECRET_ACCESS_KEY" ]]; then
  echo "[ERROR] Please export ACCESS_KEY_ID and SECRET_ACCESS_KEY before running."
  exit 1
fi
echo "[INFO] Logging into OpenXLab..."
openxlab login --ak "$ACCESS_KEY_ID" --sk "$SECRET_ACCESS_KEY"

# Download dataset
echo "[INFO] Downloading CoNSeP dataset..."
openxlab dataset get --dataset-repo OpenDataLab/CoNSeP --target-path "$DEST"

# Unzip consep.zip into $DEST/tmp_extract
ZIP_PATH="$DEST/OpenDataLab___CoNSeP/raw/consep.zip"
if [[ ! -f "$ZIP_PATH" ]]; then
  echo "[ERROR] Expected $ZIP_PATH not found."
  exit 1
fi

TMP_EXTRACT="$DEST/tmp_extract"
mkdir -p "$TMP_EXTRACT"
echo "[INFO] Extracting $ZIP_PATH..."
unzip -q "$ZIP_PATH" -d "$TMP_EXTRACT"

# Move Test and Train directly under $DEST
if [[ -d "$TMP_EXTRACT/CoNSEP" ]]; then
  mv "$TMP_EXTRACT/CoNSEP/Test" "$DEST/"
  mv "$TMP_EXTRACT/CoNSEP/Train" "$DEST/"
else
  echo "[ERROR] Unexpected structure inside zip."
  exit 1
fi

# Move metafile.yaml and README.md to $DEST
echo "[INFO] Organizing files..."
mv "$DEST/OpenDataLab___CoNSeP/metafile.yaml" "$DEST/"
mv "$DEST/OpenDataLab___CoNSeP/README.md" "$DEST/"

# Cleanup
rm -rf "$DEST/OpenDataLab___CoNSeP"
rm -rf "$TMP_EXTRACT"

echo "[INFO] Done. Final structure:"
ls -R "$DEST"
