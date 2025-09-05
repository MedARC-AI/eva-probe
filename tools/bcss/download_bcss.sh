#!/usr/bin/env bash
# Download and prepare the BCSS dataset for EVA
# Usage: ./download_bcss.sh /path/to/data

set -e

# Target root directory
ROOT=${1:-"./data"}
DEST="$ROOT/bcss"

echo "[INFO] Preparing BCSS dataset under $DEST"
mkdir -p "$DEST"

# Google Drive folder links
RGBS_ID="1cMvgHRIetLR1C_arBDUHa4iW_mddU2X3"
MASKS_ID="1uYg1I1gvL_v6w1N0qq7-s7opr5lzsSA1"
META_ID="1e-mr3zqaEeOVQQLgWEgSL8dIgAdsclnl"

# Download function using gdown
download_folder () {
    ID=$1
    OUT=$2
    echo "[INFO] Downloading to $OUT ..."
    mkdir -p "$OUT"
    gdown --folder "$ID" -O "$OUT" --remaining-ok
}

# Step 1: Download each folder
download_folder "$RGBS_ID" "$DEST/rgbs_colorNormalized"
download_folder "$MASKS_ID" "$DEST/masks"
download_folder "$META_ID" "$DEST/meta"

# Step 2: Remove logs (not needed)
if [ -d "$DEST/logs" ]; then
    rm -rf "$DEST/logs"
fi

echo "[INFO] BCSS dataset successfully downloaded and organized!"
tree -L 2 "$DEST"
