# mhist_prepare.sh
# Prepares the MHIST dataset for EVA.
# Expects the user to have manually downloaded images.zip and annotations.csv into ./data/mhist

#!/usr/bin/env bash
set -e # Exit on error

# Path to MHIST dataset
DATA_DIR="./data/mhist"

# Check if the folder exists
if [ ! -d "$DATA_DIR" ]; then
  echo "ERROR: $DATA_DIR does not exist. Please create and place the files there."
  exit 1
fi

# Create images folder if it doesn't exist
mkdir -p "$DATA_DIR/images"

# Unzip images.zip into images folder
if [ -f "$DATA_DIR/images.zip" ]; then
  echo "[INFO] Unzipping images.zip..."
  unzip -o -j "$DATA_DIR/images.zip" -d "$DATA_DIR/images"
  echo "[INFO] Unzipping completed."
else
  echo "WARNING: images.zip not found in $DATA_DIR. Skipping unzip."
fi

# Delete images.zip after extraction
if [ -f "$DATA_DIR/images.zip" ]; then
  echo "[INFO] Deleting images.zip..."
  rm "$DATA_DIR/images.zip"
fi

# Check if annotations.csv exists
if [ -f "$DATA_DIR/annotations.csv" ]; then
  echo "[INFO] annotations.csv found."
else
  echo "WARNING: annotations.csv not found in $DATA_DIR. Please make sure it is placed there."
fi

exit 0
