#!/bin/bash

# Gleason Arvaniti Dataset Downloader with Direct URLs
# Usage: ./download_gleason_arvaniti.sh [output_directory]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR%/*}"
OUTPUT_DIR="${1:-$REPO_ROOT/data/gleason_arvaniti_data}"
mkdir -p "$OUTPUT_DIR"

echo "Downloading Gleason Arvaniti dataset to: $OUTPUT_DIR"

# Direct download URLs 
declare -A URLS=(
    ["ZT111_4_A"]="https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/OCYCMP/YWA5TT"
    ["ZT111_4_B"]="https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/OCYCMP/3IKI3C"
    ["ZT111_4_C"]="https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/OCYCMP/RAFKES"
    ["ZT199_1_A"]="https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/OCYCMP/0SMDAH"
    ["ZT199_1_B"]="https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/OCYCMP/QEDF2L"
    ["ZT204_6_A"]="https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/OCYCMP/0W77ZC"
    ["ZT204_6_B"]="https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/OCYCMP/0R6XWD"
    ["ZT76_39_A"]="https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/OCYCMP/L2E0UK"
    ["ZT76_39_B"]="https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/OCYCMP/UFDMZW"
    ["ZT80_38_A"]="https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/OCYCMP/HUEM2D"
    ["ZT80_38_B"]="https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/OCYCMP/YVNUNM"
    ["ZT80_38_C"]="https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/OCYCMP/BSWH3O"
    ["Gleason_masks_train.tar.gz"]="https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/OCYCMP/PPWGZZ"
    ["Gleason_masks_test_pathologist1.tar.gz"]="https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/OCYCMP/62QWVZ"
    ["Gleason_masks_test_pathologist2.tar.gz"]="https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/OCYCMP/AIJYC5"
)

# Download all files first
for filename in "${!URLS[@]}"; do
    url="${URLS[$filename]}"
    
    if [ "$url" = "PASTE_URL_HERE" ]; then
        echo "Please update the URL for $filename in the script"
        continue
    fi
    
    # Add .tar.gz extension if not present
    # Add this code since sometimes file name does not have .tar.gz extension even though it is a .tar.gz file
    if [[ ! "$filename" =~ \.tar\.gz$ ]]; then
        filename="${filename}.tar.gz"
    fi
    
    echo "Downloading $filename..."
    curl -L -o "${OUTPUT_DIR}/${filename}" "$url"
done

# Extract all .tar.gz files to same folder
echo "Extracting all .tar.gz files..."
for file in "${OUTPUT_DIR}"/*.tar.gz; do
    if [ -f "$file" ]; then
        echo "Extracting $(basename "$file")..."
        tar -xzf "$file" -C "$OUTPUT_DIR"
        rm "$file"
    fi
done

echo "Download and extraction completed"
echo "Organizing files into required directory structure for create_patches.py"

# Create required directories
mkdir -p "${OUTPUT_DIR}/TMA_images"
mkdir -p "${OUTPUT_DIR}/Gleason_masks_test"
mkdir -p "${OUTPUT_DIR}/tma_info"

# Move test pathologist directories into Gleason_masks_test
if [ -d "${OUTPUT_DIR}/Gleason_masks_test_pathologist1" ]; then
    mv "${OUTPUT_DIR}/Gleason_masks_test_pathologist1" "${OUTPUT_DIR}/Gleason_masks_test/"
fi
if [ -d "${OUTPUT_DIR}/Gleason_masks_test_pathologist2" ]; then
    mv "${OUTPUT_DIR}/Gleason_masks_test_pathologist2" "${OUTPUT_DIR}/Gleason_masks_test/"
fi

# Move all TMA images to folder TMA_images (using rsync for better performance)
if command -v rsync &> /dev/null; then
    # Use rsync if available (faster for large numbers of files)
    find "${OUTPUT_DIR}" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.JPG" \) -print0 | xargs -0 -I {} rsync -a --remove-source-files -- {} "${OUTPUT_DIR}/TMA_images/"
else
    # Fallback to find + mv if rsync not available
    find "${OUTPUT_DIR}" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.JPG" \) -exec mv -f {} "${OUTPUT_DIR}/TMA_images/" \;
fi

# Clean up empty directories (except the ones we want to keep)
find "${OUTPUT_DIR}" -mindepth 1 -type d \( -name "TMA_images" -o -name "Gleason_masks_train" -o -name "Gleason_masks_test" -o -name "tma_info" \) -prune -o -type d -empty -exec rmdir {} \; 2>/dev/null || true

# echo "Dataset organized successfully!"
# echo "Final directory structure at: $OUTPUT_DIR"
# echo "  ├─ TMA_images/"
# echo "  ├─ Gleason_masks_train/"
# echo "  ├─ Gleason_masks_test/"
# echo "  │    ├─ Gleason_masks_test_pathologist1/"
# echo "  │    └─ Gleason_masks_test_pathologist2/"
# echo "  └─ tma_info/"

echo -e "\nRunning create_patches.py..."
# remove files with "_" prefix, these corrupted files cause errors
find "${OUTPUT_DIR}" -name '._*' -type f -delete  

PATCHES_DIR="${SCRIPT_DIR%/*}/data/arvaniti_gleason_patches"
python "$SCRIPT_DIR/create_patches.py" "${OUTPUT_DIR}" "${PATCHES_DIR}"
