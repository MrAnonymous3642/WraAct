#!/bin/bash

# === download_resources.sh ===
# Script to download and extract resources from Google Drive folders

set -e  # Exit on error

# === Function: download and extract zip files ===
download_and_extract() {
    FOLDER_NAME=$1
    GDRIVE_URL=$2
    TARGET_DIR=$3

    echo ""
    echo "[INFO] Downloading '$FOLDER_NAME' from Google Drive..."
    gdown --folder "$GDRIVE_URL" -O "$TARGET_DIR"

    echo "[INFO] Extracting .zip files in '$TARGET_DIR'..."
    find "$TARGET_DIR" -name "*.zip" -exec unzip -o {} -d "$TARGET_DIR" \;

    echo "[INFO] Cleaning up .zip files in '$TARGET_DIR'..."
    find "$TARGET_DIR" -name "*.zip" -exec rm {} \;

    echo "[SUCCESS] '$FOLDER_NAME' is ready in $TARGET_DIR"
}

# === Step 1: Ensure gdown is installed ===
if ! command -v gdown &> /dev/null; then
    echo "[INFO] Installing gdown..."
    pip install gdown
else
    echo "[INFO] gdown already installed."
fi

# === Step 2: Download and extract 'nets' folder ===
download_and_extract \
    "nets" \
    "https://drive.google.com/drive/folders/1ga8aQm2jPXHQ4BZU797UVZsE8YUoZJnS?usp=drive_link" \
    "./nets"

# === Step 3: Download and extract 'archived_logs' folder ===
download_and_extract \
    "archived_logs" \
    "https://drive.google.com/drive/folders/1ZBUSzI1lMM36GxFie7dF50gihtVjZGtB?usp=drive_link" \
    "./archived_logs"
