#!/bin/bash

# 1. Validate the input argument
if [ -z "$1" ]; then
    echo "Usage: $0 <target_directory>"
    exit 1
fi

TARGET_DIR="$1"

if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Directory '$TARGET_DIR' does not exist."
    exit 1
fi

# Ensure whiptail is available
if ! command -v whiptail &> /dev/null; then
    echo "Error: 'whiptail' is required. Install it using: sudo apt-get install whiptail"
    exit 1
fi

# Change to the target directory for clean relative paths
cd "$TARGET_DIR" || exit 1

# 2. Find subdirectories at ALL levels
# Removed -maxdepth 1 so it traverses the entire directory tree
mapfile -t SUBDIRS < <(find . -mindepth 1 -type d | sed 's|^\./||' | sort)

if [ ${#SUBDIRS[@]} -eq 0 ]; then
    echo "No subdirectories found in: $TARGET_DIR"
    exit 0
fi

# 3. Build the checklist array for whiptail
CHECKLIST=()
for dir in "${SUBDIRS[@]}"; do
    CHECKLIST+=("$dir" "" "OFF")
done

# 4. Display the dynamic checkbox UI
CHOICES=$(whiptail --title "Select Subfolders to Flatten" \
                   --checklist "Use Space to select, Enter to confirm.\nSelected folders will be recursively flattened to the top directory." \
                   22 78 12 "${CHECKLIST[@]}" 3>&1 1>&2 2>&3)

if [ $? -ne 0 ]; then
    echo "Operation cancelled by user."
    exit 0
fi

if [ -z "$CHOICES" ]; then
    echo "No folders selected. Exiting."
    exit 0
fi

# 5. Process the selections safely
eval "SELECTED=($CHOICES)"

echo "Starting recursive extraction process..."

for dir in "${SELECTED[@]}"; do
    # Check if the directory still exists. 
    # (It may have already been deleted if its parent was also selected)
    if [ -d "$dir" ]; then
        echo "Processing: $dir..."
        
        # Move all files from the chosen directory AND its subdirectories to the top folder
        find "$dir" -type f -exec mv --backup=numbered -t . {} + 2>/dev/null
        
        # Delete the folder and any empty subfolders inside it safely (bottom-up approach)
        find "$dir" -depth -type d -empty -delete 2>/dev/null
        
        if [ ! -d "$dir" ]; then
             echo "  -> Flattened and removed: $dir"
        else
             echo "  -> Warning: '$dir' was not completely removed (may contain hidden files)."
        fi
    else
        echo "  -> Skipped: $dir (already processed as part of a parent folder)"
    fi
done

echo "Extraction complete."
