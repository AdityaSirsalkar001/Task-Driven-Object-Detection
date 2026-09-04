#!/bin/bash

# Pass 1: Rename everything to a unique, temporary ID to prevent ANY overwriting
counter=1
for file in *; do
    if [[ -f "$file" ]]; then
        if [[ "$file" == "$(basename "$0")" ]]; then continue; fi
        
        if [[ "$file" == *.* ]]; then
            ext=".${file##*.}"
        else
            ext=""
        fi
        
        # Add a TMP_SAFE prefix so it cannot possibly clash with existing numbers
        temp_name="TMP_SAFE_${counter}${ext}"
        mv -n -- "$file" "$temp_name"
        
        ((counter++))
    fi
done

# Pass 2: Remove the temporary prefix to leave only the clean numbers
for file in TMP_SAFE_*; do
    if [[ -f "$file" ]]; then
        final_name="${file#TMP_SAFE_}"
        mv -n -- "$file" "$final_name"
    fi
done

echo "Safe renaming complete!"
