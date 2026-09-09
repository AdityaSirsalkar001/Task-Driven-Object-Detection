import os
import re
from pathlib import Path

# Dynamically set target directory relative to this script at the repo root
target_dir = Path(__file__).resolve().parent / "Input_Images"

def select_directories_and_range():
    # Fallback to 'Input' if 'Input_Images' hasn't been created yet
    working_dir = target_dir
    if not working_dir.exists():
        fallback = Path(__file__).resolve().parent / "Input"
        if fallback.exists():
            working_dir = fallback
        else:
            print(f"Error: Could not find '{working_dir}'.")
            return None

    # List all subdirectories
    subdirs = [d for d in working_dir.iterdir() if d.is_dir()]
    
    if not subdirs:
        print(f"No subdirectories found in '{working_dir}'.")
        return None

    # Prompt user for multiple directories
    print(f"\nAvailable directories in {working_dir.name}:")
    for i, d in enumerate(subdirs):
        print(f"[{i}] {d.name}")

    selected_folders = []
    while True:
        choice_input = input(f"Select directories (comma-separated, e.g., 0,1,2): ").strip()
        if not choice_input:
            continue
            
        try:
            choices = [int(x.strip()) for x in choice_input.split(',')]
            if all(0 <= choice < len(subdirs) for choice in choices):
                selected_folders = [subdirs[c] for c in choices]
                break
            else:
                print(f"Invalid selection. Please choose numbers between 0 and {len(subdirs)-1}.")
        except ValueError:
            print("Please enter valid comma-separated numbers (e.g., 0,2).")

    # Prompt user for image range
    start_num, end_num = 0, float('inf')
    while True:
        range_input = input("Enter image range (e.g., 1-500) or press Enter for all: ").strip()
        if not range_input:
            break
            
        match = re.match(r"(\d+)\s*-\s*(\d+)", range_input)
        if match:
            start_num, end_num = int(match.group(1)), int(match.group(2))
            if start_num <= end_num:
                break
            else:
                print("Start number must be less than or equal to the end number.")
        else:
            print("Invalid format. Please use 'start-end' (e.g., 1-500).")

    return {
        "folders": [str(f.resolve()) for f in selected_folders],
        "range": (start_num, end_num)
    }