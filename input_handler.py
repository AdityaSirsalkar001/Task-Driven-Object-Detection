import os
from pathlib import Path

# Dynamically set target directory relative to this script at the repo root
target_dir = Path(__file__).resolve().parent / "Input_Images"

def select_directory():
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

    # Prompt user
    print(f"\nAvailable directories in {working_dir.name}:")
    for i, d in enumerate(subdirs):
        print(f"[{i}] {d.name}")

    while True:
        try:
            choice_input = input(f"Select a directory (0-{len(subdirs)-1}): ").strip()
            if not choice_input:
                continue
                
            choice = int(choice_input)
            if 0 <= choice < len(subdirs):
                # Return the absolute path string 
                return str(subdirs[choice].resolve())
            else:
                print("Invalid selection. Try again.")
        except ValueError:
            print("Please enter a valid number.")