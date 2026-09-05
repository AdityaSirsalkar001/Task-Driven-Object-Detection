from pathlib import Path

# Hardcoded target directory (change this to your desired path)
target_dir = "~/Documents/RM/Input/"


def select_directory():
  """Lists subdirectories in target_dir, prompts the user to choose one,

  and returns its absolute path.
  """
  target_path = Path(target_dir).resolve()

  # Validate the target directory
  if not target_path.exists() or not target_path.is_dir():
    print(f"Error: '{target_path}' is not a valid directory.")
    return None

  # Get a list of all subdirectories inside target_dir
  subdirs = sorted([d for d in target_path.iterdir() if d.is_dir()])

  if not subdirs:
    print(f"No subdirectories found in '{target_path}'.")
    return None

  # Print the available directories
  print(f"\nDirectories in {target_path}:")
  for index, subdir in enumerate(subdirs, start=1):
    print(f"  [{index}] {subdir.name}")

  # Prompt the user to select one
  while True:
    try:
      choice = input(
          "\nEnter the number of the directory you want to select: "
      )
      if not choice.strip():
        print("Selection cancelled.")
        return None

      choice_idx = int(choice)
      if 1 <= choice_idx <= len(subdirs):
        chosen_path = subdirs[choice_idx - 1].resolve()
        return str(chosen_path)
      else:
        print(f"Please enter a number between 1 and {len(subdirs)}.")
    except ValueError:
      print("Invalid input. Please enter a valid integer.")


if __name__ == "__main__":
  # Allows testing the script directly by running it
  result = select_directory()
  if result:
    print(f"\nChosen absolute path: {result}")
