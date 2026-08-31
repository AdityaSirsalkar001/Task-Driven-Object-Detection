# RM — Shared Pipeline Repo

This repo holds the Python pipeline code for **Aditya**, **Anshdeep**, and **Aryan**. It's cloned inside a larger local `Project_Root` folder that also holds the shared virtual environment and the images (which are **not** pushed to GitHub).

## 0. About the Project

This is a **Research Methodology (RM) course project** on task-based object detection using a knowledge graph.

**Problem statement:** How can we enable a resource-constrained device (like a Raspberry Pi) to intelligently identify objects in an image that can be used to perform a specific human task, using only local, offline AI models?

**Why it matters:**
- Assistive technology — helping visually impaired individuals identify tools
- Robotics — helping a robot choose the right tool for a job
- Independent living / elderly care
- Education — tool safety and object recognition

**Approach:** Each member is exploring their own combination of models/approaches under a shared module structure (`Module_1`, `Module_2`, ...), all standardized to the same input/output format (see below) so results are directly comparable. The exact knowledge graph design and the specific models used are still being finalized and may differ across members — check each member's module for their current implementation.

**Target deployment:** Raspberry Pi (resource-constrained, offline).

## 1. Folder Structure (Project_Root)

```
Project_Root/
├── .venv/                          # Shared virtual environment (NOT in repo)
├── Input_Images/                   # All input images go here (NOT in repo)
│   └── [input images]
├── RM/                              # This GitHub repo (clone here)
│   ├── Aditya/
│   │   ├── Module_1/
│   │   │   ├── script1.py
│   │   │   ├── script2.py
│   │   │   └── config_or_kg.json
│   │   ├── Module_2/
│   │   │   └── ...
│   │   └── Module_N/
│   │       └── ...
│   ├── Anshdeep/                    # Same structure as Aditya (TBD)
│   │   └── [Module_1, Module_2, ...]
│   ├── Aryan/                       # Same structure as Aditya (TBD)
│   │   └── [Module_1, Module_2, ...]
│   └── README.md
└── Output_Images/                   # All outputs go here (NOT in repo)
    ├── Aditya/<module_name>/<task_name or timestamp>/
    ├── Anshdeep/<module_name>/<task_name or timestamp>/
    └── Aryan/<module_name>/<task_name or timestamp>/
```

**Rule of thumb:** everything that reads or writes *images* is centralized outside the repo. Everything that's *code* (and small config/JSON files like knowledge graphs) lives inside the repo, in your own member folder.

## 2. First-Time Setup

1. Create a `Project_Root` folder anywhere on your machine.
2. Inside it, clone this repo:
   ```bash
   cd Project_Root
   git clone https://github.com/AdityaSirsalkar001/RM.git
   ```
3. Inside `Project_Root`, create the two shared image folders (they're gitignored / not part of the repo):
   ```bash
   mkdir Input_Images Output_Images
   ```
4. Set up the shared virtual environment at the `Project_Root` level:
   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Windows: .venv\Scripts\activate
   pip install -r RM/requirements.txt
   ```
   *(`requirements.txt` is coming soon — once it's added to the repo, use it to install everything needed to run any member's code.)*
5. Inside `RM/`, make sure you have your own folder named after you (`Aditya/`, `Anshdeep/`, or `Aryan/`) with your module subfolders inside it, matching the structure above.

## 3. Running Code

Always run from `Project_Root` with the venv active:

```bash
cd Project_Root
source .venv/bin/activate
python RM/<Your_Name>/<Module_Name>/<script>.py
```

Example:
```bash
python RM/Aditya/Hash_Map/pipeline_with_kg.py
```

## 4. Converting Your Own Code to Fit This Structure

If your existing script reads/writes images from its own local folder (e.g. `Open_Parcel/`, `./images/`, etc.), you need to update it to:
- **Read** input images from `Project_Root/Input_Images/`
- **Write** output images to `Project_Root/Output_Images/<Your_Name>/<Module_Name>/<Task_or_Timestamp>/`

The easiest way: paste the prompt below into an LLM along with your script, one file at a time, and it will return the modified version with only the path logic changed — nothing else about your code's functionality.

<details>
<summary><strong>Click to expand: Path Standardization Prompt</strong></summary>

```
Python Pipeline Code - Path Standardization Task
Project Structure

I need you to modify Python pipeline code files to standardize file paths across my team's repository. Here's the folder structure:
text

Project_Root/
├── .venv/                          # Shared virtual environment
├── Input_Images/                   # Standard input folder
│   └── [all input images go here]
├── RM/                             # GitHub repository clone
│   ├── Aditya/                     # Team member 1 folder
│   │   ├── [Module_1]/             # Module 1 (e.g., Hash_Map)
│   │   │   ├── [main_pipeline_file].py
│   │   │   ├── [kg_manager_file].py
│   │   │   ├── [llm_fallback_file].py
│   │   │   └── [knowledge_graph_file].json
│   │   ├── [Module_2]/             # Module 2 (e.g., Hierarchical_KG)
│   │   │   ├── [main_pipeline_file].py
│   │   │   ├── [kg_manager_file].py
│   │   │   ├── [llm_fallback_file].py
│   │   │   └── [knowledge_graph_file].json
│   │   ├── [Module_3]/             # Module 3 (e.g., Attribute_KG)
│   │   │   ├── [main_pipeline_file].py
│   │   │   ├── [kg_manager_file].py
│   │   │   └── [knowledge_graph_file].json
│   │   └── [Module_4]/             # Module 4 (e.g., Scene_graph)
│   │       └── [main_pipeline_file].py
│   ├── Aryan/                      # Team member 2 folder
│   │   └── [their code files]
│   ├── Anshdeep/                   # Team member 3 folder
│   │   └── [their code files]
│   └── README.md
└── Output_Images/                  # Standard output folder
    ├── Aditya/                     # Team member 1 outputs
    │   ├── [Module_1]/             # Outputs from Module 1
    │   │   ├── [task_name]/        # Task-specific subfolder
    │   │   │   └── [annotated images]
    │   ├── [Module_2]/             # Outputs from Module 2
    │   │   ├── [task_name]/        # Task-specific subfolder
    │   │   │   └── [annotated images]
    │   ├── [Module_3]/             # Outputs from Module 3
    │   │   ├── [task_name]/        # Task-specific subfolder
    │   │   │   └── [annotated images]
    │   └── [Module_4]/             # Outputs from Module 4
    │       └── [timestamp]/        # Timestamp-based subfolder
    │           └── [annotated images & JSON]
    ├── Aryan/
    │   └── [Module_Name]/
    │       └── [task_name]/
    └── Anshdeep/
        └── [Module_Name]/
            └── [task_name]/

Task Requirements

For each code file you receive, you MUST:
1. IDENTIFY THE FILE LOCATION

    Determine the exact path of the file within the repository

    Example: Project_Root/RM/Aditya/[Module_1]/[main_pipeline_file].py

    Calculate the correct number of .parent calls needed to reach Project_Root

2. STANDARDIZE INPUT PATHS

    Find ALL code that reads images from a folder

    Change ALL input paths to read from: Project_Root/Input_Images/

    Remove hardcoded paths like Open_Parcel/, ./images/, etc.

    Use this pattern:

python

# [PATH CHANGE] Resolve Project_Root
SCRIPT_DIR = Path(__file__).parent.parent.parent.parent  # Adjust based on file depth

# [PATH CHANGE] Centralized input directory
INPUT_FOLDER = SCRIPT_DIR / "Input_Images"

3. STANDARDIZE OUTPUT PATHS

    Find ALL code that saves output images

    Change ALL output paths to: Project_Root/Output_Images/{Team_Member}/{Module_Name}/{Task_Name}/

    Use this pattern:

python

# [PATH CHANGE] Team member name (from folder structure)
TEAM_MEMBER = "Aditya"  # or "Aryan", "Anshdeep"

# [PATH CHANGE] Module name (the folder containing this code)
MODULE_NAME = "[Module_1]"  # e.g., "Hash_Map", "Hierarchical_KG", "Attribute_KG", "Scene_graph"

# [PATH CHANGE] Output base directory
OUTPUT_BASE = SCRIPT_DIR / "Output_Images" / TEAM_MEMBER / MODULE_NAME

# [PATH CHANGE] Create output base directory if missing
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

For Modules with User Query Input (Task-Based):
python

# [PATH CHANGE] Create task-specific folder
# Sanitize the task name to be filesystem-safe
task_folder_name = "".join(c for c in user_query if c.isalnum() or c in (' ', '-', '_')).strip().replace(' ', '_')
OUTPUT_FOLDER = OUTPUT_BASE / task_folder_name
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
print(f"[*] Output will be saved to: {OUTPUT_FOLDER}")

Example: If user inputs "open the parcel", outputs go to:
text

Project_Root/Output_Images/Aditya/[Module_1]/open_the_parcel/

For Modules Without User Query Input (Timestamp-Based):
python

# [PATH CHANGE] Create timestamp-based folder
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FOLDER = OUTPUT_BASE / timestamp
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
print(f"[*] Output will be saved to: {OUTPUT_FOLDER}")

Example: Outputs go to:
text

Project_Root/Output_Images/Aditya/[Module_4]/20240115_143022/

4. HANDLE MODULE-SPECIFIC RESOURCES

    DO NOT change paths for JSON files, knowledge graphs, or configuration files

    These stay in their original module folders

    For example: knowledge_graph.json stays in Project_Root/RM/Aditya/[Module_1]/

    Only change image input/output paths

5. PRESERVE ALL FUNCTIONALITY

    ONLY change path-related code

    Do NOT modify model loading, inference logic, preprocessing, or any other functionality

    Keep all existing code logic exactly as is

    Do not change model configurations, imports (except path imports), or unrelated code

6. AUTO-CREATE OUTPUT FOLDERS

    Ensure the output folder is created if it doesn't exist

    Use OUTPUT_BASE.mkdir(parents=True, exist_ok=True) for base folder

    Use OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True) for task-specific or timestamp folder

7. HANDLE OUTPUT SUB-FOLDERS

    If the code saves outputs to sub-folders like Annotated_KG/ or Results/, maintain that structure

    Example: Project_Root/Output_Images/Aditya/[Module_1]/open_the_parcel/Annotated_KG/

8. ADD OUTPUT LOCATION INDICATOR

    Add a print statement to show where outputs will be saved:

python

print(f"[*] Output will be saved to: {OUTPUT_FOLDER}")

Path Resolution Guide
For Files in Team Member Subfolders (Module Folders):
text

File: Project_Root/RM/Aditya/[Module_1]/[main_pipeline_file].py
Path calculation:
.parent              → [Module_1]/
.parent.parent       → Aditya/
.parent.parent.parent → RM/
.parent.parent.parent.parent → Project_Root/

SCRIPT_DIR = Path(__file__).parent.parent.parent.parent

For Files Directly in Team Member Folder:
text

File: Project_Root/RM/Aditya/[main_code_file].py
Path calculation:
.parent      → Aditya/
.parent.parent → RM/
.parent.parent.parent → Project_Root/

SCRIPT_DIR = Path(__file__).parent.parent.parent

For Shared Module Folders (outside team member folders):
text

File: Project_Root/RM/[Shared_Module]/[main_pipeline_file].py
Path calculation:
.parent      → [Shared_Module]/
.parent.parent → RM/
.parent.parent.parent → Project_Root/

SCRIPT_DIR = Path(__file__).parent.parent.parent

Module-Specific Output Folder Logic
Modules with User Query Input (Task-Based):

These modules take a user query as input. Create a task-specific folder:
python

task_folder_name = "".join(c for c in user_query if c.isalnum() or c in (' ', '-', '_')).strip().replace(' ', '_')
OUTPUT_FOLDER = OUTPUT_BASE / task_folder_name
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

Example: If user inputs "open the parcel", outputs go to:
text

Project_Root/Output_Images/Aditya/[Module_1]/open_the_parcel/

Modules Without User Query Input (Timestamp-Based):

These modules process all images automatically. Use a timestamp-based folder:
python

from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FOLDER = OUTPUT_BASE / timestamp
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

Example: Outputs go to:
text

Project_Root/Output_Images/Aditya/[Module_4]/20240115_143022/

What to Output

For each file you modify, provide:

    The complete modified code with all changes

    Clearly mark all path-related changes with # [PATH CHANGE] comments

    Explain what you changed and why

    The exact output path where files will be saved

    Do not omit any existing code from the modified file

Example Modification
Current Code:
python

from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
INPUT_FOLDER = SCRIPT_DIR / "Open_Parcel"
OUTPUT_FOLDER = SCRIPT_DIR / "Annotated_KG"

INPUT_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

def main():
    user_query = input("Enter task: ").strip()
    # ... rest of code

Modified Code (for Aditya/[Module_1] with task-based folder):
python

from pathlib import Path

# [PATH CHANGE] Resolve Project_Root (4 levels up)
SCRIPT_DIR = Path(__file__).parent.parent.parent.parent

# [PATH CHANGE] Centralized input directory
INPUT_FOLDER = SCRIPT_DIR / "Input_Images"

# [PATH CHANGE] Team-specific output directory
TEAM_MEMBER = "Aditya"
MODULE_NAME = "[Module_1]"
OUTPUT_BASE = SCRIPT_DIR / "Output_Images" / TEAM_MEMBER / MODULE_NAME
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

def main():
    user_query = input("Enter task: ").strip()
    
    # [PATH CHANGE] Create task-specific output folder
    task_folder_name = "".join(c for c in user_query if c.isalnum() or c in (' ', '-', '_')).strip().replace(' ', '_')
    OUTPUT_FOLDER = OUTPUT_BASE / task_folder_name
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    print(f"[*] Output will be saved to: {OUTPUT_FOLDER}")
    
    # ... rest of code

Modified Code (for Aditya/[Module_4] without task input):
python

from pathlib import Path
from datetime import datetime

# [PATH CHANGE] Resolve Project_Root (4 levels up)
SCRIPT_DIR = Path(__file__).parent.parent.parent.parent

# [PATH CHANGE] Centralized input directory
INPUT_FOLDER = SCRIPT_DIR / "Input_Images"

# [PATH CHANGE] Team-specific output directory
TEAM_MEMBER = "Aditya"
MODULE_NAME = "[Module_4]"
OUTPUT_BASE = SCRIPT_DIR / "Output_Images" / TEAM_MEMBER / MODULE_NAME
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

# [PATH CHANGE] Create timestamp-based output folder
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FOLDER = OUTPUT_BASE / timestamp
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
print(f"[*] Output will be saved to: {OUTPUT_FOLDER}")

def main():
    # ... rest of code

Important Notes

    The .venv folder stays at root and must NOT be modified

    All code stays inside RM/ folder (GitHub repository)

    Input images are centralized in Project_Root/Input_Images/

    Outputs are organized by: Team_Member → Module → Task/Timestamp

    JSON files, knowledge graphs, and other resources stay in their original module folders

    Do NOT rename functions, variables, classes, or files

    Make the smallest possible changes necessary to standardize paths

Summary Checklist

Before submitting the modified code, verify:

    □

    SCRIPT_DIR correctly resolves to Project_Root
    □

    INPUT_FOLDER points to Project_Root/Input_Images/
    □

    OUTPUT_BASE points to Project_Root/Output_Images/{Team_Member}/{Module_Name}/
    □

    Task-specific folder created (for modules with user query input)
    □

    Timestamp-based folder created (for modules without user query input)
    □

    All output folders are auto-created with mkdir(parents=True, exist_ok=True)
    □

    JSON/knowledge graph files stay in their original module folders
    □

    All functionality remains unchanged
    □

    Print statement added to show output location
    □

    All path changes marked with # [PATH CHANGE]

Running the Code

Always run the code from the project root with the virtual environment activated:
bash

# Navigate to project root
cd /path/to/Project_Root

# Activate virtual environment
source .venv/bin/activate

# Run the script
python RM/Aditya/[Module_1]/[main_pipeline_file].py

Please provide the code files one by one for modification.
```

</details>

## 5. Do's and Don'ts

- ✅ Do put your code (scripts, module-local JSON/config files) inside `RM/<Your_Name>/<Module_Name>/`
- ✅ Do read images only from `Project_Root/Input_Images/`
- ✅ Do write output images only to `Project_Root/Output_Images/<Your_Name>/<Module_Name>/...`
- ❌ Don't commit images to the repo
- ❌ Don't commit `.venv/`
- ❌ Don't rename functions, variables, classes, or files while doing the path conversion
- ❌ Don't touch other members' folders

## 6. Team

| Member | Repo Folder |
|---|---|
| Aditya | `RM/Aditya/` |
| Anshdeep | `RM/Anshdeep/` |
| Aryan | `RM/Aryan/` |

---
*Requirements file (`requirements.txt`) to be added — once available, install with `pip install -r RM/requirements.txt` after activating the venv.*