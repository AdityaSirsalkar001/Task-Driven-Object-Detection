import re
from pathlib import Path
from datetime import datetime

# Resolves the directory where this script is located (the RM repo root)
REPO_ROOT = Path(__file__).resolve().parent
OUTPUT_BASE_DIR = REPO_ROOT / "Output_Images"

def get_output_dir(team_member: str, module_name: str, user_query: str = None) -> str:
    """
    Generates and creates the correct output directory path inside the repo.
    Structure: RM/Output_Images/<Team_Member>/<Module_Name>/<Task_or_Timestamp>/
    """
    # Create the base path for the specific team member and module
    module_dir = OUTPUT_BASE_DIR / team_member / module_name
    
    if user_query:
        # Task-based folder: Sanitize the query to be filesystem-safe
        folder_name = re.sub(r'[^a-zA-Z0-9 \-_]', '', user_query).strip().replace(' ', '_')
        if not folder_name:
            folder_name = "unnamed_task"
        final_dir = module_dir / folder_name
    else:
        # Timestamp-based folder (for automated pipeline runs without a query)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_dir = module_dir / timestamp
        
    # Automatically create the directories if they don't exist
    final_dir.mkdir(parents=True, exist_ok=True)
    
    return str(final_dir)

if __name__ == "__main__":
    # Test execution
    print(f"Base Output Directory: {OUTPUT_BASE_DIR}\n")
    
    # 1. Test a task-based output path
    task_path = get_output_dir("Aditya", "Graph_Approach", "open the parcel")
    print(f"Task-based path created:\n  {task_path}\n")
    
    # 2. Test a timestamp-based output path
    time_path = get_output_dir("Anshdeep_Singh", "stages")
    print(f"Timestamp-based path created:\n  {time_path}")