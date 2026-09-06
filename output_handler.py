from pathlib import Path
from datetime import datetime
import re

REPO_ROOT = Path(__file__).resolve().parent
OUTPUT_BASE_DIR = REPO_ROOT / "Output_Images"

def get_output_dir(team_member: str, module_name: str, user_query: str = None) -> str:
    # Determine the final subfolder name (either the task/query or a timestamp)
    if user_query:
        # Sanitize the query to ensure it is a valid folder name
        folder_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', user_query.strip())
    else:
        folder_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Construct the full output path
    target_dir = OUTPUT_BASE_DIR / team_member / module_name / folder_name
    
    # Auto-create the directory structure
    target_dir.mkdir(parents=True, exist_ok=True)
    
    return str(target_dir.resolve())