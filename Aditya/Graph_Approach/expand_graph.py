import os
import sys
import re
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor, Florence2ForConditionalGeneration
from datetime import datetime
from tqdm import tqdm

# Import existing KG and SLM functions
from atomic_kg_builder import VectorAtomicKnowledgeGraph, extract_atomic_attributes

# ==========================================
# 1. CONFIGURATION & CONSTANTS
# ==========================================

# Traverses: Graph_Approach (1) -> Aditya (2) -> Root (3)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(REPO_ROOT))

import input_handler
import output_handler

TEAM_MEMBER = "Aditya"
MODULE_NAME = "Graph_Approach"
TASK_NAME = "Expand_Graph"

os.environ["HF_HOME"] = str(REPO_ROOT / "HuggingFaceModels")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    print("=" * 60)
    print("[*] STARTING AUTOMATED VISION-TO-GRAPH INGESTION")
    print("=" * 60)

    # 1. Fetch input directory dynamically using the handler
    selection = input_handler.select_directories_and_range()
    if not selection:
        print("[-] No input directory selected or found. Exiting.")
        return

    print(f"[*] Selected Folders: {len(selection['folders'])}")
    print(f"[*] Image Range: {selection['range'][0]} to {selection['range'][1]}")

    # 2. Gather and filter images across all selected folders
    start_idx, end_idx = selection["range"]
    image_files_raw = []

    for folder_path_str in selection["folders"]:
        folder_path = Path(folder_path_str)
        if folder_path.exists():
            for file in folder_path.iterdir():
                if file.is_file() and file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    match = re.search(r'(\d+)', file.stem)
                    if match:
                        num = int(match.group(1))
                        if start_idx <= num <= end_idx:
                            image_files_raw.append((num, file))

    # Sort files globally by numeric value
    image_files_raw.sort(key=lambda x: x[0])
    image_files = [file for _, file in image_files_raw]

    if not image_files:
        print("[-] No images found matching the selected criteria in the chosen folders!")
        return
    
    print(f"[+] Found {len(image_files)} images to process")

    # 3. Create timestamp-based output folder for logs
    output_dir_str = output_handler.get_output_dir(TEAM_MEMBER, MODULE_NAME, TASK_NAME)
    OUTPUT_FOLDER = Path(output_dir_str)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RUN_OUTPUT_FOLDER = OUTPUT_FOLDER / f"run_{timestamp}"
    RUN_OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    print(f"[*] Log output will be saved to: {RUN_OUTPUT_FOLDER}")

    # 4. Initialize the Knowledge Graph
    kg = VectorAtomicKnowledgeGraph()
    
    # 5. Load Florence-2 (Stays permanently on GPU)
    print(f"\n[*] Booting Florence-2 on {DEVICE.upper()}...")
    florence_id = "florence-community/Florence-2-base-ft"
    florence_processor = AutoProcessor.from_pretrained(florence_id)
    florence_model = Florence2ForConditionalGeneration.from_pretrained(
        florence_id, 
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    ).to(DEVICE).eval()

    task_prompt = "<DENSE_REGION_CAPTION>"
    
    new_objects_added = 0
    total_unique_objects = set()

    # 6. Process all images
    for img_path in tqdm(image_files, desc="Scanning Images for Objects"):
        image_name = img_path.name
        
        print(f"\n[*] Scanning Image: {image_name}")
        image = Image.open(img_path).convert("RGB")
        width, height = image.size

        # A. Detect Objects with Florence-2
        inputs = florence_processor(text=task_prompt, images=image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            generated_ids = florence_model.generate(**inputs, max_new_tokens=1024, do_sample=False, num_beams=3)
            
        generated_text = florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        results = florence_processor.post_process_generation(
            generated_text, task=task_prompt, image_size=(width, height)
        ).get(task_prompt, {})
        
        raw_labels = results.get('labels', [])
        
        # Deduplicate labels to avoid processing the same object twice in one image
        unique_objects = list(set([label.strip().lower() for label in raw_labels if label.strip()]))
        print(f"    -> Florence detected: {unique_objects}")
        
        total_unique_objects.update(unique_objects)

        # B. Extract Attributes & Add to Graph
        for obj_name in unique_objects:
            if obj_name in kg.objects:
                print(f"    -> [SKIP] '{obj_name}' is already in the graph.")
                continue
                
            print(f"    -> [SLM] Deconstructing '{obj_name}'...")
            # The SLM will automatically load into memory during this call
            attributes = extract_atomic_attributes(obj_name)
            
            if attributes:
                kg.add_object_with_attributes(obj_name, attributes)
                new_objects_added += 1
                print(f"       [+] Ingested '{obj_name}' with {len(attributes)} attributes.")
            else:
                print(f"       [-] SLM failed to extract attributes for '{obj_name}'.")

    # 7. Write Summary Log
    summary_log_path = RUN_OUTPUT_FOLDER / "ingestion_summary.txt"
    with open(summary_log_path, "w") as f:
        f.write(f"Ingestion Run: {timestamp}\n")
        f.write(f"Images Scanned: {len(image_files)}\n")
        f.write(f"Total Unique Objects Found: {len(total_unique_objects)}\n")
        f.write(f"New Objects Added to Graph: {new_objects_added}\n")
        f.write(f"Final Graph Size: {len(kg.objects)} objects\n")

    print("\n" + "=" * 60)
    print(f"[*] Vision Ingestion Complete!")
    print(f"[*] Total images processed: {len(image_files)}")
    print(f"[*] Unique objects detected: {len(total_unique_objects)}")
    print(f"[*] New objects added to graph: {new_objects_added}")
    print(f"[*] Graph saved to: {kg.filepath.resolve()}")
    print(f"[*] Log saved to: {summary_log_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()