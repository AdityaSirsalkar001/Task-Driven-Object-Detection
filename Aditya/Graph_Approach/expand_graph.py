import os
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor, Florence2ForConditionalGeneration

# Import your existing KG and SLM functions from your builder script
from atomic_kg_builder import VectorAtomicKnowledgeGraph, extract_atomic_attributes

# ==========================================
# 1. CONFIGURATION & CONSTANTS
# ==========================================

# [PATH CHANGE] Resolve Project_Root (4 levels up from RM/Aditya/Graph_Approach/)
SCRIPT_DIR = Path(__file__).parent.parent.parent.parent

# [PATH CHANGE] Centralized input directory
INPUT_FOLDER = SCRIPT_DIR / "Input_Images"

# [PATH CHANGE] Team-specific output directory
TEAM_MEMBER = "Aditya"
MODULE_NAME = "Graph_Approach"
OUTPUT_BASE = SCRIPT_DIR / "Output_Images" / TEAM_MEMBER / MODULE_NAME
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

# [PATH CHANGE] Model cache directory
os.environ["HF_HOME"] = str(SCRIPT_DIR / "HuggingFaceModels")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    print("=" * 60)
    print("[*] STARTING AUTOMATED VISION-TO-GRAPH INGESTION")
    print("=" * 60)

    # [PATH CHANGE] Create timestamp-based output folder for logs
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_FOLDER = OUTPUT_BASE / f"expand_graph_{timestamp}"
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    print(f"[*] Log output will be saved to: {OUTPUT_FOLDER}")

    # 1. Initialize the Knowledge Graph
    kg = VectorAtomicKnowledgeGraph()
    
    # 2. Load Florence-2 (Stays permanently on GPU)
    print(f"[*] Booting Florence-2 on {DEVICE.upper()}...")
    florence_id = "florence-community/Florence-2-base-ft"
    florence_processor = AutoProcessor.from_pretrained(florence_id)
    florence_model = Florence2ForConditionalGeneration.from_pretrained(
        florence_id, 
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    ).to(DEVICE).eval()

    task_prompt = "<DENSE_REGION_CAPTION>"

    # [PATH CHANGE] Iterate through images in Input_Images folder
    # Get all .jpg images from the input folder
    image_files = sorted([f for f in INPUT_FOLDER.glob("*.jpg")])
    
    if not image_files:
        print(f"[-] No .jpg images found in {INPUT_FOLDER}")
        return
    
    print(f"[*] Found {len(image_files)} images to process")
    
    new_objects_added = 0
    total_unique_objects = set()

    # [PATH CHANGE] Process all images found in Input_Images
    for img_path in image_files:
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

    print("\n" + "=" * 60)
    print(f"[*] Vision Ingestion Complete!")
    print(f"[*] Total images processed: {len(image_files)}")
    print(f"[*] Unique objects detected: {len(total_unique_objects)}")
    print(f"[*] New objects added to graph: {new_objects_added}")
    print(f"[*] Graph saved to: {kg.filepath.resolve()}")
    print(f"[*] Log saved to: {OUTPUT_FOLDER}")
    print("=" * 60)

if __name__ == "__main__":
    main()