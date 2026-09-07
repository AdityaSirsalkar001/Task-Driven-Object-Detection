import os
import torch
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, Florence2ForConditionalGeneration

# Import your existing KG and SLM functions from your builder script
# (Assuming your previous script is named vector_atomic_kg_builder.py)
from atomic_kg_builder import VectorAtomicKnowledgeGraph, extract_atomic_attributes

# --- CONFIGURATION ---
# SCRIPT_DIR = Path(__file__).parent
INPUT_FOLDER = Path("~/Documents/RM/Input/wine").expanduser()  # Your image folder
NUM_IMAGES = 10  # Set 'n' for your 1 to n loop
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():

    print(INPUT_FOLDER)
    print("=" * 60)
    print("[*] STARTING AUTOMATED VISION-TO-GRAPH INGESTION")
    print("=" * 60)

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

    # 3. Iterate sequentially from 1 to n
    for i in range(1, 201):
        image_name = f"{i}.jpg"
        img_path = INPUT_FOLDER / image_name
        
        if not img_path.exists():
            print(f"[-] Image '{image_name}' not found. Skipping...")
            continue

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
                print(f"       [+] Ingested '{obj_name}' with {len(attributes)} attributes.")
            else:
                print(f"       [-] SLM failed to extract attributes for '{obj_name}'.")

    print("\n" + "=" * 60)
    print(f"[*] Vision Ingestion Complete! Graph saved to {kg.filepath.name}")
    print("=" * 60)

if __name__ == "__main__":
    main()
