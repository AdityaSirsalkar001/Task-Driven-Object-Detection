import os
import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image, ImageDraw
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, Florence2ForConditionalGeneration
from atomic_kg_builder import VectorAtomicKnowledgeGraph
import re
import json
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION & CONSTANTS
# ==========================================

# Resolve Repo Root dynamically
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(REPO_ROOT))

# Import shared handlers from repo root
import input_handler
import output_handler

# Team-specific configuration for the output handler
TEAM_MEMBER = "Aditya"
MODULE_NAME = "Graph_Approach"

# Model cache directory mapped to Repo Root
os.environ["HF_HOME"] = str(REPO_ROOT / "HuggingFaceModels")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Using 3B model
SLM_ID = "Qwen/Qwen2.5-3B-Instruct"

def extract_task_attributes(task_query: str, tokenizer, model) -> list:
    """Uses the SLM to determine the atomic attributes and their normalized priority weights."""
    system_prompt = (
        "You are an expert in physical task analysis and tool affordance reasoning.\n\n"
        "TASK: Given a physical task description, identify the critical physical properties (attributes) "
        "that the TOOL or INSTRUMENT must possess to successfully perform this task.\n\n"
        "IMPORTANT RULES:\n"
        "1. Think step by step about the physical interaction:\n"
        "   - What action is being performed? (cutting, gripping, piercing, etc.)\n"
        "   - What forces are involved? (compression, tension, shear, etc.)\n"
        "   - What material properties matter? (hardness, flexibility, texture, etc.)\n"
        "   - What geometric features are needed? (sharp edge, pointed tip, flat surface, etc.)\n\n"
        "2. Focus ONLY on the tool's properties, not the target object's properties.\n"
        "   - Example: For 'cutting paper', focus on the scissors' sharp_edge, NOT on paper's thinness.\n\n"
        "3. Use simple, atomic, snake_case attribute names.\n"
        "   - Good: sharp_edge, rigid_body, pointed_tip, high_friction_surface\n"
        "   - Bad: ability_to_cut_things, must_be_sharp_and_durable\n\n"
        "4. Assign priority weights to each attribute:\n"
        "   - Critical attributes (without which the task is impossible) should get HIGH weights (0.5-0.8)\n"
        "   - Important but secondary attributes should get MEDIUM weights (0.2-0.4)\n"
        "   - Nice-to-have attributes should get LOW weights (0.05-0.15)\n"
        "   - Sum of all weights must equal exactly 1.0\n"
        "   - DO NOT make all weights equal - prioritize the most important attributes\n\n"
        "5. Consider the example below:\n"
        "   Task: 'open a parcel'\n"
        "   Analysis: Opening a parcel requires cutting through tape/cardboard. The primary action is cutting, "
        "   which requires a sharp edge. The tool must be rigid enough to apply force without bending. "
        "   A pointed tip helps initiate the cut but is not always essential.\n"
        "   Output: [{\"attribute\": \"sharp_edge\", \"priority\": 0.7}, "
        "{\"attribute\": \"rigid_body\", \"priority\": 0.2}, "
        "{\"attribute\": \"pointed_tip\", \"priority\": 0.1}]\n\n"
        "6. ALWAYS return your answer in the following exact format:\n"
        "```json\n[{\"attribute\": \"attribute_name_1\", \"priority\": weight_1}, "
        "{\"attribute\": \"attribute_name_2\", \"priority\": weight_2}, ...]\n```\n"
        "Where each weight is a float between 0.0 and 1.0, and all weights sum to 1.0."
    )
    
    user_prompt = (
        f"Physical Task: '{task_query}'\n\n"
        f"Analyze this task carefully and identify the essential physical properties of the tool required.\n"
        f"Remember to prioritize the most critical attributes with higher weights."
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=300,
            do_sample=False,
            temperature=0.1,
            top_p=0.95
        )
        
    generated_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    raw_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    
    try:
        json_match = re.search(r"```json\s*(\[.*?\])\s*```", raw_text, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group(1))
            if isinstance(parsed, list) and len(parsed) > 0:
                return validate_and_normalize(parsed)
        
        array_match = re.search(r"(\[.*\])", raw_text, re.DOTALL)
        if array_match:
            parsed = json.loads(array_match.group(1))
            if isinstance(parsed, list) and len(parsed) > 0:
                return validate_and_normalize(parsed)
                
    except Exception as e:
        print(f"[DEBUG] Parsing error: {e}")
        print(f"[DEBUG] Raw output: {raw_text[:200]}...")
        
    return []

def validate_and_normalize(attrs_list):
    """Validates the attribute list and normalizes weights to sum to 1.0."""
    valid_attrs = []
    total_weight = 0.0
    
    for item in attrs_list:
        if not isinstance(item, dict):
            continue
        if 'attribute' not in item or 'priority' not in item:
            continue
        
        attr_name = item['attribute'].strip().lower().replace(' ', '_')
        priority = float(item['priority'])
        
        if priority <= 0 or priority > 1.0:
            continue
            
        valid_attrs.append({
            'attribute': attr_name,
            'priority': priority
        })
        total_weight += priority
    
    if valid_attrs and total_weight > 0:
        for attr in valid_attrs:
            attr['priority'] /= total_weight
            
    valid_attrs.sort(key=lambda x: x['priority'], reverse=True)
    return valid_attrs[:5]

def process_single_image(img_path, common_objects, florence_processor, florence_model, output_folder):
    """Process a single image with Florence-2."""
    print(f"\n[*] Processing image: {img_path.name}")
    
    if not img_path.exists():
        print(f"[-] Image not found: {img_path}")
        return False
    
    image = Image.open(img_path).convert("RGB")
    width, height = image.size
    draw = ImageDraw.Draw(image)
    items_drawn = False

    task_prompt = "<OPEN_VOCABULARY_DETECTION>"
    text_input = task_prompt + ", ".join(common_objects)
    
    inputs = florence_processor(text=text_input, images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        generated_ids = florence_model.generate(**inputs, max_new_tokens=1024, do_sample=False, num_beams=3)
        
    generated_text = florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    results = florence_processor.post_process_generation(generated_text, task=task_prompt, image_size=(width, height)).get(task_prompt, {})

    for bbox, label in zip(results.get('bboxes', []), results.get('bboxes_labels', [])):
        x1, y1, x2, y2 = map(int, bbox)
        draw.rectangle([x1, y1, x2, y2], outline="green", width=4)
        draw.text((x1, max(0, y1 - 15)), label.upper(), fill="green")
        items_drawn = True
        print(f"    -> [SUCCESS] Boxed Object: {label.upper()}")

    if items_drawn:
        output_name = f"solved_{img_path.name}"
        output_path = output_folder / output_name
        image.save(output_path)
        print(f"[+] Task successful! Saved annotated image to '{output_path}'")
        return True
    else:
        print(f"[-] No objects detected in this image.")
        return False

def main():
    # 1. Fetch input directory dynamically using the handler
    selection = input_handler.select_directories_and_range()
    if not selection:
        print("[-] No input directory selected or found. Exiting.")
        return

    # 2. Prompt for user task dynamically
    task_query = input("\n[?] Enter the physical task you want to perform (e.g., 'open a parcel'): ").strip()
    if not task_query:
        print("[-] No task entered. Exiting.")
        return

    print("\n" + "=" * 60)
    print(f"[*] Batch Processing Mode")
    print(f"[*] Task Query: {task_query}")
    print(f"[*] Selected Folders: {len(selection['folders'])}")
    print(f"[*] Image Range: {selection['range'][0]} to {selection['range'][1]}")
    print("=" * 60)

    # 3. Gather and filter images across all selected folders
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
    print(f"[+] First few images: {[f.name for f in image_files[:5]]}...")

    # Create task-specific output folder using output_handler
    output_dir_str = output_handler.get_output_dir(TEAM_MEMBER, MODULE_NAME, task_query)
    OUTPUT_FOLDER = Path(output_dir_str)
    print(f"[*] Output will be saved to: {OUTPUT_FOLDER}")

    # 4. Load SLM & Extract Required Attributes with Weights
    print(f"\n[*] Booting Qwen SLM ({SLM_ID}) on {DEVICE.upper()}...")
    tokenizer = AutoTokenizer.from_pretrained(SLM_ID)
    slm_model = AutoModelForCausalLM.from_pretrained(
        SLM_ID,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE
    ).eval()
    
    print(f"[*] Extracting attributes for task: '{task_query}'...")
    required_attrs = extract_task_attributes(task_query, tokenizer, slm_model)
    print(f"[!] Weighted attributes generated by SLM: {required_attrs}")

    if not required_attrs:
        print("[-] Failed to extract attributes. Exiting.")
        return

    # 5. Graph Traversal: Compute Weighted Scores for Objects
    print("\n[*] Searching Vector Atomic Knowledge Graph with Weighted Scoring...")
    kg = VectorAtomicKnowledgeGraph()
    
    object_scores = {}
    SCORE_THRESHOLD = 0.5 
    
    for item in required_attrs:
        req_attr = item.get("attribute")
        priority = item.get("priority", 0.0)
        
        matched_node_id = kg._find_or_create_attribute_node(req_attr)
        if matched_node_id and matched_node_id in kg.attributes:
            objs_with_attr = kg.attributes[matched_node_id]["objects"]
            print(f"    ├─ '{req_attr}' (Weight: {priority:.2f}) maps to -> {objs_with_attr}")
            
            for obj in objs_with_attr:
                object_scores[obj] = object_scores.get(obj, 0.0) + priority

    print(f"[+] Computed Object Scores: {object_scores}")

    common_objects = [obj for obj, score in object_scores.items() if score >= SCORE_THRESHOLD]
    print(f"[+] Graph identified objects meeting >= {int(SCORE_THRESHOLD * 100)}% threshold: {common_objects}")

    # ULTIMATE FALLBACK LOGIC
    if not common_objects:
        print("[-] No known objects met the required score threshold.")
        print("[*] Trying with top 3 highest scoring objects instead...")
        sorted_objects = sorted(object_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        common_objects = [obj for obj, score in sorted_objects if score > 0]
        print(f"[+] Fallback objects: {common_objects}")
        
        if not common_objects:
            print(f"[-] Graph is entirely empty for these properties.")
            print(f"[*] ULTIMATE FALLBACK: Passing raw task '{task_query}' directly to Vision Model.")
            common_objects = [task_query]

    # Free up SLM memory
    del slm_model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 6. Load Florence-2
    print(f"\n[*] Booting Florence-2 on {DEVICE.upper()}...")
    florence_id = "florence-community/Florence-2-base-ft"
    florence_processor = AutoProcessor.from_pretrained(florence_id)
    florence_model = Florence2ForConditionalGeneration.from_pretrained(
        florence_id, torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32
    ).to(DEVICE).eval()

    # 7. Process all images
    print(f"\n[*] Starting batch processing of {len(image_files)} images...")
    successful_count = 0
    failed_count = 0
    
    log_file = OUTPUT_FOLDER / "processing_log.txt"
    with open(log_file, "w") as f:
        f.write(f"Batch Processing Log\n")
        f.write(f"Task: {task_query}\n")
        f.write(f"Attributes: {required_attrs}\n")
        f.write(f"Objects: {common_objects}\n")
        f.write(f"{'='*60}\n\n")
    
    for img_path in tqdm(image_files, desc="Processing images"):
        success = process_single_image(img_path, common_objects, florence_processor, florence_model, OUTPUT_FOLDER)
        
        with open(log_file, "a") as f:
            status = "SUCCESS" if success else "FAILED"
            f.write(f"{img_path.name}: {status}\n")
        
        if success:
            successful_count += 1
        else:
            failed_count += 1
    
    print("\n" + "=" * 60)
    print("[*] Batch Processing Complete!")
    print(f"[+] Successful: {successful_count}")
    print(f"[-] Failed: {failed_count}")
    print(f"[*] Total: {len(image_files)}")
    print(f"[*] Results saved to: {OUTPUT_FOLDER}")
    print(f"[*] Log file: {log_file}")
    print("=" * 60)

if __name__ == "__main__":
    main()