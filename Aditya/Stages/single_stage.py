import os
import sys
from pathlib import Path
import json
import re
import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForMultimodalLM, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

# [PATH CHANGE] Resolve Repo Root (RM/) dynamically based on file location
# This script is in RM/Aditya/Stages/single_stage.py (3 levels deep)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(REPO_ROOT))

# [PATH CHANGE] Import the shared handlers from the repo root
import input_handler
import output_handler

# [PATH CHANGE] Update model base directory to use the shared repo Models folder
MODEL_BASE_DIR = str(REPO_ROOT / "Models")
os.environ["HF_HOME"] = MODEL_BASE_DIR
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def extract_box_data(text, img_w, img_h):
    json_match = re.search(r'\[\s*\{.*?\}\s*\]', text, re.DOTALL)
    if json_match:
        try:
            items = json.loads(json_match.group(0))
            if items and isinstance(items, list) and len(items) > 0:
                item = items[0]
                if "bbox_2d" in item and "label" in item:
                    lbl = str(item["label"]).strip()
                    y1, x1, y2, x2 = map(float, item["bbox_2d"])
                    scale = 1000.0 if max(x1, y1, x2, y2) > 1.0 else 1.0
                    return lbl, int((x1/scale)*img_w), int((y1/scale)*img_h), int((x2/scale)*img_w), int((y2/scale)*img_h)
        except Exception:
            pass

    qwen_match = re.search(r"<\|object_ref_start\|>(.*?)<\|object_ref_end\|>.*?\[([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\]", text)
    if qwen_match:
        lbl = qwen_match.group(1).strip()
        y1, x1, y2, x2 = map(float, qwen_match.groups()[1:])
        scale = 1000.0 if max(x1, y1, x2, y2) > 1.0 else 1.0
        return lbl, int((x1/scale)*img_w), int((y1/scale)*img_h), int((x2/scale)*img_w), int((y2/scale)*img_h)

    match = re.search(r"(.*?)\s*\[([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\]", text)
    if match:
        lbl = match.group(1).strip()
        x1, y1, x2, y2 = map(float, match.groups()[1:])
        scale = 1000.0 if max(x1, y1, x2, y2) > 1.0 else 1.0
        return lbl, int((x1/scale)*img_w), int((y1/scale)*img_h), int((x2/scale)*img_w), int((y2/scale)*img_h)
    
    match = re.search(r"([A-Za-z\s]+?)\s*\[([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\]", text)
    if match:
        lbl = match.group(1).strip()
        x1, y1, x2, y2 = map(float, match.groups()[1:])
        scale = 1000.0 if max(x1, y1, x2, y2) > 1.0 else 1.0
        return lbl, int((x1/scale)*img_w), int((y1/scale)*img_h), int((x2/scale)*img_w), int((y2/scale)*img_h)
    
    match = re.search(r"\[([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\]", text)
    if match:
        x1, y1, x2, y2 = map(float, match.groups())
        scale = 1000.0 if max(x1, y1, x2, y2) > 1.0 else 1.0
        before_coords = text[:match.start()].strip()
        before_coords = re.sub(r'<\|.*?\|>', ' ', before_coords)
        words = re.findall(r'[A-Za-z]+', before_coords)
        if words:
            lbl = ' '.join(words[-3:]) 
        else:
            lbl = "Object"
        return lbl, int((x1/scale)*img_w), int((y1/scale)*img_h), int((x2/scale)*img_w), int((y2/scale)*img_h)
    
    return None

def clean_label(label):
    label = re.sub(r'<\|.*?\|>', '', label)
    label = re.sub(r'[{}"\']', '', label)
    label = re.sub(r'(?i)(bb|bounding)?\s*box\s*\d*', '', label)
    label = re.sub(r'(?i)(b_2d|bbox)', '', label) 
    label = re.sub(r'^\s*[-:]\s*', '', label)
    label = re.sub(r'[,;]\s*$', '', label)
    label = re.sub(r'\b\d+\b', '', label)
    label = ' '.join(label.split())
    
    if not label or len(label.strip()) < 2:
        label = "Object"
    
    return label.strip()

def analyze_image(image_path, task, model, processor, device):
    img = Image.open(image_path)
    width, height = img.size
    
    prompt = f"Locate a specific tool or item that can be used to {task}. State its exact specific name (e.g., 'scissors', 'knife') and provide its bounding box."
    
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": str(image_path.resolve())},
            {"type": "text", "text": prompt}
        ]
    }]
    
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text_prompt], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt"
    ).to(device)
    
    outputs = model.generate(**inputs, max_new_tokens=150, do_sample=False)
    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    raw_response = processor.decode(generated_ids, skip_special_tokens=False)
    
    print(f"    Raw response: {raw_response.strip()}")
    
    detection = extract_box_data(raw_response, width, height)
    if detection:
        label, xmin, ymin, xmax, ymax = detection
        label_clean = clean_label(label)
        
        if label_clean.lower() in ["tool", "item", "object"]:
            print(f"    -> Warning: Model returned generic term. See raw response.")
        
        print(f"    -> Detected: {label_clean}")
        return True, label_clean, (xmin, ymin, xmax, ymax)
    else:
        print(f"    -> No suitable object found")
        return False, None, None

def process_selected_images(task, model, processor, input_dir, device, start_idx, end_idx):
    
    # [PATH CHANGE] Dynamically fetch the output path via output_handler 
    output_dir_str = output_handler.get_output_dir(
        team_member="Aditya", 
        module_name="Stages", 
        user_query=task
    )
    output_dir = Path(output_dir_str)
    print(f"[*] Output will be saved to: {output_dir}")
    
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    
    existing_images = {}
    for p in input_dir.iterdir():
        if p.is_file() and p.suffix.lower() in valid_exts:
            try:
                num = int(p.stem)
                existing_images[num] = p
            except ValueError:
                continue
                
    if not existing_images:
        print(f"No numbered image files found in '{input_dir}'.")
        return
        
    if start_idx > end_idx:
        print("Error: Start index cannot be greater than end index.")
        return
        
    print(f"\n[*] Task: '{task}'")
    print(f"[*] Processing images {start_idx} to {end_idx} for directory '{input_dir.name}' on {device.upper()}...\n")
    
    processed_count = 0
    detected_objects = []
    json_results = []
    
    for idx in range(start_idx, end_idx + 1):
        img_path = existing_images.get(idx)
        
        if not img_path:
            for ext in valid_exts:
                candidate = input_dir / f"{idx}{ext}"
                if candidate.exists():
                    img_path = candidate
                    break
                    
        if not img_path or not img_path.exists():
            print(f"\n[*] Skipping index {idx}: Image file not found.")
            continue
            
        print(f"\n[*] Processing image {idx}/{end_idx}: {img_path.name}")
        success, detected_label, bbox = analyze_image(img_path, task, model, processor, device)
        if success:
            processed_count += 1
            xmin, ymin, xmax, ymax = bbox
            
            if detected_label:
                detected_objects.append((img_path.name, detected_label))
            
            json_results.append({
                "image_name": img_path.name,
                "label": detected_label,
                "bbox": [xmin, ymin, xmax, ymax]
            })
                
            img = Image.open(img_path).convert("RGB")
            draw = ImageDraw.Draw(img)
            
            draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
            draw.text((xmin, max(0, ymin - 15)), detected_label, fill="red")
            
            save_path = output_dir / img_path.name
            img.save(save_path)
            
            # [PATH CHANGE] Replaced project_root reference with REPO_ROOT for logging relative paths
            print(f"    -> Saved annotated image to: {save_path.relative_to(REPO_ROOT)}")
    
    json_output_path = output_dir / "detections.json"
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=4)
        
    print(f"\n" + "="*50)
    print(f"[*] Processing complete for {input_dir.name}! Found objects in {processed_count} images.")
    
    try:
        rel_json_path = json_output_path.relative_to(REPO_ROOT)
    except ValueError:
        rel_json_path = json_output_path
    print(f"[*] Saved JSON detection data to: {rel_json_path}")
    print("="*50)


if __name__ == "__main__":
    print(f"Using Model Directory: {MODEL_BASE_DIR}")

    # [PATH CHANGE] Fetch single input directory via input_handler
    selected_dir_str = input_handler.select_directory()
    if not selected_dir_str:
        print("No directories selected. Exiting.")
        exit()
        
    selected_dirs = [Path(selected_dir_str)]

    print(f"\nSelected 1 directory to process.")
    
    # 2. Ask for the image range globally
    print("\n--- Image Range Configuration ---")
    try:
        start_input = input("Enter start image index for the directory (default 1): ").strip()
        global_start = int(start_input) if start_input else 1
        
        end_input = input("Enter end image index for the directory (default: maximum available): ").strip()
        global_end = int(end_input) if end_input else float('inf')
    except ValueError:
        print("Invalid input. Using full range.")
        global_start = 1
        global_end = float('inf')

    # 3. Ask for the specific task for the directory
    directory_configs = {}
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    
    print("\n--- Task Configuration ---")
    for d in selected_dirs:
        existing_nums = []
        for p in d.iterdir():
            if p.is_file() and p.suffix.lower() in valid_exts:
                try:
                    existing_nums.append(int(p.stem))
                except ValueError:
                    pass
                    
        if not existing_nums:
            print(f"\n[!] Warning: No numbered images found in '{d.name}'. Skipping.")
            continue
            
        max_n = max(existing_nums)
        
        # Clamp the indices safely
        dir_start = max(1, global_start)
        dir_end = min(max_n, int(global_end)) if global_end != float('inf') else max_n
        
        print(f"\nConfiguring '{d.name}' (Will process images {dir_start} to {dir_end})")
        while True:
            task_input = input(f"  -> Enter task for this directory (e.g., 'open a parcel'): ").strip()
            if task_input:
                break
            print("  -> Task cannot be empty.")
            
        directory_configs[d] = {
            "task": task_input,
            "start": dir_start,
            "end": dir_end
        }

    # 4. Load the model
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    if torch.cuda.is_available():
        device = "cuda"
        print(f"\n[INFO] CUDA is available. Loading model onto GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("\n[INFO] CUDA is NOT available. Falling back to CPU.")

    print("Loading model... This may take a moment.")
    
    processor = AutoProcessor.from_pretrained(model_id)
    
    if device == "cuda":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",                
            bnb_4bit_use_double_quant=True,          
            bnb_4bit_compute_dtype=torch.bfloat16    
        )
        model = AutoModelForMultimodalLM.from_pretrained(
            model_id, 
            device_map={"": 0}, 
            quantization_config=quantization_config
        )
    else:
        model = AutoModelForMultimodalLM.from_pretrained(
            model_id, 
            device_map={"": "cpu"},
            torch_dtype=torch.float32
        )
    
    print("Model loaded successfully!")

    # 5. Process all configured directories automatically
    for d, config in directory_configs.items():
        process_selected_images(
            task=config["task"], 
            model=model, 
            processor=processor, 
            input_dir=d, 
            device=device,
            start_idx=config["start"],
            end_idx=config["end"]
        )
        
    print("\n[SUCCESS] All selected directories have been processed!")