import os
from pathlib import Path

MODEL_BASE_DIR = "/media/anshdeep-singh/Aditya/HuggingFaceModels"
os.environ["HF_HOME"] = MODEL_BASE_DIR
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForMultimodalLM, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
import re
import json

def extract_box_data(text, img_w, img_h):
    # Pattern -1: JSON Array Format
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

    # Pattern 0: Qwen2.5-VL Native Output Tokens
    qwen_match = re.search(r"<\|object_ref_start\|>(.*?)<\|object_ref_end\|>.*?\[([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\]", text)
    if qwen_match:
        lbl = qwen_match.group(1).strip()
        y1, x1, y2, x2 = map(float, qwen_match.groups()[1:])
        scale = 1000.0 if max(x1, y1, x2, y2) > 1.0 else 1.0
        return lbl, int((x1/scale)*img_w), int((y1/scale)*img_h), int((x2/scale)*img_w), int((y2/scale)*img_h)

    # Pattern 1: Standard format "object_name [x1, y1, x2, y2]"
    match = re.search(r"(.*?)\s*\[([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\]", text)
    if match:
        lbl = match.group(1).strip()
        x1, y1, x2, y2 = map(float, match.groups()[1:])
        scale = 1000.0 if max(x1, y1, x2, y2) > 1.0 else 1.0
        return lbl, int((x1/scale)*img_w), int((y1/scale)*img_h), int((x2/scale)*img_w), int((y2/scale)*img_h)
    
    # Pattern 2: Try to find any object name before coordinates
    match = re.search(r"([A-Za-z\s]+?)\s*\[([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\]", text)
    if match:
        lbl = match.group(1).strip()
        x1, y1, x2, y2 = map(float, match.groups()[1:])
        scale = 1000.0 if max(x1, y1, x2, y2) > 1.0 else 1.0
        return lbl, int((x1/scale)*img_w), int((y1/scale)*img_h), int((x2/scale)*img_w), int((y2/scale)*img_h)
    
    # Pattern 3: Coordinate fallback
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
    """Clean up the label by removing common unwanted text"""
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
    
    # Send inputs to the determined device
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

def process_all_images(task, model, processor, project_root, device):
    input_dir = project_root / "Anshdeep_Singh" / "Open_Parcel"
    output_dir = project_root / "Output_Images" / "Aditya" / "Stages"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_dir.exists():
        print(f"Input_Images directory not found at: {input_dir}")
        return
    
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    images = []
    
    for p in input_dir.iterdir():
        if p.is_file() and p.suffix.lower() in valid_exts:
            try:
                num = int(p.stem)
                images.append((num, p))
            except ValueError:
                continue
    
    images.sort(key=lambda x: x[0])
    images = [p for _, p in images]
    
    if not images:
        print(f"No image files found in '{input_dir}'. Add test images and run again.")
        return
    
    print(f"\n[*] Found {len(images)} images to process")
    print(f"[*] Task: '{task}'")
    print(f"[*] Processing all images on {device.upper()}...\n")
    
    processed_count = 0
    detected_objects = []
    
    for idx, img_path in enumerate(images, 1):
        print(f"\n[*] Processing image {idx}/{len(images)}: {img_path.name}")
        success, detected_label, bbox = analyze_image(img_path, task, model, processor, device)
        if success:
            processed_count += 1
            if detected_label:
                detected_objects.append((img_path.name, detected_label))
                
            img = Image.open(img_path).convert("RGB")
            draw = ImageDraw.Draw(img)
            xmin, ymin, xmax, ymax = bbox
            
            draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
            draw.text((xmin, max(0, ymin - 15)), detected_label, fill="red")
            
            save_path = output_dir / img_path.name
            img.save(save_path)
            print(f"    -> Saved annotated image to: {save_path.relative_to(project_root)}")
    
    print(f"\n" + "="*50)
    print(f"[*] Processing complete! Found objects in {processed_count}/{len(images)} images")
    
    if detected_objects:
        print("\n[*] Detected Objects Summary:")
        for img_name, obj_label in detected_objects:
            print(f"    {img_name}: {obj_label}")
    else:
        print("[*] No objects were detected in any image")
    print("="*50)

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    input_dir = project_root / "Anshdeep_Singh" / "Open_Parcel"
    
    print(project_root)
    
    if not input_dir.exists():
        print(f"Input_Images directory not found at: {input_dir}")
        exit()

    print(input_dir)
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    
    # --- Device Selection Logic ---
    if torch.cuda.is_available():
        device = "cuda"
        print(f"\n[INFO] CUDA is available. Loading model onto GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("\n[INFO] CUDA is NOT available. Falling back to CPU. (Note: 4-bit quantization requires GPU, so this might run slowly or fail on CPU)")

    print("Loading model... This may take a moment.")
    
    processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
    
    # 4-bit quantization is currently only supported on GPUs via bitsandbytes
    if device == "cuda":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",               
            bnb_4bit_use_double_quant=True,          
            bnb_4bit_compute_dtype=torch.bfloat16    
        )
        model = AutoModelForMultimodalLM.from_pretrained(
            model_id, 
            device_map={"": 0}, # Explicitly map entirely to GPU 0 
            quantization_config=quantization_config,
            local_files_only=True
        )
    else:
        # Fallback for CPU (BitsAndBytes 4-bit doesn't work on CPU)
        model = AutoModelForMultimodalLM.from_pretrained(
            model_id, 
            device_map={"": "cpu"},
            torch_dtype=torch.float32, 
            local_files_only=True
        )
    
    print("Model loaded successfully!")
    
    task = input("Enter your task (e.g., 'open a parcel', 'cut paper', 'hammer nail'): ")
    if not task:
        print("No task provided. Exiting.")
        exit(1)
    
    process_all_images(task, model, processor, project_root, device)
