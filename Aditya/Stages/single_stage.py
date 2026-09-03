import torch
from pathlib import Path
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForMultimodalLM
from qwen_vl_utils import process_vision_info
import re
import os
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

def analyze_image(image_path, task, model, processor):
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
    ).to(model.device)
    
    outputs = model.generate(**inputs, max_new_tokens=150, do_sample=False)
    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    raw_response = processor.decode(generated_ids, skip_special_tokens=False)
    
    # Print the raw response to debug what the model actually says
    print(f"    Raw response: {raw_response.strip()}")
    
    detection = extract_box_data(raw_response, width, height)
    if detection:
        label, xmin, ymin, xmax, ymax = detection
        label_clean = clean_label(label)
        
        if label_clean.lower() in ["tool", "item", "object"]:
            print(f"    -> Warning: Model returned generic term. See raw response.")
        
        print(f"    -> Detected: {label_clean}")
        # Return the coordinates along with the label so we can draw them in the main loop
        return True, label_clean, (xmin, ymin, xmax, ymax)
    else:
        print(f"    -> No suitable object found")
        return False, None, None

def process_all_images(task, model, processor):
    project_root = Path(__file__).parent.parent.parent.parent
    input_dir = project_root / "Input_Images"
    
    # Define and create output directory based on your hierarchical folder structure
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
    print(f"[*] Processing all images...\n")
    
    processed_count = 0
    detected_objects = []
    
    for idx, img_path in enumerate(images, 1):
        print(f"\n[*] Processing image {idx}/{len(images)}: {img_path.name}")
        success, detected_label, bbox = analyze_image(img_path, task, model, processor)
        if success:
            processed_count += 1
            if detected_label:
                detected_objects.append((img_path.name, detected_label))
                
            # Open image, draw bounding box, and save to Output_Images/Aditya/Stages
            img = Image.open(img_path).convert("RGB")
            draw = ImageDraw.Draw(img)
            xmin, ymin, xmax, ymax = bbox
            
            # Draw rectangle and label
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
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    
    print("Loading model... This may take a moment.")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForMultimodalLM.from_pretrained(model_id, device_map="auto", dtype=torch.bfloat16)
    print("Model loaded successfully!")
    
    task = input("Enter your task (e.g., 'open a parcel', 'cut paper', 'hammer nail'): ")
    if not task:
        print("No task provided. Exiting.")
        exit(1)
    
    process_all_images(task, model, processor)