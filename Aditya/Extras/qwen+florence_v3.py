import os
import json
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import (
    AutoProcessor, 
    AutoModelForMultimodalLM, 
    Florence2ForConditionalGeneration,
    AutoModelForCausalLM,
    AutoTokenizer
)
from qwen_vl_utils import process_vision_info

# ==========================================
# 1. GLOBAL CONFIGURATION & MODEL LOADING
# ==========================================
# Force DEVICE to CPU strictly
DEVICE = "cpu"
SCRIPT_DIR = Path(__file__).parent
OUTPUT_FOLDER = SCRIPT_DIR / "Annotated_Images"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# Define the log file path
LOG_FILEPATH = SCRIPT_DIR / "qwen+slm+florence_v3.jsonl"

print(f"[*] Booting up models on {DEVICE.upper()} (This will use System RAM and may take a while)...")

# 1. Load Qwen VLM (Vision Engine)
print("[*] Loading Qwen2.5-VL-3B-Instruct (VLM)...")
vlm_id = "Qwen/Qwen2.5-VL-3B-Instruct"
vlm_processor = AutoProcessor.from_pretrained(vlm_id)
vlm_model = AutoModelForMultimodalLM.from_pretrained(
    vlm_id, 
    device_map="cpu",           # Force to CPU
    torch_dtype=torch.float32   # Standard precision for best CPU compatibility
)

# 2. Load Qwen SLM (Logic Filter Engine)
print("[*] Loading Qwen2.5-3B-Instruct (SLM)...")
slm_id = "Qwen/Qwen2.5-3B-Instruct"
slm_tokenizer = AutoTokenizer.from_pretrained(slm_id)
slm_model = AutoModelForCausalLM.from_pretrained(
    slm_id,
    device_map="cpu",           # Force to CPU
    torch_dtype=torch.float32   # Standard precision for best CPU compatibility
)

# 3. Load Florence-2 (Grounding Engine)
print("[*] Loading Florence-2-base-ft...")
florence_id = "florence-community/Florence-2-base-ft"
florence_processor = AutoProcessor.from_pretrained(florence_id)
# Map Florence explicitly to CPU
florence_model = Florence2ForConditionalGeneration.from_pretrained(florence_id).to(DEVICE).eval()

print("[*] All engines loaded successfully on CPU!\n")

# ==========================================
# 2. QWEN VLM LOGIC (VISION)
# ==========================================
def get_objects_from_vlm(image_path, user_query):
    """
    Passes the image and user question to the VLM.
    """
    clean_image_path = str(Path(image_path).resolve())
    
    system_prompt = (
        f"Look at the provided image. {user_query}. "
        "Identify the specific objects in the image that answer this. "
        "Return strictly a comma-separated list of short noun phrases (e.g., 'knife, red scissors'). "
        "Do not include any conversational text, explanations, or punctuation other than commas."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": clean_image_path},
                {"type": "text", "text": system_prompt}
            ]
        }
    ]

    text_prompt = vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = vlm_processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(DEVICE)

    outputs = vlm_model.generate(**inputs, max_new_tokens=50, do_sample=False)
    
    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    clean_response = vlm_processor.decode(generated_ids, skip_special_tokens=True).strip()
    
    return clean_response

# ==========================================
# 3. QWEN SLM LOGIC (FILTER)
# ==========================================
def filter_objects_with_slm(task_prompt, candidate_objects_list):
    """
    Asks the text-only SLM to logically verify if the VLM's objects 
    are actually viable for the user's task.
    """
    if not candidate_objects_list:
        return []
        
    objects_str = ", ".join(candidate_objects_list)
    
    system_prompt = (
        f"A user wants to accomplish this task: '{task_prompt}'. "
        f"An AI vision system found these objects in the room: [{objects_str}]. "
        "Your job is to act as a strict logical filter. Evaluate each object and keep ONLY the ones "
        "that are practically and safely viable for this specific task. "
        "Return STRICTLY a comma-separated list of the approved viable objects. "
        "Do not output any introductory text, explanations, or formatting. Just the comma-separated words. "
        "If absolutely none of the objects are viable, output strictly 'NONE'."
    )
    
    messages = [
        {"role": "system", "content": "You are a highly logical filter. You strictly follow output formatting instructions."},
        {"role": "user", "content": system_prompt}
    ]
    
    text = slm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = slm_tokenizer([text], return_tensors="pt").to(DEVICE)
    
    generated_ids = slm_model.generate(
        **model_inputs,
        max_new_tokens=50,
        do_sample=False
    )
    
    # Extract only the newly generated tokens
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = slm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    
    if response.upper() == "NONE" or not response:
        return []
        
    # Strip whitespace, remove duplicates, and cast to list
    filtered_list = list({q.strip() for q in response.split(',') if q.strip()})
    return filtered_list

# ==========================================
# 4. FLORENCE DETECTION LOGIC
# ==========================================
def run_florence(task_prompt, text_input, image):
    """Runs Florence-2 to extract bounding boxes for a specific phrase."""
    prompt = task_prompt + text_input 
    inputs = florence_processor(text=prompt, images=image, return_tensors="pt").to(DEVICE)
    
    generated_ids = florence_model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        do_sample=False,
        num_beams=3
    )
    
    generated_text = florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    
    return florence_processor.post_process_generation(
        generated_text, 
        task=task_prompt, 
        image_size=(image.width, image.height)
    )

def draw_bounding_boxes(image, detection_results):
    """Draws boxes and labels on the image based on Florence-2 output."""
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)
    
    bboxes = detection_results.get('bboxes', [])
    labels = detection_results.get('bboxes_labels', detection_results.get('labels', []))

    for bbox, label in zip(bboxes, labels):
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        text_position = (x1, max(0, y1 - 15))
        draw.text(text_position, str(label), fill="red")

    return draw_image

# ==========================================
# 5. GUI & MAIN PIPELINE
# ==========================================
def select_image_from_parent_folder():
    """Opens a GUI file dialog starting at the parent directory."""
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        initialdir=Path.cwd(),
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.webp *.bmp")]
    )
    root.destroy()
    return file_path

if __name__ == "__main__":
    print("Starting VLM -> SLM -> Florence continuous pipeline...")
    
    take_prompt_input = input("Do you want to keep the prompt dynamic? (yes/no): ").strip().lower()
    take_prompt = take_prompt_input in ("yes", "y", "true", "1")
    user_query = "What can I use to open a parcel?"
    
    # Example automated loop
    for i in range(1, 11):
        # 1. Setup Image
        image_path = f"Open_Parcel/{i}.jpg"
        if not os.path.exists(image_path):
            print(f"[-] Image {image_path} not found. Skipping...")
            continue
            
        image_name = f"{i}.jpg"
        print(f"\n==========================================")
        print(f"Processing image: {image_name}")
        
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"[-] Error loading image: {e}")
            continue
            
        # 2. Get User Query
        if take_prompt:
            custom_query = input(f"\nEnter your reasoning query (default: '{user_query}'): ").strip()
            if custom_query:
                user_query = custom_query
        
        # 3. Qwen VLM Processing (Vision)
        print(f"[*] Asking Qwen VLM: '{user_query}'")
        vlm_objects_string = get_objects_from_vlm(image_path, user_query)
        
        # Clean and deduplicate VLM raw output
        raw_queries = list({q.strip() for q in vlm_objects_string.split(',') if q.strip()})
        print(f"[+] VLM found: {raw_queries}")
        
        if not raw_queries:
            print("[-] VLM could not find any objects matching your query.")
            continue
            
        # 4. Qwen SLM Processing (Logic Filter)
        print(f"[*] Asking Qwen SLM to verify viability...")
        slm_approved_queries = filter_objects_with_slm(user_query, raw_queries)
        print(f"[+] SLM approved: {slm_approved_queries}")
        
        if not slm_approved_queries:
            print("[-] SLM determined none of the objects are logically viable for this task.")
            continue
            
        # 5. Florence Processing (Grounding)
        print(f"[*] Asking Florence to locate {len(slm_approved_queries)} approved items...")
        task = "<OPEN_VOCABULARY_DETECTION>"
        all_bboxes = []
        all_labels = []
        florence_detections = []
        
        for q in slm_approved_queries:
            print(f"  -> Searching for: '{q}'")
            results = run_florence(task, q, image)
            od_results = results.get(task, {})
            
            if isinstance(od_results, str):
                continue
            
            bboxes = od_results.get('bboxes', [])
            labels = od_results.get('bboxes_labels', od_results.get('labels', []))
            
            all_bboxes.extend(bboxes)
            all_labels.extend(labels)
            
            for bbox, label in zip(bboxes, labels):
                florence_detections.append({
                    "label": label,
                    "bbox": bbox
                })
            
        if not all_bboxes:
             print("\n[-] Florence could not physically locate the approved objects.")
             continue
             
        merged_results = {
            'bboxes': all_bboxes,
            'bboxes_labels': all_labels
        }
        
        # 6. Output and Save Image
        annotated_image = draw_bounding_boxes(image, merged_results)
        output_filename = OUTPUT_FOLDER / f"pipeline_output_{image_name}"
        annotated_image.save(output_filename)
        print(f"[+] Saved output to '{output_filename}'")
        
        # 7. Log Data to JSONL File (Tracking both VLM and SLM outputs)
        log_entry = {
            "image_filename": image_name,
            "prompt": user_query,
            "vlm_raw_objects": raw_queries,
            "slm_filtered_objects": slm_approved_queries,
            "florence_detections": florence_detections
        }
        
        with open(LOG_FILEPATH, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')
            
        print(f"[+] Appended dataset entry to '{LOG_FILEPATH.name}'")