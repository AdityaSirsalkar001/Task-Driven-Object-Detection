import os
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import (
    AutoProcessor, 
    AutoModelForMultimodalLM, 
    Florence2ForConditionalGeneration
)
from qwen_vl_utils import process_vision_info

# ==========================================
# 1. GLOBAL CONFIGURATION & MODEL LOADING
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCRIPT_DIR = Path(__file__).parent
OUTPUT_FOLDER = SCRIPT_DIR / "Annotated_Images"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

print(f"[*] Booting up models on {DEVICE}...")

# Load Qwen (Reasoning Engine)
print("[*] Loading Qwen2.5-VL-3B-Instruct...")
qwen_id = "Qwen/Qwen2.5-VL-3B-Instruct"
qwen_processor = AutoProcessor.from_pretrained(qwen_id)
qwen_model = AutoModelForMultimodalLM.from_pretrained(
    qwen_id, 
    device_map="auto", # Allows fallback to CPU RAM if GPU VRAM is full
    dtype=torch.bfloat16
)

# Load Florence-2 (Grounding/Detection Engine)
print("[*] Loading Florence-2-base-ft...")
florence_id = "florence-community/Florence-2-base-ft"
florence_processor = AutoProcessor.from_pretrained(florence_id)
florence_model = Florence2ForConditionalGeneration.from_pretrained(florence_id).to(DEVICE).eval()

print("[*] All engines loaded successfully!\n")

# ==========================================
# 2. QWEN REASONING LOGIC
# ==========================================
def get_objects_from_qwen(image_path, user_query):
    """
    Passes the image and user question to Qwen. Forces Qwen to reply ONLY 
    with a comma-separated list of items found in the image.
    """
    clean_image_path = str(Path(image_path).resolve())
    
    # Strict prompt to prevent Qwen from outputting conversational text
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

    text_prompt = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = qwen_processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(DEVICE)

    outputs = qwen_model.generate(**inputs, max_new_tokens=50, do_sample=False)
    
    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    clean_response = qwen_processor.decode(generated_ids, skip_special_tokens=True).strip()
    
    return clean_response

# ==========================================
# 3. FLORENCE DETECTION LOGIC
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
# 4. GUI & MAIN PIPELINE
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
    print("Starting Qwen + Florence continuous pipeline...")
    take_prompt= input("Do you want to keep prompt fixed or dynamic (yes/no): ").strip()
    take_prompt= take_prompt in ("Yes", "y", "yes", "true", "1")
    user_query="What can I use to open a parcel?"
    while True:
        # 1. Select Image
        image_path = select_image_from_parent_folder()
        if not image_path:
            print("No image selected. Exiting...")
            break
            
        print(f"\n==========================================")
        print(f"Selected image: {Path(image_path).name}")
        
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"[-] Error loading image: {e}")
            continue
            
        # 2. Get User Query
        if(take_prompt):
            user_query = input("\nEnter your reasoning query (e.g., 'What can be used to open a parcel?'): ")
        
        # 3. Qwen Processing (Reasoning)
        print("[*] Asking Qwen to analyze the image...")
        qwen_objects_string = get_objects_from_qwen(image_path, user_query)
        print(f"[+] Qwen identified: '{qwen_objects_string}'")
        
        # Split Qwen's comma-separated string into a clean list
        queries = [q.strip() for q in qwen_objects_string.split(',') if q.strip()]
        
        if not queries:
            print("[-] Qwen could not find any objects matching your query.")
            continue
            
        # 4. Florence Processing (Grounding)
        print(f"[*] Asking Florence to locate {len(queries)} items...")
        task = "<OPEN_VOCABULARY_DETECTION>"
        all_bboxes = []
        all_labels = []
        
        for q in queries:
            print(f"  -> Searching for: '{q}'")
            results = run_florence(task, q, image)
            od_results = results.get(task, {})
            
            if isinstance(od_results, str):
                continue
                
            all_bboxes.extend(od_results.get('bboxes', []))
            all_labels.extend(od_results.get('bboxes_labels', od_results.get('labels', [])))
            
        if not all_bboxes:
             print("\n[-] Florence could not locate the objects Qwen suggested.")
             continue
             
        merged_results = {
            'bboxes': all_bboxes,
            'bboxes_labels': all_labels
        }
        
        # 5. Output and Save
        annotated_image = draw_bounding_boxes(image, merged_results)
        
        output_filename = OUTPUT_FOLDER / f"pipeline_output_{Path(image_path).name}"
        annotated_image.save(output_filename)
        annotated_image.show()
        print(f"[+] Saved output to '{output_filename}'")
