import torch
import requests
import json
import re
from pathlib import Path
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Florence2ForConditionalGeneration

# ==========================================
# 1. GLOBAL CONFIGURATION
# ==========================================
SCRIPT_DIR = Path(__file__).parent
INPUT_FOLDER = SCRIPT_DIR / "Open_Parcel"
INPUT_FOLDER.mkdir(parents=True, exist_ok=True)

OUTPUT_FOLDER = SCRIPT_DIR / "Annotated_Ollama"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

OLLAMA_MODEL_NAME = "gpt-oss:20b" 
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def format_florence_for_ollama(od_results):
    bboxes = od_results.get('bboxes', [])
    labels = od_results.get('labels', [])
    
    if not bboxes:
        return "No objects detected.", [], []

    formatted_text = "Image Inventory:\n"
    for i, label in enumerate(labels):
        formatted_text += f"[ID: {i}] {label.capitalize()}\n"
        
    return formatted_text, labels, bboxes

def parse_ollama_json(text):
    json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if not json_match:
        json_match = re.search(r'(\{.*?\})', text, re.DOTALL)
        
    if json_match:
        try:
            return json.loads(json_match.group(1).strip())
        except json.JSONDecodeError:
            pass
    return None

def query_ollama_model(prompt):
    payload = {
        "model": OLLAMA_MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0
        }
    }
    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        return response.json().get("response", "")
    except requests.exceptions.RequestException as e:
        print(f"[-] Failed to communicate with Ollama: {e}")
        return ""

    # ==========================================
# 3. CORE HYBRID PIPELINE
# ==========================================
def process_image(image_path, user_task, florence_model, florence_processor):
    image_name = image_path.name
    print(f"\n==========================================")
    print(f"Processing: {image_name}")
    print(f"==========================================")
    
    try:
        original_image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[-] Error loading image '{image_name}': {e}")
        return
        
    width, height = original_image.size

    # --- PHASE 1: FLORENCE-2 (Dense Inventory Scan) ---
    print("[*] Running Florence-2 Dense Inventory Scan...")
    task = "<DENSE_REGION_CAPTION>"
    
    inputs = florence_processor(text=task, images=original_image, return_tensors="pt").to(florence_model.device)
    
    generated_ids = florence_model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        do_sample=False,
        num_beams=3
    )
    
    generated_text = florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    florence_results = florence_processor.post_process_generation(
        generated_text, task=task, image_size=(width, height)
    )
    
    od_results = florence_results.get(task, {})
    inventory_text, labels, bboxes = format_florence_for_ollama(od_results)
    
    if not bboxes:
        print("[-] Florence found nothing in this image.")
        return

    # --- NEW INTERMEDIATE STEP: Draw and save Florence's Raw Output ---
    florence_img = original_image.copy()
    draw_florence = ImageDraw.Draw(florence_img)
    
    print(f"[*] Florence found {len(bboxes)} total objects. Saving raw view...")
    for i in range(len(bboxes)):
        abs_xmin, abs_ymin, abs_xmax, abs_ymax = map(int, bboxes[i])
        raw_label = f"[{i}] {labels[i]}"
        
        draw_florence.rectangle([abs_xmin, abs_ymin, abs_xmax, abs_ymax], outline="green", width=3)
        draw_florence.text((abs_xmin, max(0, abs_ymin - 15)), raw_label, fill="green")
        
    raw_output_filename = f"step1_raw_florence_{image_name}"
    raw_output_path = OUTPUT_FOLDER / raw_output_filename
    florence_img.save(raw_output_path)
    print(f"[+] Saved Florence Raw Output to: {raw_output_path}")

    # --- PHASE 2: OLLAMA 20B (Inference & Filtering) ---
    print("\n[*] Querying Ollama 20B Model for Task Matching...")
    
    prompt = f"""The user needs to accomplish this task: "{user_task}"

Step 1: Brainstorm a broad, diverse list of 5 to 10 specific, physical, inanimate tools that could realistically be used for this task.
CRITICAL RULE: Do NOT suggest or select humans, people, hands, body parts, or animals under any circumstances. Only tools.

Step 2: Look at the following inventory found in an image:
{inventory_text}

Step 3: Act as a strict filter. Compare your list of 5-10 inferred tools against the inventory list. Select ALL items from the inventory that match any of your tools. If multiple valid tools exist, select all of their IDs.

Output your final decision in a strict JSON block like this:
```json
{{
    "inferred_tools": ["tool 1", "tool 2", "tool 3", "tool 4", "tool 5"],
    "matched_ids": [0, 2, 7],
    "reasoning": "brief explanation of why these IDs were chosen"
}}
Note: "matched_ids" must be a list of integers. If no valid items are found, return an empty list []."""

    ollama_response = query_ollama_model(prompt)

    if not ollama_response:
        return

    parsed_data = parse_ollama_json(ollama_response)

    if not parsed_data or "matched_ids" not in parsed_data:
        print("[-] Error: Ollama did not return a valid JSON format with 'matched_ids'.")
        print("Raw Output:", ollama_response)
        return

    matched_ids = parsed_data.get("matched_ids", [])
    inferred_tools = parsed_data.get("inferred_tools", [])
    reasoning = parsed_data.get("reasoning", "No reasoning provided.")

    print(f"\n[+] Expanded Tool List : {inferred_tools}")
    print(f"[+] Model Reasoning    : {reasoning}")

    # --- PHASE 3: DRAWING & SAVING FINAL OUTPUT ---
    if not matched_ids:
        print("[-] Filter applied: No suitable items from the 5-10 tool list were found in the image.")
        return

    final_img = original_image.copy()
    draw_final = ImageDraw.Draw(final_img)
    items_drawn = []

    for selected_id in matched_ids:
        if isinstance(selected_id, int) and 0 <= selected_id < len(bboxes):
            abs_xmin, abs_ymin, abs_xmax, abs_ymax = map(int, bboxes[selected_id])
            chosen_label = labels[selected_id]
            items_drawn.append(f"[ID: {selected_id}] {chosen_label}")
            
            draw_final.rectangle([abs_xmin, abs_ymin, abs_xmax, abs_ymax], outline="red", width=5)
            draw_final.text((abs_xmin, max(0, abs_ymin - 15)), chosen_label.upper(), fill="red")

    if items_drawn:
        print(f"[+] Successfully filtered and highlighted: {', '.join(items_drawn)}")
        final_output_filename = f"step2_final_filtered_{image_name}"
        final_output_path = OUTPUT_FOLDER / final_output_filename
        final_img.save(final_output_path)
        print(f"[+] Saved Final Filtered Output to: {final_output_path}")
    else:
        print("[-] Valid IDs were returned, but they were out of bounds.")


# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    images_to_process = sorted([p for p in INPUT_FOLDER.iterdir() if p.is_file() and p.suffix.lower() in valid_extensions])
    
    if not images_to_process:
        print(f"[-] No images found in '{INPUT_FOLDER}'. Please add images and run again.")
    else:
        print("\n" + "="*50)
        user_input_task = input("Enter the task you want to perform (e.g., 'open the parcel'): ")
        print("="*50 + "\n")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"[*] Loading Florence-2 into VRAM ({device})...")
        florence_id = "florence-community/Florence-2-base-ft"
        florence_model = Florence2ForConditionalGeneration.from_pretrained(
            florence_id, 
            torch_dtype=torch.bfloat16
        ).to(device).eval()
        florence_processor = AutoProcessor.from_pretrained(florence_id)
        
        print(f"[*] Routing reasoning requests to Ollama Model: '{OLLAMA_MODEL_NAME}'")
        
        for img_path in images_to_process:
            process_image(img_path, user_input_task, florence_model, florence_processor)
            
        print(f"\n[*] All {len(images_to_process)} images processed successfully!")
