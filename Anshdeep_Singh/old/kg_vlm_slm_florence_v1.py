import os
import gc
import json
import torch
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np

from transformers import (
    AutoProcessor, 
    AutoModelForMultimodalLM, 
    Florence2ForConditionalGeneration,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from qwen_vl_utils import process_vision_info
import kg_manager  # Assumes your sentence-transformer logic is here

# ==========================================
# 1. GLOBAL CONFIGURATION & FOLDERS
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCRIPT_DIR = Path(__file__).parent
INPUT_FOLDER = SCRIPT_DIR / "Open_Parcel"
OUTPUT_FOLDER = SCRIPT_DIR / "Annotated_Attribute_KG"

INPUT_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# 4-bit quantization config for Qwen models
gpu_quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

# ==========================================
# 2. VRAM SWAP MANAGER
# ==========================================
class ModelManager:
    """Safely swaps 4-bit models in and out of 6GB VRAM."""
    def __init__(self):
        self.active_model = None
        self.slm_model = None
        self.slm_tokenizer = None
        self.vlm_model = None
        self.vlm_processor = None
        
        # Load Florence-2 permanently
        print("[*] Loading Florence-2 (Permanent) on GPU...")
        self.florence_id = "florence-community/Florence-2-base-ft"
        self.florence_processor = AutoProcessor.from_pretrained(self.florence_id)
        self.florence_model = Florence2ForConditionalGeneration.from_pretrained(
            self.florence_id,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            device_map=DEVICE
        ).eval()

    def clear_vram(self):
        """Destroys active Qwen models to free space for the next one."""
        if self.slm_model is not None:
            del self.slm_model
            self.slm_model = None
        if self.vlm_model is not None:
            del self.vlm_model
            self.vlm_model = None
            
        gc.collect()
        torch.cuda.empty_cache()
        self.active_model = None

    def load_slm(self):
        """Loads Qwen SLM into GPU."""
        if self.active_model == "SLM": return
        self.clear_vram()
        print("\n[*] 🔄 SWAPPING VRAM: Loading Qwen SLM to GPU...")
        slm_id = "Qwen/Qwen2.5-3B-Instruct"
        if self.slm_tokenizer is None:
            self.slm_tokenizer = AutoTokenizer.from_pretrained(slm_id)
        self.slm_model = AutoModelForCausalLM.from_pretrained(
            slm_id, quantization_config=gpu_quant_config, device_map=DEVICE
        ).eval()
        self.active_model = "SLM"

    def load_vlm(self):
        """Loads Qwen VLM into GPU."""
        if self.active_model == "VLM": return
        self.clear_vram()
        print("\n[*] 🔄 SWAPPING VRAM: Loading Qwen VLM to GPU...")
        vlm_id = "Qwen/Qwen2.5-VL-3B-Instruct"
        if self.vlm_processor is None:
            self.vlm_processor = AutoProcessor.from_pretrained(
                vlm_id, min_pixels=256 * 28 * 28, max_pixels=768 * 28 * 28
            )
        self.vlm_model = AutoModelForMultimodalLM.from_pretrained(
            vlm_id, quantization_config=gpu_quant_config, device_map=DEVICE
        ).eval()
        self.active_model = "VLM"

# ==========================================
# 3. EXTRACTION AND VISION LOGIC
# ==========================================
def extract_attributes(user_query, manager):
    manager.load_slm()
    system_prompt = (
        "You are an expert at physical reasoning. "
        "Describe the physical attributes of an object needed to accomplish the user's task. "
        "Keep it strictly under 10 words. "
        "Example 1: Task: 'cut paper' -> 'small handheld object with sharp blades' "
        "Example 2: Task: 'drink water' -> 'hollow watertight container' "
        "Output ONLY the attribute description, nothing else."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Task: '{user_query}'"}
    ]
    
    text = manager.slm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = manager.slm_tokenizer([text], return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        gen_ids = manager.slm_model.generate(**inputs, max_new_tokens=20, do_sample=False)
        
    gen_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, gen_ids)]
    return manager.slm_tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0].strip().lower()

def vlm_image_scan(image_path, user_query, manager):
    manager.load_vlm()
    clean_path = str(Path(image_path).resolve())
    system_prompt = (
        f"Look at the provided image. The user needs to: '{user_query}'. "
        "Identify the specific objects IN THIS EXACT IMAGE that can be used to accomplish this. "
        "Return strictly a comma-separated list of short noun phrases (e.g., 'red scissors, pocket knife'). "
        "If absolutely nothing in the image can be used, return strictly 'NONE'."
    )
    
    messages = [{"role": "user", "content": [{"type": "image", "image": clean_path}, {"type": "text", "text": system_prompt}]}]
    text_prompt = manager.vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = manager.vlm_processor(
        text=[text_prompt], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    ).to(DEVICE)
    
    with torch.no_grad():
        outputs = manager.vlm_model.generate(**inputs, max_new_tokens=50, do_sample=False)
        
    gen_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    response = manager.vlm_processor.decode(gen_ids, skip_special_tokens=True).strip()
    
    if response == "NONE": return []
    return list({q.strip().lower() for q in response.split(',') if q.strip()})

def run_florence(manager, task_prompt, image, text_input=""):
    inputs = manager.florence_processor(text=task_prompt + text_input, images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        gen_ids = manager.florence_model.generate(**inputs, max_new_tokens=1024, do_sample=False, num_beams=3)
    gen_text = manager.florence_processor.batch_decode(gen_ids, skip_special_tokens=False)[0]
    return manager.florence_processor.post_process_generation(
        gen_text, task=task_prompt, image_size=(image.width, image.height)
    ).get(task_prompt, {})

# ==========================================
# 4. MAIN PIPELINE
# ==========================================
def main():
    manager = ModelManager()
    graph = kg_manager.load_graph()
    
    print("\n" + "="*50)
    user_query = input("Enter the task you want to perform (e.g., 'open the parcel'): ").strip()
    print("="*50 + "\n")
    if not user_query: return

    # 1. Extract Attributes (Swaps SLM to GPU)
    required_attributes = extract_attributes(user_query, manager)
    print(f"\n[*] Extracted Physical Attributes: '{required_attributes}'")

    # 2. Semantic Graph Check
    matched_attr, similarity = kg_manager.semantic_search(required_attributes, graph)
    candidate_objects = []
    
    if matched_attr:
        print(f"[+] GRAPH HIT: Semantic match with '{matched_attr}' (Score: {similarity:.2f})")
        candidate_objects = graph[matched_attr]
        print(f"    -> Associated Graph Objects: {candidate_objects}")
    else:
        print(f"[-] GRAPH MISS: No matching attributes in knowledge graph.")

    images = [p for p in INPUT_FOLDER.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
    
    for img_path in images:
        print(f"\n[*] Processing: {img_path.name}")
        image = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        items_drawn = False

        # 3. Florence Verification (Always on GPU)
        if candidate_objects:
            print("[*] Florence scanning for Graph Objects...")
            results = run_florence(manager, "<DENSE_REGION_CAPTION>", image)
            
            for bbox, label in zip(results.get('bboxes', []), results.get('labels', [])):
                if any(tool.lower() in label.lower() for tool in candidate_objects):
                    x1, y1, x2, y2 = map(int, bbox)
                    draw.rectangle([x1, y1, x2, y2], outline="green", width=4)
                    draw.text((x1, max(0, y1 - 15)), label.upper(), fill="green")
                    items_drawn = True
                    print(f"    -> [SUCCESS] Found Graph Object: {label.upper()}")

        # 4. Fallback Logic (Swaps VLM to GPU)
        if not items_drawn:
            print("[-] Primary tools missing (or graph empty). Engaging VLM Fallback...")
            vlm_discovered_objects = vlm_image_scan(img_path, user_query, manager)
            
            if not vlm_discovered_objects:
                print("[-] VLM determined no viable objects exist in this image.")
                continue
                
            print(f"[+] VLM discovered new objects: {vlm_discovered_objects}")
            
            # Update Knowledge Graph dynamically based on extracted attributes
            existing_objects = graph.get(required_attributes, [])
            updated_objects = list(set(existing_objects + vlm_discovered_objects))
            graph[required_attributes] = updated_objects
            kg_manager.save_graph(graph)
            print(f"[+] Knowledge Graph Updated: '{required_attributes}' -> {updated_objects}")

            # Re-run Florence specifically for newly discovered VLM items
            for obj in vlm_discovered_objects:
                od_results = run_florence(manager, "<OPEN_VOCABULARY_DETECTION>", image, obj)
                for bbox, label in zip(od_results.get('bboxes', []), od_results.get('bboxes_labels', [])):
                    x1, y1, x2, y2 = map(int, bbox)
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
                    draw.text((x1, max(0, y1 - 15)), label.upper() + " (VLM)", fill="red")
                    print(f"    -> [FALLBACK SUCCESS] Boxed VLM Object: {label.upper()}")
        
        output_file = OUTPUT_FOLDER / f"attr_kg_output_{img_path.name}"
        image.save(output_file)
        print(f"[+] Annotated image saved to: '{output_file}'")

    # Clear VRAM when completely done to exit cleanly
    manager.clear_vram()

if __name__ == "__main__":
    main()
