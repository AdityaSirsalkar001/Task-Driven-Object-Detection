import os
import re
import json
import torch
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime

from transformers import (
    AutoProcessor,
    AutoModelForMultimodalLM,
    Florence2ForConditionalGeneration,
    BitsAndBytesConfig
)
from qwen_vl_utils import process_vision_info

# =====================================================================
# 1. DIRECTORY SETUP & HARDWARE CONFIGURATION
# =====================================================================
# [PATH CHANGE] Resolve Project_Root (4 levels up from current file location)
# Current: Project_Root/RM/Aditya/Scene_graph/scene_graph_vlm_pipeline.py
# .parent → Scene_graph/
# .parent.parent → Aditya/
# .parent.parent.parent → RM/
# .parent.parent.parent.parent → Project_Root/
SCRIPT_DIR = Path(__file__).parent.parent.parent.parent

# [PATH CHANGE] Centralized input directory
INPUT_FOLDER = SCRIPT_DIR / "Input_Images"

# [PATH CHANGE] Team-specific output directory
TEAM_MEMBER = "Aditya"
MODULE_NAME = "Scene_graph"

# [PATH CHANGE] Output base directory: Project_Root/Output_Images/Aditya/Scene_graph/
OUTPUT_BASE = SCRIPT_DIR / "Output_Images" / TEAM_MEMBER / MODULE_NAME

# [PATH CHANGE] Create output base directory if missing
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

# [PATH CHANGE] Create a timestamp-based folder for this run
# This ensures each run's outputs are organized by timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FOLDER = OUTPUT_BASE / timestamp
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
print(f"[*] Output will be saved to: {OUTPUT_FOLDER}")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[*] Hardware Execution Device: {DEVICE.upper()}")

# 4-bit quantization config (keeps Qwen-VL under ~2.5GB VRAM)
gpu_quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

# =====================================================================
# 2. MODEL INITIALIZATION (SHARED 6GB VRAM BUDGET)
# =====================================================================
print("\n[*] [1/2] Loading Qwen2.5-VL-3B-Instruct (4-Bit NF4)...")
vlm_id = "Qwen/Qwen2.5-VL-3B-Instruct"
vlm_processor = AutoProcessor.from_pretrained(
    vlm_id,
    min_pixels=256 * 28 * 28,
    max_pixels=768 * 28 * 28
)
vlm_model = AutoModelForMultimodalLM.from_pretrained(
    vlm_id,
    quantization_config=gpu_quant_config,
    device_map=DEVICE
).eval()

print("[*] [2/2] Loading Florence-2-base-ft (FP16)...")
florence_id = "florence-community/Florence-2-base-ft"
florence_processor = AutoProcessor.from_pretrained(florence_id)
florence_model = Florence2ForConditionalGeneration.from_pretrained(
    florence_id,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    device_map=DEVICE
).eval()

print("[+] All vision engines loaded into VRAM successfully.\n")

# =====================================================================
# 3. VLM SCENE GRAPH EXTRACTION ENGINE
# =====================================================================
def extract_scene_graph_from_image(image_path: Path) -> dict:
    """
    Prompts Qwen2.5-VL to analyze the image and output structured JSON
    containing entities, atomic attributes, and semantic/spatial relationships.
    """
    clean_image_path = str(image_path.resolve())
    
    system_prompt = (
        "You are an expert computer vision and scene graph parsing AI. "
        "Analyze the provided image thoroughly. Extract all major physical objects, "
        "their visible attributes, and all spatial or semantic relationships between them.\n\n"
        "You MUST return ONLY a strict, valid JSON object with this exact schema:\n"
        "{\n"
        '  "entities": [\n'
        '    {"name": "lady", "attributes": ["sitting", "wearing jacket"]},\n'
        '    {"name": "bag", "attributes": ["crossbody", "brown"]},\n'
        '    {"name": "jacket", "attributes": ["warm", "dark"]},\n'
        '    {"name": "plate", "attributes": ["white", "round"]},\n'
        '    {"name": "car", "attributes": ["parked", "black"]},\n'
        '    {"name": "window", "attributes": ["glass", "transparent"]},\n'
        '    {"name": "building", "attributes": ["tall", "concrete"]}\n'
        "  ],\n"
        '  "relationships": [\n'
        '    {"subject": "lady", "predicate": "wearing", "object": "jacket"},\n'
        '    {"subject": "lady", "predicate": "holding", "object": "plate"},\n'
        '    {"subject": "lady", "predicate": "carrying", "object": "bag"},\n'
        '    {"subject": "car", "predicate": "behind", "object": "lady"},\n'
        '    {"subject": "car", "predicate": "has", "object": "window"},\n'
        '    {"subject": "car", "predicate": "in front of", "object": "building"}\n'
        "  ]\n"
        "}\n\n"
        "CRITICAL RULES:\n"
        "1. Identify short, concrete, physical object names (e.g., 'lady', 'car', 'bag', 'jacket', 'shoe', 'building').\n"
        "2. Predicates must describe exact physical interactions or spatial layouts (e.g., 'wearing', 'carrying', 'holding', 'sitting on', 'behind', 'in front of', 'has', 'next to').\n"
        "3. Output strictly raw JSON. Do not include markdown introductions, explanations, or trailing commentary."
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

    with torch.no_grad():
        outputs = vlm_model.generate(**inputs, max_new_tokens=1024, do_sample=False)

    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    raw_response = vlm_processor.decode(generated_ids, skip_special_tokens=True).strip()

    # Robust JSON extraction
    json_match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw_response, re.DOTALL)
    json_str = json_match.group(1).strip() if json_match else raw_response.strip()

    try:
        scene_graph = json.loads(json_str)
        if "entities" not in scene_graph:
            scene_graph["entities"] = []
        if "relationships" not in scene_graph:
            scene_graph["relationships"] = []
        return scene_graph
    except Exception as e:
        print(f"[-] JSON Parsing Warning: {e}. Raw response snippet: {raw_response[:200]}")
        # Fallback to direct regex extraction if format was slightly malformed
        fallback_entities = []
        fallback_relationships = []
        obj_match = re.findall(r'"name":\s*"([^"]+)"', raw_response)
        for obj in set(obj_match):
            fallback_entities.append({"name": obj, "attributes": []})
        return {"entities": fallback_entities, "relationships": fallback_relationships}

# =====================================================================
# 4. FLORENCE-2 GROUNDING ENGINE
# =====================================================================
def ground_entities_with_florence(image: Image.Image, entities: list) -> dict:
    """
    Takes the entity names extracted by the VLM and uses Florence-2
    Open Vocabulary Detection to locate accurate bounding box coordinates.
    """
    task_prompt = "<OPEN_VOCABULARY_DETECTION>"
    grounded_entities = {}

    for entity in entities:
        obj_name = entity.get("name", "").strip().lower()
        if not obj_name:
            continue

        inputs = florence_processor(
            text=task_prompt + obj_name,
            images=image,
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            generated_ids = florence_model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=512,
                do_sample=False,
                num_beams=3
            )

        generated_text = florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        results = florence_processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image.width, image.height)
        ).get(task_prompt, {})

        bboxes = results.get("bboxes", [])
        if bboxes:
            # Save the highest confidence/first detected box for this entity
            grounded_entities[obj_name] = [int(v) for v in bboxes[0]]

    return grounded_entities

# =====================================================================
# 5. VISUALIZATION & ANNOTATION
# =====================================================================
def draw_scene_graph_boxes(image: Image.Image, grounded_boxes: dict) -> Image.Image:
    """
    Draws red bounding boxes and bold labels matching the target diagram style.
    """
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    for label, bbox in grounded_boxes.items():
        x1, y1, x2, y2 = bbox
        
        # Draw clean red boundary
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        
        # Draw background banner for readable red text box
        text_label = label.upper()
        draw.rectangle([x1, max(0, y1 - 18), x1 + len(text_label) * 9 + 8, y1], fill="red")
        draw.text((x1 + 4, max(0, y1 - 16)), text_label, fill="white")

    return annotated

# =====================================================================
# 6. MAIN EXECUTION PIPELINE
# =====================================================================
def main():
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    images = [p for p in INPUT_FOLDER.iterdir() if p.is_file() and p.suffix.lower() in valid_exts]

    if not images:
        print(f"[-] No valid images found in '{INPUT_FOLDER}'.")
        print(f"[*] Please add test images into '{INPUT_FOLDER}' and rerun.")
        return

    print(f"[*] Found {len(images)} image(s) to process.\n")
    print("=" * 70)

    for idx, img_path in enumerate(images, start=1):
        print(f"[*] [{idx}/{len(images)}] Processing: {img_path.name}")
        image = Image.open(img_path).convert("RGB")

        # 1. Generate Scene Graph Triplets via VLM
        print("    -> Generating Scene Graph (Entities, Attributes, Relations)...")
        scene_graph_data = extract_scene_graph_from_image(img_path)

        entities = scene_graph_data.get("entities", [])
        relationships = scene_graph_data.get("relationships", [])

        print(f"    -> Extracted {len(entities)} Entities and {len(relationships)} Relationships.")

        # 2. Ground Entities using Florence-2
        print("    -> Grounding bounding boxes with Florence-2...")
        grounded_boxes = ground_entities_with_florence(image, entities)

        # 3. Attach bounding box coordinates directly to entity nodes in graph
        for entity in entities:
            name = entity.get("name", "").strip().lower()
            if name in grounded_boxes:
                entity["bbox"] = grounded_boxes[name]

        # 4. Save Final Scene Graph JSON
        json_output_path = OUTPUT_FOLDER / f"scene_graph_{img_path.stem}.json"
        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(scene_graph_data, f, indent=4)
        print(f"    [+] Scene Graph JSON saved to: '{json_output_path.name}'")

        # 5. Draw and Save Annotated Image
        annotated_image = draw_scene_graph_boxes(image, grounded_boxes)
        image_output_path = OUTPUT_FOLDER / f"annotated_{img_path.name}"
        annotated_image.save(image_output_path)
        print(f"    [+] Annotated Image saved to: '{image_output_path.name}'")

        # 6. Print Formatted Triplets to Console
        print("\n    --- [SCENE GRAPH PREVIEW] ---")
        for rel in relationships:
            subj = rel.get("subject", "")
            pred = rel.get("predicate", "")
            obj = rel.get("object", "")
            print(f"    ({subj}) ---[{pred}]---> ({obj})")
        print("=" * 70)

    print("\n[+] Scene graph processing complete for all images!")
    print(f"[+] Inspect outputs in: {OUTPUT_FOLDER}")

if __name__ == "__main__":
    main()