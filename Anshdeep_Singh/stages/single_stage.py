import os
from pathlib import Path

# Enforce the custom cache path and completely disable network calls
MODEL_BASE_DIR = "/media/anshdeep-singh/Aditya/HuggingFaceModels"
os.environ["HF_HOME"] = MODEL_BASE_DIR
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import re
import torch
from pathlib import Path
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForMultimodalLM
from qwen_vl_utils import process_vision_info
import re

def extract_box_data(text, img_w, img_h):
    match = re.search(r"(.*?)\s*\[([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\]", text)
    if match:
        lbl = match.group(1).strip()
        x1, y1, x2, y2 = map(float, match.groups()[1:])
        scale = 1000.0 if max(x1, y1, x2, y2) > 1.0 else 1.0
        return lbl, int((x1/scale)*img_w), int((y1/scale)*img_h), int((x2/scale)*img_w), int((y2/scale)*img_h)
    return None

def analyze_image(image_path, prompt, model, processor):
    img = Image.open(image_path)
    width, height = img.size
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": str(Path(image_path).resolve())},
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
    detection = extract_box_data(raw_response, width, height)
    if detection:
        label, xmin, ymin, xmax, ymax = detection
        draw = ImageDraw.Draw(img)
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=4)
        img.show()
    else:
        print("No valid bounding box found.")
        print("Raw model response:", raw_response)

if __name__ == "__main__":
    model_target = "Qwen/Qwen2.5-VL-3B-Instruct"

    processor = AutoProcessor.from_pretrained(
        model_target,
        local_files_only=True
    )
    
    # Define the 4-bit quantization configuration
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",               # Recommended for better precision
        bnb_4bit_use_double_quant=True,          # Saves even more memory
        bnb_4bit_compute_dtype=torch.bfloat16    # Computations stay in bfloat16
    )

    # Load model with quantization config applied
    model = AutoModelForMultimodalLM.from_pretrained(
        model_target,
        device_map="auto",
        quantization_config=quantization_config,
        local_files_only=True
    )

    image_path = input("Enter image path: ")
    prompt = input("Enter your prompt: ")
    analyze_image(image_path, prompt, model, processor)
