import os
from pathlib import Path

# Enforce the custom cache path and completely disable network calls
MODEL_BASE_DIR = "/media/anshdeep-singh/Aditya/HuggingFaceModels"
os.environ["HF_HOME"] = MODEL_BASE_DIR
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
from transformers import AutoProcessor, AutoModelForMultimodalLM, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
import uvicorn

print("Loading model onto GPU... This will take a moment.")

# Using the exact target that worked in your script
model_target = "Qwen/Qwen2.5-VL-3B-Instruct"

processor = AutoProcessor.from_pretrained(
    model_target,
    local_files_only=True
)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForMultimodalLM.from_pretrained(
    model_target,
    device_map="auto",
    quantization_config=quantization_config,
    local_files_only=True
)

print("Model loaded successfully! Starting server...")

app = FastAPI()

class GenerateRequest(BaseModel):
    image_path: str
    prompt: str

@app.post("/generate")
def generate_text(req: GenerateRequest):
    try:
        img_path_resolved = str(Path(req.image_path).resolve())
        img = Image.open(img_path_resolved)
        width, height = img.size
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img_path_resolved},
                {"type": "text", "text": req.prompt}
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
        
        return {
            "raw_response": raw_response,
            "width": width,
            "height": height
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
