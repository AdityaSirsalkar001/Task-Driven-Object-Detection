import torch
from pathlib import Path
from transformers import AutoProcessor, AutoModelForMultimodalLM
from qwen_vl_utils import process_vision_info

# 1. Load the Processor and the Model
model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForMultimodalLM.from_pretrained(
    model_id, 
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# --- THE FIX IS HERE ---
# Use pathlib to get the clean, absolute path WITHOUT "file://"
script_dir = Path(__file__).parent
image_file = script_dir / "Open_Parcel" / "3.jpg"

# Convert to a standard string (e.g., "/home/anshdeep-singh/.../3.jpg")
clean_image_path = str(image_file.resolve())

# 2. Format the Prompt
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": clean_image_path},
            {"type": "text", "text": "What can I use to open a parcel?"}
        ]
    }
]

# 3. Process the Inputs using Qwen's official utility
# This safely extracts the image from the local path and formats it as a tensor
text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
    text=[text_prompt],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt"
).to(model.device)

# 4. Generate the Output
outputs = model.generate(**inputs, max_new_tokens=100)

# 5. Decode and Print the Result
generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
response = processor.decode(generated_ids, skip_special_tokens=True)

print(response)
