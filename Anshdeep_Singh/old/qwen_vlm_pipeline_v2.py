import torch
import csv
import re
from pathlib import Path
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForMultimodalLM
from qwen_vl_utils import process_vision_info

# ==========================================
# 1. GLOBAL CONFIGURATION
# ==========================================
SCRIPT_DIR = Path(__file__).parent
INPUT_FOLDER = SCRIPT_DIR / "Open_Parcel"

OUTPUT_FOLDER = SCRIPT_DIR / "Annotated_Images"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

CSV_FILEPATH = SCRIPT_DIR / "qwen_bounding_boxes_dataset.csv"
PROMPT_TEXT = "What object in this image can be used to open a parcel(eg. knife, scissors, cutter, switchblade, sharp key etc)? Give a short label for the object and return its bounding box."

# ==========================================
# 2. HELPER FUNCTION: PARSE BOUNDING BOXES
# ==========================================
def extract_box_data(text, img_w, img_h):
    """
    Checks multiple possible Qwen output formats and returns absolute pixel coordinates.
    Returns: (label, abs_xmin, abs_ymin, abs_xmax, abs_ymax) or None
    """
    # Format 1: Bracket Array e.g. "[0.36, 0.81, 0.54, 0.9]" OR "[360, 810, 540, 900]"
    bracket_match = re.search(r"(.*?)\s*\[([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\]", text)
    if bracket_match:
        lbl = bracket_match.group(1).replace("<|im_start|>", "").replace("<|im_end|>", "").strip()
        x1, y1, x2, y2 = map(float, bracket_match.groups()[1:])
        
        # Determine if the array is normalized (0.0-1.0) or on a 1000-point grid
        scale = 1000.0 if max(x1, y1, x2, y2) > 1.0 else 1.0
        
        return (
            lbl, 
            int((x1 / scale) * img_w), 
            int((y1 / scale) * img_h), 
            int((x2 / scale) * img_w), 
            int((y2 / scale) * img_h)
        )

    # Format 2: Legacy Tags e.g. "<|box_start|>(ymin,xmin),(ymax,xmax)<|box_end|>"
    tag_match = re.search(r"(.*?)<\|box_start\|>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|>", text)
    if tag_match:
        lbl = tag_match.group(1).replace("<|im_start|>", "").replace("<|im_end|>", "").strip()
        ymin, xmin, ymax, xmax = map(float, tag_match.groups()[1:])
        
        # Legacy tags are strictly on the 1000-point grid and mapped (ymin, xmin, ymax, xmax)
        return (
            lbl, 
            int((xmin / 1000.0) * img_w), 
            int((ymin / 1000.0) * img_h), 
            int((xmax / 1000.0) * img_w), 
            int((ymax / 1000.0) * img_h)
        )
        
    return None

# ==========================================
# 3. CORE PROCESSING FUNCTION
# ==========================================
def analyze_image(image_name, model, processor):
    """
    Processes a single image, draws the bounding box, and logs it to a CSV.
    """
    input_image_path = INPUT_FOLDER / image_name
    
    # Safety check: Ensure the image actually exists before trying to run AI on it
    if not input_image_path.exists():
        print(f"\n[-] Error: Could not find '{image_name}' in {INPUT_FOLDER}")
        return

    print(f"\n==========================================")
    print(f"Processing: {image_name}")
    print(f"==========================================")
    
    clean_image_path = str(input_image_path.resolve())

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": clean_image_path},
                {"type": "text", "text": PROMPT_TEXT}
            ]
        }
    ]

    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=150, do_sample=False)
    
    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]

    raw_response = processor.decode(generated_ids, skip_special_tokens=False)
    clean_response = processor.decode(generated_ids, skip_special_tokens=True).strip()

    img = Image.open(input_image_path)
    width, height = img.size

    detection = extract_box_data(raw_response, width, height)

    if detection:
        label, abs_xmin, abs_ymin, abs_xmax, abs_ymax = detection
        if not label:
            label = "Detected Object"
            
        print(f"[+] Detected Object Label : '{label}'")
        print(f"[+] Pixel Coordinates     : [xmin={abs_xmin}, ymin={abs_ymin}, xmax={abs_xmax}, ymax={abs_ymax}]")
        
        draw = ImageDraw.Draw(img)
        draw.rectangle([abs_xmin, abs_ymin, abs_xmax, abs_ymax], outline="red", width=4)
        
        print("[+] Opening annotated image window...")
        # img.show()
        
        annotated_image_name = f"boxed_{image_name}"
        annotated_image_path = OUTPUT_FOLDER / annotated_image_name
        img.save(annotated_image_path)
        print(f"[+] Saved annotated image to: {annotated_image_path}")
        
        file_exists = CSV_FILEPATH.exists()
        with open(CSV_FILEPATH, mode='a', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            if not file_exists:
                writer.writerow(["Image_Filename", "Original_Prompt", "Detected_Label", "Bounding_Box_[xmin,ymin,xmax,ymax]"])
            
            writer.writerow([
                image_name, 
                PROMPT_TEXT, 
                label,
                f"[{abs_xmin}, {abs_ymin}, {abs_xmax}, {abs_ymax}]"
            ])
        print(f"[+] Appended dataset entry to: {CSV_FILEPATH}")

    else:
        print("[-] No valid bounding box found in the model's response.")
        print(f"Model output: {clean_response}")


# ==========================================
# 4. MAIN EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    
    print("[*] Booting up Qwen2.5-VL-3B-Instruct engine...")
    
    # 1. Load the model ONCE into memory
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForMultimodalLM.from_pretrained(
        model_id, 
        device_map="auto",
        dtype=torch.bfloat16
    )
    
    print("[*] Engine loaded successfully!\n")

    for i in range(1, 6):
        # name= i + ".jpg"
        analyze_image(f"{i}.jpg", model, processor)
        
    # 2. Call the function by just passing the filename 
    
    # Example: How you can now loop through your newly renamed files automatically
    # for i in range(1, 5):
    #     analyze_image(f"{i}.jpg", model, processor)
