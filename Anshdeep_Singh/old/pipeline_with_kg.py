import torch
from pathlib import Path
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Florence2ForConditionalGeneration

import kg_manager
import llm_fallback

# --- DIRECTORY PATHS ---
SCRIPT_DIR = Path(__file__).parent
INPUT_FOLDER = SCRIPT_DIR / "Open_Parcel"
OUTPUT_FOLDER = SCRIPT_DIR / "Annotated_KG"

INPUT_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

def main():
    # 1. Load Knowledge Graph
    graph = kg_manager.load_graph()
    
    print("\n" + "="*50)
    user_query = input("Enter the task you want to perform (e.g., 'open the parcel'): ").strip()
    print("="*50 + "\n")

    if not user_query:
        print("[-] No input provided. Exiting.")
        return

    # 2. Check Knowledge Graph (Semantic Search)
    print("[*] Searching Knowledge Graph...")
    matched_task, similarity = kg_manager.semantic_search(user_query, graph)

    if matched_task:
        print(f"[+] GRAPH HIT: Matched '{user_query}' -> '{matched_task}' (Score: {similarity:.2f})")
        approved_tools = graph[matched_task]
    else:
        print(f"[-] GRAPH MISS: Task not found in graph (Highest similarity: {similarity:.2f})")
        # Around line 35
        print("[*] Invoking local Qwen SLM to generate tools offline...")
        approved_tools = llm_fallback.query_slm_for_tools(user_query)
        
        if approved_tools:
            # Save new dynamic discovery to knowledge graph permanently
            graph[user_query] = approved_tools
            kg_manager.save_graph(graph)
            print(f"[+] Knowledge Graph Updated! Saved '{user_query}' with tools: {approved_tools}")
        else:
            print("[-] SLM could not generate tools for this task.")
            return

    print(f"\n[+] Active Filter Tools: {approved_tools}")

    # 3. Load Florence-2 Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[*] Booting Florence-2 on {device.upper()}...")
    florence_id = "florence-community/Florence-2-base-ft"
    
    florence_model = Florence2ForConditionalGeneration.from_pretrained(
        florence_id, 
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
    ).to(device).eval()
    
    florence_processor = AutoProcessor.from_pretrained(florence_id)

    # 4. Scan Input Directory
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    images = [p for p in INPUT_FOLDER.iterdir() if p.is_file() and p.suffix.lower() in valid_exts]

    if not images:
        print(f"[-] No image files found in '{INPUT_FOLDER}'. Add test images and run again.")
        return

    for img_path in images:
        print(f"\n[*] Processing: {img_path.name}")
        image = Image.open(img_path).convert("RGB")
        width, height = image.size

        # Florence-2 Dense Region Captioning
        task_prompt = "<DENSE_REGION_CAPTION>"
        inputs = florence_processor(text=task_prompt, images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            generated_ids = florence_model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                do_sample=False,
                num_beams=3
            )
            
        generated_text = florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        results = florence_processor.post_process_generation(
            generated_text, 
            task=task_prompt, 
            image_size=(width, height)
        ).get(task_prompt, {})

        bboxes = results.get('bboxes', [])
        labels = results.get('labels', [])

        # Filter detected bounding boxes against active tools list
        draw = ImageDraw.Draw(image)
        matched_any = False

        for bbox, label in zip(bboxes, labels):
            if any(tool.lower() in label.lower() for tool in approved_tools):
                x1, y1, x2, y2 = map(int, bbox)
                draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
                draw.text((x1, max(0, y1 - 15)), label.upper(), fill="red")
                matched_any = True
                print(f"    -> Detected Object Match: {label.upper()}")

        if matched_any:
            output_file = OUTPUT_FOLDER / f"kg_output_{img_path.name}"
            image.save(output_file)
            print(f"[+] Output saved to: '{output_file}'")
        else:
            print("[-] No matching tools from the Knowledge Graph were located in this scene.")

if __name__ == "__main__":
    main()