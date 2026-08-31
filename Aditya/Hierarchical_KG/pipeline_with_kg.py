import torch
from pathlib import Path
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Florence2ForConditionalGeneration

import kg_manager
import llm_fallback

# --- PATH CHANGES: Standardized paths for Project_Root ---
# [PATH CHANGE] Resolve Project_Root (4 levels up from current file location)
# Current: Project_Root/RM/Aditya/Hierarchical_KG/pipeline_with_kg.py
# .parent → Hierarchical_KG/
# .parent.parent → Aditya/
# .parent.parent.parent → RM/
# .parent.parent.parent.parent → Project_Root/
SCRIPT_DIR = Path(__file__).parent.parent.parent.parent

# [PATH CHANGE] Centralized input directory
INPUT_FOLDER = SCRIPT_DIR / "Input_Images"

# [PATH CHANGE] Team member name (Aditya's code)
TEAM_MEMBER = "Aditya"

# [PATH CHANGE] Module name (the folder containing this code)
MODULE_NAME = "Hierarchical_KG"

# [PATH CHANGE] Output base directory: Project_Root/Output_Images/Aditya/Hierarchical_KG/
OUTPUT_BASE = SCRIPT_DIR / "Output_Images" / TEAM_MEMBER / MODULE_NAME

# [PATH CHANGE] Create output base directory if missing
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

def main():
    # 1. Load Knowledge Graph
    graph = kg_manager.load_graph()
    
    print("\n" + "="*50)
    user_query = input("Enter the task you want to perform (e.g., 'open the parcel'): ").strip()
    print("="*50 + "\n")

    if not user_query:
        print("[-] No input provided. Exiting.")
        return

    # [PATH CHANGE] Create task-specific output folder
    # Sanitize the task name to be filesystem-safe
    task_folder_name = "".join(c for c in user_query if c.isalnum() or c in (' ', '-', '_')).strip().replace(' ', '_')
    OUTPUT_FOLDER = OUTPUT_BASE / task_folder_name
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    print(f"[*] Output will be saved to: {OUTPUT_FOLDER}")

    # 2. Check Knowledge Graph via Semantic Similarity
    print("[*] Searching Hierarchical Knowledge Graph...")
    matched_task, similarity = kg_manager.semantic_match_task(user_query, graph)

    if matched_task:
        print(f"[+] GRAPH HIT: '{user_query}' -> '{matched_task}' (Score: {similarity:.2f})")
        reasoning = kg_manager.traverse_graph_for_entities(matched_task, graph)
        print(f"    ├─ Required Actions    : {reasoning['actions']}")
        print(f"    ├─ Required Attributes : {reasoning['attributes']}")
        print(f"    └─ Candidate Tools     : {reasoning['entities']}")
        approved_tools = reasoning["entities"]
    else:
        print(f"[-] GRAPH MISS (Score: {similarity:.2f}). Generating ontological triples via SLM...")
        subgraph = llm_fallback.query_slm_for_hierarchical_subgraph(user_query)

        if subgraph:
            task_name = subgraph["task"]
            actions_dict = subgraph.get("actions", {})
            attributes_dict = subgraph.get("attributes", {})

            # Ingest new triples into Master Graph
            graph.setdefault("tasks", {})[task_name] = {"REQUIRES_ACTION": list(actions_dict.keys())}
            for act_name, attrs in actions_dict.items():
                graph.setdefault("actions", {})[act_name] = {"REQUIRES_ATTRIBUTE": attrs}
            for attr_name, tools in attributes_dict.items():
                graph.setdefault("attributes", {})[attr_name] = {"POSSESSED_BY": tools}

            kg_manager.save_graph(graph)
            print(f"[+] Knowledge Graph Expanded with structured nodes for '{task_name}'.")

            reasoning = kg_manager.traverse_graph_for_entities(task_name, graph)
            approved_tools = reasoning["entities"]
        else:
            print("[-] SLM could not decompose the task.")
            return

    print(f"\n[+] Final Active Filter Tools: {approved_tools}")

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
            output_file = OUTPUT_FOLDER / f"hierarchical_kg_{img_path.name}"
            image.save(output_file)
            print(f"[+] Output saved to: '{output_file}'")
        else:
            print("[-] No matching tools from the Knowledge Graph were located in this scene.")

if __name__ == "__main__":
    main()