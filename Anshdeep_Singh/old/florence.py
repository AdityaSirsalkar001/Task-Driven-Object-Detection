import os
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

import torch
from transformers import AutoProcessor, Florence2ForConditionalGeneration
from PIL import Image, ImageDraw

# Load Florence-2 model
model_id = "florence-community/Florence-2-base-ft"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading Florence-2 on {device}...")
model = Florence2ForConditionalGeneration.from_pretrained(model_id).to(device).eval()
processor = AutoProcessor.from_pretrained(model_id) 

def run_florence(task_prompt, text_input, image):
    # Combine the task token with the user query (e.g., "<CAPTION_TO_PHRASE>Find something sharp...")
    prompt = task_prompt + text_input 
    
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        do_sample=False,
        num_beams=3
    )
    
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    
    # Florence's post-processor requires the bare task token (task_prompt) 
    # to parse the output dictionary correctly, NOT the full prompt string.
    return processor.post_process_generation(
        generated_text, 
        task=task_prompt, 
        image_size=(image.width, image.height)
    )
    
def draw_bounding_boxes(image, detection_results):
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)
    
    bboxes = detection_results.get('bboxes', [])
    
    # --- FIX: Check for 'bboxes_labels' first, fallback to 'labels' ---
    labels = detection_results.get('bboxes_labels', detection_results.get('labels', []))

    # Helpful for debugging to see exactly what is being paired up
    print(f"Found {len(bboxes)} boxes and {len(labels)} labels")

    for bbox, label in zip(bboxes, labels):
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        text_position = (x1, max(0, y1 - 15))
        draw.text(text_position, str(label), fill="red")

    return draw_image
    
def select_image_from_parent_folder():
    """Opens a GUI file dialog starting at the parent directory."""
    root = tk.Tk()
    root.withdraw()  # Hide the extra Tkinter root window

    # Get the parent directory of the current working directory
    parent_dir = Path.cwd()

    # Open file picker starting in the parent folder
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        initialdir=parent_dir,
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.webp *.bmp")]
    )
    
    # Clean up the hidden Tkinter window so it doesn't freeze in a loop
    root.destroy()
    
    return file_path

if __name__ == "__main__":
    print("Starting continuous detection mode...")
    
    while True:
        # Open file picker
        image_path = select_image_from_parent_folder()

        # Break the loop if the user cancels or closes the window
        if not image_path:
            print("No image selected. Exiting...")
            break
            
        print(f"\nSelected image: {image_path}")
        
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image: {e}")
            continue
            
        # task = "<OPEN_VOCABULARY_DETECTION>"
        task = "<CAPTION_TO_PHRASE_GROUNDING>"
        
        user_query = input("\nEnter query (e.g., 'knife, fork, spoon'): ")
        
        # Split the input by commas and clean whitespace
        # queries = [q.strip() for q in user_query.split(',') if q.strip()]
        
        # print(f"Running detection for {len(queries)} items...")
        
        all_bboxes = []
        all_labels = []

        results = run_florence(task, user_query, image)
        od_results = results.get(task, {})

        bboxes = od_results.get('bboxes', [])
        labels = od_results.get('bboxes_labels', od_results.get('labels', []))

        print(bboxes)
        print("\n")
        print(labels)

        # exit()
        # Loop through each item individually
        # for q in queries:
        #     print(f"  -> Searching for: '{q}'")
        #     results = run_florence(task, q, image)
        #     od_results = results.get(task, {})
        #     
        #     # Skip if Florence failed to find this specific object
        #     if isinstance(od_results, str):
        #         continue
        #         
        #     # Extract and merge the boxes and labels
        #     bboxes = od_results.get('bboxes', [])
        #     labels = od_results.get('bboxes_labels', od_results.get('labels', []))
        #     
        #     all_bboxes.extend(bboxes)
        #     all_labels.extend(labels)
        #     
        # Check if anything was found across all queries
        if not bboxes:
             print("\n[!] The model returned no bounding boxes for any of the requested items.")
             continue
             
        # Package the combined results into the dictionary format your drawing function expects
        merged_results = {
            'bboxes': bboxes,
            'bboxes_labels': labels
        }
        
        annotated_image = draw_bounding_boxes(image, merged_results)
        
        # Save output in the current working directory
        output_filename = "output_with_boxes.jpg"
        annotated_image.save(output_filename)
        annotated_image.show()
        print(f"Saved output to '{output_filename}'")
