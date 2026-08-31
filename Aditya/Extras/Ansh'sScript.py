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

def run_florence(task_prompt, image):
    inputs = processor(text=task_prompt, images=image, return_tensors="pt").to(device)
    
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        do_sample=False,
        num_beams=3
    )
    
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    return processor.post_process_generation(
        generated_text, 
        task=task_prompt, 
        image_size=(image.width, image.height)
    )

def draw_bounding_boxes(image, detection_results):
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)
    
    bboxes = detection_results.get('bboxes', [])
    labels = detection_results.get('labels', [])

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
            
        task = "<OD>"  # Object Detection
        
        print("Running detection...")
        results = run_florence(task, image)
        
        od_results = results.get(task, {})
        
        annotated_image = draw_bounding_boxes(image, od_results)
        
        # Save output in the current working directory
        output_filename = "output_with_boxes.jpg"
        annotated_image.save(output_filename)
        annotated_image.show()
        print(f"Saved output to '{output_filename}'")