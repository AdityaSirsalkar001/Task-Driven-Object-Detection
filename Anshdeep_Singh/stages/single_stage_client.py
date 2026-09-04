import re
import requests
from PIL import Image, ImageDraw

def extract_box_data(text, img_w, img_h):
    match = re.search(r"(.*?)\s*\[([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\]", text)
    if match:
        lbl = match.group(1).strip()
        x1, y1, x2, y2 = map(float, match.groups()[1:])
        scale = 1000.0 if max(x1, y1, x2, y2) > 1.0 else 1.0
        return lbl, int((x1/scale)*img_w), int((y1/scale)*img_h), int((x2/scale)*img_w), int((y2/scale)*img_h)
    return None

def main():
    SERVER_URL = "http://127.0.0.1:8000/generate"
    
    while True:
        print("\n" + "="*40)
        image_path = input("Enter image path (or type 'q' to quit): ").strip()
        if image_path.lower() == 'q':
            break
            
        prompt = input("Enter your prompt: ").strip()
        print("Waiting for GPU model...")
        
        # Send data to the server
        try:
            response = requests.post(SERVER_URL, json={
                "image_path": image_path,
                "prompt": prompt
            })
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to the server: {e}")
            print("Make sure server.py is running in another terminal!")
            continue
            
        raw_response = data["raw_response"]
        width = data["width"]
        height = data["height"]
        
        print(f"\nModel Output:\n{raw_response}")
        
        # Parse and Draw
        detection = extract_box_data(raw_response, width, height)
        if detection:
            label, xmin, ymin, xmax, ymax = detection
            print(f"Drawing box for '{label}' at [{xmin}, {ymin}, {xmax}, {ymax}]")
            
            img = Image.open(image_path)
            draw = ImageDraw.Draw(img)
            # You can tweak line thickness, colors, or fonts right here
            draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=4)
            img.show()
        else:
            print("No valid bounding box found.")

if __name__ == "__main__":
    main()
