from rembg import remove
from PIL import Image

input = Image.open("pizza.avif")

output = remove(input)

output.save("output.png")
