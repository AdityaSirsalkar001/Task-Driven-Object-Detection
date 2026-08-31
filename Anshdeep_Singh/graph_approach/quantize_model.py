import os
# Ensure Hugging Face uses your external drive
os.environ["HF_HOME"] = "/media/anshdeep-singh/Aditya/HuggingFaceModels"

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SLM_ID = "Qwen/Qwen2.5-7B-Instruct"
# Define the path where you want to save the 8-bit model
SAVE_DIR = "../Qwen-7B-8bit/"

print("1. Loading and quantizing to 8-bit (This will take a while)...")
bnb_config = BitsAndBytesConfig(load_in_8bit=True)

# Note: We keep device_map="cpu" since you are operating on system RAM
model = AutoModelForCausalLM.from_pretrained(
    SLM_ID,
    quantization_config=bnb_config,
    device_map="cpu"
)

tokenizer = AutoTokenizer.from_pretrained(SLM_ID)

print(f"2. Saving 8-bit model to {SAVE_DIR}...")
# Save the quantized model locally
model.save_pretrained(SAVE_DIR, max_shard_size="1GB")
tokenizer.save_pretrained(SAVE_DIR)

print("Done! You can now use the pre-quantized model.")
