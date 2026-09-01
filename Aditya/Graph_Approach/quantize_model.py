import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ==========================================
# 1. CONFIGURATION & CONSTANTS
# ==========================================

# [PATH CHANGE] Resolve Project_Root (4 levels up from RM/Aditya/Graph_Approach/)
SCRIPT_DIR = Path(__file__).parent.parent.parent.parent

# [PATH CHANGE] Model cache directory
os.environ["HF_HOME"] = str(SCRIPT_DIR / "HuggingFaceModels")

SLM_ID = "Qwen/Qwen2.5-7B-Instruct"

# [PATH CHANGE] Save quantized model to Project_Root level (shared across members)
SAVE_DIR = SCRIPT_DIR / "Models" / "Qwen-7B-8bit"

def main():
    print("=" * 60)
    print("[*] QUANTIZING Qwen 7B TO 8-BIT")
    print("=" * 60)
    
    print(f"[*] Model ID: {SLM_ID}")
    print(f"[*] Save directory: {SAVE_DIR}")
    print(f"[*] HF_HOME: {os.environ['HF_HOME']}")
    print("=" * 60)

    # [PATH CHANGE] Create save directory if it doesn't exist
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/4] Loading and quantizing to 8-bit (This will take a while)...")
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    # Note: We keep device_map="cpu" since this is a system RAM operation
    print("[*] Loading model... (This may take several minutes)")
    model = AutoModelForCausalLM.from_pretrained(
        SLM_ID,
        quantization_config=bnb_config,
        device_map="cpu"
    )

    print("[*] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(SLM_ID)

    print(f"\n[2/4] Saving 8-bit model to {SAVE_DIR}...")
    # Save the quantized model locally with sharding
    model.save_pretrained(SAVE_DIR, max_shard_size="1GB")
    tokenizer.save_pretrained(SAVE_DIR)

    print("\n[3/4] Verifying saved model...")
    # Verify the model was saved correctly
    if SAVE_DIR.exists():
        model_files = list(SAVE_DIR.glob("*.bin")) + list(SAVE_DIR.glob("*.safetensors"))
        print(f"    -> Found {len(model_files)} model shard files")
        print(f"    -> Tokenizer files: {len(list(SAVE_DIR.glob('*.json')))}")
    
    print("\n[4/4] Done!")
    print("=" * 60)
    print(f"[+] Quantization complete!")
    print(f"[+] Model saved to: {SAVE_DIR}")
    print(f"[+] You can now use the pre-quantized model by setting:")
    print(f"    SLM_ID = '{SAVE_DIR}'")
    print("=" * 60)

if __name__ == "__main__":
    main()