import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Global variables for Lazy Loading
_tokenizer = None
_model = None

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

def get_local_model():
    """Loads Qwen SLM into CPU/RAM only when needed."""
    global _tokenizer, _model
    if _model is None or _tokenizer is None:
        print(f"[*] Loading local SLM ({MODEL_ID}) into RAM for offline generation...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,  # Consumes only ~6.5 GB RAM
            device_map="cpu"
        ).eval()
    return _tokenizer, _model

def query_slm_for_tools(user_task):
    """Generates a list of 10 physical tools locally without internet or API keys."""
    tokenizer, model = get_local_model()

    # Strict system instructions enforcing 10 items
    system_prompt = (
        "You are an expert physical tool brainstormer. "
        "When given a task, you MUST brainstorm and list EXACTLY 10 distinct, physical, inanimate everyday tools or objects "
        "that could realistically be used (directly or as improvised alternatives) to perform or assist with that task.\n"
        "CRITICAL RULES:\n"
        "1. Do NOT suggest human body parts (no hands, fingers, feet, teeth).\n"
        "2. Do NOT suggest living beings (no humans, animals).\n"
        "3. Provide exactly 10 items in the list.\n"
        "4. Return STRICTLY valid JSON with no conversational text or explanation."
    )

    user_prompt = f"""Task: "{user_task}"

Output your answer in this exact JSON format:
```json
{{
    "tools": [
        "tool 1",
        "tool 2",
        "tool 3",
        "tool 4",
        "tool 5",
        "tool 6",
        "tool 7",
        "tool 8",
        "tool 9",
        "tool 10"
    ]
}}
```"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to("cpu")

    print(f"[*] Generating 10 tools locally via Qwen SLM for: '{user_task}'...")
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=300,   # Increased to safely accommodate 10 full items
            do_sample=False,
            temperature=0.1
        )

    generated_ids = [
        output_ids[len(input_ids):] 
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    raw_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    # Parse the JSON response
    try:
        json_match = re.search(r'```json\s*(.*?)\s*```', raw_text, re.DOTALL)
        if not json_match:
            json_match = re.search(r'(\{.*?\})', raw_text, re.DOTALL)
            
        if json_match:
            parsed = json.loads(json_match.group(1).strip())
            tools = parsed.get("tools", [])
            # Return cleaned list of tools
            return [t.strip().lower() for t in tools if t.strip()]
    except Exception as e:
        print(f"[-] Local Parsing Error: {e}")

    return []