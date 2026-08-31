import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

_tokenizer = None
_model = None
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"


def get_local_model():
    global _tokenizer, _model
    if _model is None or _tokenizer is None:
        print(f"[*] Loading Local SLM ({MODEL_ID}) into RAM...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="cpu"
        ).eval()
    return _tokenizer, _model


def query_slm_for_hierarchical_subgraph(user_task):
    """
    Prompts the SLM to break a new task down into Task -> Actions -> Attributes -> Entities.
    """
    tokenizer, model = get_local_model()

    system_prompt = (
        "You are an ontological graph parser. Break down a physical task into graph triples.\n"
        "Schema:\n"
        "1. Task REQUIRES_ACTION (1-3 physical sub-actions)\n"
        "2. Action REQUIRES_ATTRIBUTE (1-2 physical properties per action)\n"
        "3. Attribute POSSESSED_BY (3-6 physical objects/tools per attribute)\n"
        "Return ONLY a JSON object matching this exact structure."
    )

    user_prompt = f"""Deconstruct task: "{user_task}"

Output strictly in this JSON format:
```json
{{
  "task": "{user_task}",
  "actions": {{
    "sub_action_1": ["attribute_1", "attribute_2"]
  }},
  "attributes": {{
    "attribute_1": ["tool_1", "tool_2", "tool_3"],
    "attribute_2": ["tool_4", "tool_5"]
  }}
}}
```"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to("cpu")

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=400, do_sample=False, temperature=0.1)

    generated_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    raw_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    try:
        json_match = re.search(r"```json\s*(.*?)\s*```", raw_text, re.DOTALL)
        if not json_match:
            json_match = re.search(r"(\{.*\})", raw_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1).strip())
    except Exception as e:
        print(f"[-] JSON Extraction Failed: {e}")

    return None