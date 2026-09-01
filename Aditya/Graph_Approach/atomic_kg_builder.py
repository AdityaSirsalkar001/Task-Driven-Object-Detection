import os
import json
import re
from pathlib import Path
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==========================================
# 1. CONFIGURATION & CONSTANTS
# ==========================================

# [PATH CHANGE] Resolve Project_Root (4 levels up from RM/Aditya/Graph_Approach/)
SCRIPT_DIR = Path(__file__).parent.parent.parent.parent

# [PATH CHANGE] Keep KG file inside module folder (not centralized)
KG_FILE = Path(__file__).parent / "vector_atomic_kg.json"
SIMILARITY_THRESHOLD = 0.75  # 95% threshold for synonym merging

CPU_DEVICE = "cpu"
SLM_ID = "Qwen/Qwen2.5-3B-Instruct"

# [PATH CHANGE] Cache directory for HuggingFace models
os.environ["HF_HOME"] = str(SCRIPT_DIR / "HuggingFaceModels")

# Global lazy loaders
_embedder = None
_slm_tokenizer = None
_slm_model = None

def get_embedder():
    """Loads the sentence embedding model."""
    global _embedder
    if _embedder is None:
        print("[*] Loading Sentence Embedding Model (all-MiniLM-L6-v2)...")
        _embedder = SentenceTransformer("all-MiniLM-L6-v2", device=CPU_DEVICE)
    return _embedder

def get_slm():
    """Loads the SLM onto CPU/RAM only when extraction is requested."""
    global _slm_tokenizer, _slm_model
    if _slm_model is None or _slm_tokenizer is None:
        print(f"[*] Loading SLM ({SLM_ID}) on CPU/RAM...")
        _slm_tokenizer = AutoTokenizer.from_pretrained(SLM_ID)
        _slm_model = AutoModelForCausalLM.from_pretrained(
            SLM_ID,
            torch_dtype=torch.bfloat16,
            device_map=CPU_DEVICE
        ).eval()
    return _slm_tokenizer, _slm_model


# ==========================================
# 2. VECTOR KNOWLEDGE GRAPH CLASS
# ==========================================
class VectorAtomicKnowledgeGraph:
    def __init__(self, filepath=KG_FILE, threshold=SIMILARITY_THRESHOLD):
        self.filepath = filepath
        self.threshold = threshold
        self.objects = {}       # obj_name -> list of attr_node_ids
        self.attributes = {}    # attr_node_id -> {canonical_name, aliases, embedding, objects}
        self.embedder = get_embedder()
        self.load()

    def load(self):
        """Loads the graph and embeddings from disk if present."""
        if os.path.exists(self.filepath):
            with open(self.filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.objects = data.get("objects", {})
                self.attributes = data.get("attributes", {})
        else:
            self.objects = {}
            self.attributes = {}

    def save(self):
        """Persists graph structure and high-dimensional vectors to JSON."""
        data = {
            "metadata": {
                "similarity_threshold": self.threshold,
                "total_objects": len(self.objects),
                "total_unique_attribute_nodes": len(self.attributes)
            },
            "objects": self.objects,
            "attributes": self.attributes
        }
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _find_or_create_attribute_node(self, raw_attr_name: str) -> str:
        """
        Calculates cosine similarity against all existing attribute embeddings.
        Returns existing node ID if similarity >= threshold, otherwise creates a new node.
        """
        clean_text = raw_attr_name.strip().lower().replace("_", " ")
        if not clean_text:
            return None

        # 1. Encode query attribute & L2 normalize
        query_vec = self.embedder.encode(clean_text, convert_to_numpy=True)
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm

        # 2. Compare against existing attribute nodes
        if self.attributes:
            node_ids = list(self.attributes.keys())
            # Stack existing normalized vectors
            matrix = np.array([self.attributes[nid]["embedding"] for nid in node_ids])
            
            # Matrix dot-product yields cosine similarity directly
            similarities = np.dot(matrix, query_vec)
            best_idx = int(np.argmax(similarities))
            best_score = float(similarities[best_idx])
            best_match_name = self.attributes[node_ids[best_idx]]["canonical_name"]

            if best_score >= self.threshold:
                matched_id = node_ids[best_idx]
                # Register alias if unseen
                if clean_text not in self.attributes[matched_id]["aliases"]:
                    self.attributes[matched_id]["aliases"].append(clean_text)
                
                print(f"    ├─ [MERGE] '{clean_text}' matched '{best_match_name}' (Sim: {best_score:.4f})")
                return matched_id
            else:
                # Log the highest score that failed to pass the threshold
                print(f"    ├─ [MISS] '{clean_text}' missed threshold. Highest match was '{best_match_name}' (Sim: {best_score:.4f} < {self.threshold})")

        # 3. Create a brand new attribute node if < threshold
        new_id = f"attr_{clean_text.replace(' ', '_')}"
        
        # Ensure unique ID key
        suffix = 1
        base_id = new_id
        while new_id in self.attributes:
            new_id = f"{base_id}_{suffix}"
            suffix += 1

        self.attributes[new_id] = {
            "canonical_name": clean_text.replace(" ", "_"),
            "aliases": [clean_text],
            "embedding": query_vec.tolist(),
            "objects": []
        }
        print(f"    ├─ [NEW NODE] Created attribute node: '{new_id}'")
        return new_id
        
    def add_object_with_attributes(self, object_name: str, raw_attributes: list):
        """
        Links an object node to semantically matched or newly created attribute nodes.
        """
        obj_key = object_name.strip().lower()
        if not obj_key:
            return

        if obj_key not in self.objects:
            self.objects[obj_key] = []

        for attr in raw_attributes:
            attr_node_id = self._find_or_create_attribute_node(attr)
            if not attr_node_id:
                continue

            # Link Object -> Attribute Node
            if attr_node_id not in self.objects[obj_key]:
                self.objects[obj_key].append(attr_node_id)

            # Link Attribute Node -> Object
            if obj_key not in self.attributes[attr_node_id]["objects"]:
                self.attributes[attr_node_id]["objects"].append(obj_key)

        self.save()


# ==========================================
# 3. SLM ATOMIC DECONSTRUCTION
# ==========================================
def extract_atomic_attributes(object_name: str) -> list:
    """Uses CPU SLM to split an object into primitive physical traits."""
    tokenizer, model = get_slm()

    system_prompt = (
        "You are an ontological physics deconstructor. Break down a physical object into atomic physical attributes.\n"
        "An atomic attribute cannot be broken down further.\n"
        "Rules:\n"
        "1. No compound phrases (e.g., instead of 'sharp blade', output 'sharp_edge' and 'rigid_body').\n"
        "2. Focus on geometry, material, mechanical mechanics, and physical traits.\n"
        "3. Return ONLY a valid JSON list of short strings."
    )

    user_prompt = f"""Deconstruct object: "{object_name}"

Return format:
```json
["attribute_1", "attribute_2", "attribute_3"]
```"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(CPU_DEVICE)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=150, do_sample=False)

    generated_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    raw_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    try:
        match = re.search(r"```json\s*(.*?)\s*```", raw_text, re.DOTALL)
        if not match:
            match = re.search(r"(\[.*\])", raw_text, re.DOTALL)
        if match:
            return json.loads(match.group(1).strip())
    except Exception as e:
        print(f"[-] Parsing error for '{object_name}': {e}")

    return []


# ==========================================
# 4. RUNNER & DEMONSTRATION
# ==========================================
if __name__ == "__main__":
    kg = VectorAtomicKnowledgeGraph()

    # Seed list with synonymous properties to verify deduplication
    test_ingestion = [
            # 1. Containers & Vessels
            ("thermos", ["hollow cavity", "watertight vessel", "thermal insulator", "cylindrical body", "metallic"]),
            ("colander", ["concave surface", "perforated surface", "rigid body", "metallic", "water permeable"]),
            ("spray_bottle", ["watertight vessel", "spray nozzle", "plastic material", "cylindrical body", "pump mechanism"]),
    
            # 2. Hardware & Tools
            ("adjustable_wrench", ["torsional grip", "rigid body", "heavy mass", "metallic", "adjustable jaw"]),
            ("wood_chisel", ["sharp edge", "rigid wedge", "metallic", "wooden handle", "flat impact surface"]),
            ("sandpaper", ["abrasive surface", "flexible sheet", "paper backing", "rough texture", "flat surface"]),
            ("plunger", ["suction head", "flexible rubber", "extended handle", "wooden pole", "airtight seal"]),
    
            # 3. Fasteners & Bindings
            ("wood_screw", ["pointed tip", "threaded shaft", "metallic", "rigid body", "slotted head"]),
            ("steel_nail", ["pointed tip", "smooth shaft", "metallic", "rigid body", "flat impact surface"]),
            ("zip_tie", ["flexible strip", "plastic material", "locking mechanism", "toothed edge", "high tensile strength"]),
    
            # 4. Cleaning & Maintenance
            ("kitchen_sponge", ["absorbent fibers", "porous surface", "flexible body", "soft material", "water retentive"]),
            ("wire_brush", ["bristled edge", "metallic bristles", "rigid handle", "abrasive surface", "stiff texture"])
        ]

    print("=" * 60)
    print(f"[*] INGESTING OBJECTS (Cosine Similarity Threshold = {SIMILARITY_THRESHOLD})")
    print("=" * 60)

    for obj_name, raw_attrs in test_ingestion:
        print(f"\n[+] Ingesting: '{obj_name}' with properties: {raw_attrs}")
        kg.add_object_with_attributes(obj_name, raw_attrs)

    print("\n" + "=" * 60)
    print(f"[*] Ingestion Complete!")
    print(f"[*] Total Objects Ingested : {len(kg.objects)}")
    print(f"[*] Unique Attribute Nodes : {len(kg.attributes)}")
    print(f"[*] Graph saved to         : {kg.filepath.resolve()}")
    print("=" * 60)