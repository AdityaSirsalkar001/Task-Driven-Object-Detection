import json
import os
# --- ADD THESE TWO LINES ---
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
# ---------------------------
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

SCRIPT_DIR = Path(__file__).parent
KG_FILE = SCRIPT_DIR / "knowledge_graph.json"
SIMILARITY_THRESHOLD = 0.80  # 80% similarity threshold

print("[*] Loading Semantic Embedding Engine (all-MiniLM-L6-v2)...")
embedder = SentenceTransformer('all-MiniLM-L6-v2', local_files_only=True)

def load_graph():
    if os.path.exists(KG_FILE):
        with open(KG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    
    # Default seed graph if file does not exist yet
    seed_graph = {
        "open a parcel": ["scissors", "knife", "box cutter", "blade", "key"]
    }
    save_graph(seed_graph)
    return seed_graph

def save_graph(graph):
    with open(KG_FILE, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=4)

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    return dot_product / (norm_a * norm_b) if (norm_a and norm_b) else 0.0

def semantic_search(user_task, graph):
    if not graph:
        return None, 0.0

    user_vec = embedder.encode(user_task)
    best_match = None
    highest_score = -1.0

    for saved_task in graph.keys():
        saved_vec = embedder.encode(saved_task)
        score = cosine_similarity(user_vec, saved_vec)
        
        if score > highest_score:
            highest_score = score
            best_match = saved_task

    if highest_score >= SIMILARITY_THRESHOLD:
        return best_match, float(highest_score)
    
    return None, float(highest_score)