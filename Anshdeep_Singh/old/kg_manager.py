import json
import os
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION ---
SCRIPT_DIR = Path(__file__).parent
KG_FILE = SCRIPT_DIR / "attribute_knowledge_graph.json"
SIMILARITY_THRESHOLD = 0.80  # 80% similarity threshold for physical attributes

print("[*] Loading Semantic Embedding Engine (all-MiniLM-L6-v2)...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def load_graph():
    """Loads the attribute-based JSON graph or creates a default seed graph."""
    if os.path.exists(KG_FILE):
        with open(KG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    
    # Default seed graph focusing on physical attributes instead of specific tasks
    seed_graph = {
        "small handheld object with sharp blades": ["scissors", "knife", "box cutter", "blade", "key", "utility knife"],
        "hollow watertight container": ["cup", "glass", "mug", "bottle", "flask", "jug", "bowl"],
        "long flat rigid tool for prying": ["screwdriver", "crowbar", "butter knife", "ruler", "coin", "plier"],
        "absorbent material for wiping": ["towel", "tissue", "sponge", "cloth", "mop", "rag", "paper towel", "napkin"],
        "heavy blunt object for striking": ["hammer", "mallet", "heavy rock", "wrench", "baseball bat", "brick"]
    }
    save_graph(seed_graph)
    return seed_graph

def save_graph(graph):
    """Permanently writes updated attributes and tools to the JSON file."""
    with open(KG_FILE, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=4)

def cosine_similarity(vec1, vec2):
    """Calculates the mathematical similarity between two text vectors."""
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    return dot_product / (norm_a * norm_b) if (norm_a and norm_b) else 0.0

def semantic_search(extracted_attributes, graph):
    """Checks if the SLM's extracted attributes match any existing attributes in the graph."""
    if not graph:
        return None, 0.0

    # Convert the required physical attributes into a mathematical vector
    user_vec = embedder.encode(extracted_attributes)
    
    best_match = None
    highest_score = -1.0

    # Compare against all known attribute requirements in the database
    for saved_attributes in graph.keys():
        saved_vec = embedder.encode(saved_attributes)
        score = cosine_similarity(user_vec, saved_vec)
        
        if score > highest_score:
            highest_score = score
            best_match = saved_attributes

    if highest_score >= SIMILARITY_THRESHOLD:
        return best_match, float(highest_score)

    return None, float(highest_score)
