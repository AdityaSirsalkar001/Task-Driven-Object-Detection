import json
import os
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

SCRIPT_DIR = Path(__file__).parent
KG_FILE = SCRIPT_DIR / "hierarchical_kg.json"
SIMILARITY_THRESHOLD = 0.75

print("[*] Loading Semantic Embedding Engine (all-MiniLM-L6-v2)...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")


def load_graph():
    """Loads the hierarchical knowledge graph from disk."""
    if os.path.exists(KG_FILE):
        with open(KG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"tasks": {}, "actions": {}, "attributes": {}}


def save_graph(graph):
    """Writes graph updates to disk."""
    with open(KG_FILE, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2)


def cosine_similarity(vec1, vec2):
    dot = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    return float(dot / (norm_a * norm_b)) if (norm_a and norm_b) else 0.0


def semantic_match_task(user_query, graph):
    """Finds the closest existing task node using semantic cosine similarity."""
    tasks = list(graph.get("tasks", {}).keys())
    if not tasks:
        return None, 0.0

    query_vec = embedder.encode(user_query)
    task_vecs = embedder.encode(tasks)

    best_task = None
    highest_score = -1.0

    for task_name, t_vec in zip(tasks, task_vecs):
        score = cosine_similarity(query_vec, t_vec)
        if score > highest_score:
            highest_score = score
            best_task = task_name

    if highest_score >= SIMILARITY_THRESHOLD:
        return best_task, highest_score

    return None, highest_score


def traverse_graph_for_entities(task_name, graph):
    """
    Executes 3-hop traversal: Task -> REQUIRES_ACTION -> REQUIRES_ATTRIBUTE -> POSSESSED_BY -> Entities
    """
    task_data = graph.get("tasks", {}).get(task_name, {})
    actions = task_data.get("REQUIRES_ACTION", [])

    all_attributes = set()
    for act in actions:
        action_data = graph.get("actions", {}).get(act, {})
        req_attrs = action_data.get("REQUIRES_ATTRIBUTE", [])
        all_attributes.update(req_attrs)

    all_entities = set()
    for attr in all_attributes:
        attr_data = graph.get("attributes", {}).get(attr, {})
        entities = attr_data.get("POSSESSED_BY", [])
        all_entities.update(entities)

    return {
        "actions": list(actions),
        "attributes": list(all_attributes),
        "entities": sorted(list(all_entities))
    }