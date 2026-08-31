# 📁 Attribute_KG

⬅ [Back to Aditya overview](../README.md)

**What it does:** Instead of keying the graph on the *task name*, it keys on the **physical attributes** the task needs — making the KG reusable across many similar tasks.

## Flow

```mermaid
flowchart TD
    A["User enters Task<br/>e.g. 'cut paper'"] --> B["SLM (Qwen2.5-3B)<br/>extracts physical attributes"]
    B --> C["e.g. 'small handheld object<br/>with sharp blades'"]
    C --> D[SentenceTransformer + Cosine Similarity<br/>vs Attribute KG]
    D --> E{Similarity ≥ 0.80?}
    E -- Yes --> F[Get candidate objects<br/>from KG]
    E -- No --> G["VLM Fallback:<br/>Qwen2.5-VL-3B scans image"]
    G --> H[Finds a potentially useful object]
    H --> I[Add object to Attribute KG]
    F --> J[Florence-2 searches image<br/>for candidates]
    I --> K[Florence-2 re-run<br/>to localize new object]
    J --> L[Draw boxes]
    K --> L
    L --> M["✅ Final Image"]
```

## Worked example

```
Task: "cut paper"
   ↓ SLM extracts attribute
"small handheld object with sharp blades"
   ↓ matched in Attribute KG
scissors / knife / box cutter
   ↓ Florence-2 locates in image
```

If no attribute matches, the pipeline **doesn't just give up** — the VLM looks at the image directly, finds something useful, and teaches the KG a new attribute → object mapping for next time.

## VRAM management

Three models (SLM, VLM, Florence-2) can't all fit comfortably, so only one Qwen model is loaded at a time:

```mermaid
flowchart LR
    A[SLM needed] --> B[Load SLM] --> C[Use it] --> D[Unload SLM]
    D --> E[Load VLM] --> F[Use it] --> G[Unload VLM]
```

**Florence-2 stays loaded permanently.** Whole pipeline runs within ~6 GB VRAM using 4-bit quantization.

## Files
| File | Role |
|---|---|
| `kg_vlm_slm_florence_v1.py` | Main pipeline + VRAM swapping |
| `kg_manager.py` | Attribute embedding + matching |
| `attribute_knowledge_graph.json` | Attribute → tool mappings |

## Output
```
attr_kg_output_{image}.jpg
```
Green boxes = KG hits, red boxes = new VLM discoveries.

## Run
```bash
cd Attribute_KG
python kg_vlm_slm_florence_v1.py
# try: "open the parcel"
```