# Three-Stage Task-Affordance Retrieval Pipeline

## System Overview

```text
User Prompt -> Qwen2.5-3B-Instruct (Task Normalization)
            -> Task-Affordance Knowledge Graph Traversal
            -> Florence-2-base-ft (Visual Open-Vocabulary Grounding)
            -> Role-Specific Ranking & Deduplication (JSON Trace + Bounding Box Overlay)
```

This repository implements a three-stage task-conditioned visual object retrieval architecture:
1. **Stage 1 (Text Normalization)**: Qwen2.5-3B-Instruct maps user natural language input (e.g. *"I need to open this box"*) to one canonical task string (`open_parcel`).
2. **Stage 2 (Affordance Knowledge Graph)**: Traverses a persistent task-affordance Knowledge Graph (`task_affordance_kg.yaml`) using NetworkX (`three_stage/knowledge/affordance_graph.py`). Retrieves task semantic roles, required functional affordances, candidate object classes, and visual detector queries (aliases).
3. **Stage 3 (Visual Grounding)**: Florence-2-base-ft (`<OPEN_VOCABULARY_DETECTION>`) visually verifies **only** the detector aliases retrieved by the KG traversal.
4. **Evidence Filtering & Deduplication**: Valid boxes are grouped only inside the same semantic family. A canonical query or at least two strongly overlapping aliases can support an unscored detection. Standard Florence `<OD>` is optional supporting evidence, not a hard veto.

> [!NOTE]
> **Key Architectural Design**:
> - **No Scene Graph**: Does NOT construct or compute object-to-object spatial relationships.
> - **No Sentence-Transformer Attribute Embeddings**: Attributes are stored as secondary metadata but are not used for retrieval or ranking in this version.
> - **Affordance Gating**: Object classes must possess the required affordance to fulfill an instrument or destination role for a task.
> - **Single Task Scope**: Maps every input to a single canonical task context before graph traversal.

---

## Repository Structure

| File / Component | Responsibility |
| --- | --- |
| `../task_affordance_kg.yaml` | Complete persistent task-affordance-object Knowledge Graph specification |
| `../run_pipeline.py` | Single main Python entry point |
| `three_stage/config.py` | Central configuration management & path resolution |
| `three_stage/config.json` | JSON configuration defaults |
| `three_stage/parser.py` | Qwen2.5-3B-Instruct task parser with canonical normalization fallback |
| `three_stage/knowledge/affordance_graph.py` | NetworkX task graph builder, validation engine, and traversal logic |
| `three_stage/knowledge/detector_vocab.py` | Concept-to-alias vocabulary mapping utilities |
| `three_stage/detector.py` | Florence-2 open-vocabulary detection wrapper |
| `three_stage/ranking.py` | Conservative frame checks, cross-query IoU clustering, label resolution, and role ranking |
| `three_stage/pipeline.py` | `ThreeStagePipeline` orchestrator |
| `three_stage/__main__.py` | CLI entry point supporting `run`, `validate-kg`, `query-kg`, and `--debug-trace` |
| `three_stage/evaluation.py` | Benchmark evaluation framework including `false_positives_per_image` metric |
| `three_stage/tests/test_pipeline.py` | Model-free tests covering validation, graph traversal, evaluation, and ranking |
| `three_stage/ablations/` | Controlled ablation configurations (`no_affordances`, `no_hierarchy`, `no_kg_weights`, `no_aliases`) |
| `run_three_stage.sh` | Convenience wrapper script executing inside the project environment |

---

## Knowledge Graph Schema & Task Coverage

The Knowledge Graph is defined in `task_affordance_kg.yaml`.

### Graph Structure
- **Nodes**: `Task`, `Role`, `Affordance`, `ObjectCategory`, `ObjectClass`, `DetectorAlias`.
- **Relations**:
  - `REQUIRES_ROLE`: `Task` -> `Role` (e.g. `open_parcel` -> `instrument`, `target`)
  - `REQUIRES_AFFORDANCE`: `Role` -> `Affordance` (e.g. `instrument` -> `opening_package`)
  - `HAS_AFFORDANCE`: `ObjectClass` -> `Affordance` (e.g. `box_cutter` -> `opening_package`)
  - `IS_A`: `ObjectClass` -> `ObjectCategory` (e.g. `shovel` -> `digging_tool`)
  - `DETECTOR_ALIAS`: `ObjectClass` -> `DetectorAlias` (e.g. `box_cutter` -> `["box cutter", "utility knife", "cutter"]`)

### 9 Canonical Benchmark Tasks

| Canonical Task | Main task knowledge | Candidate concepts |
| --- | --- | --- |
| `dig_hole` | Digging, breaking ground, scooping, boring, and trenching tools | 18 |
| `extinguish_fire` | Direct suppression, suppressant delivery, smothering, and fixed systems | 20, including fire context |
| `open_parcel` | Cutting packaging, slicing tape, and opening tools | 13, with instrument as required output |
| `place_flowers` | Holding stems, supporting plants, and temporary containers | 20, including flower context |
| `pour_sugar` | Holding, scooping, dispensing, and pouring granules | 17 |
| `sit_comfortably` | Supporting a seated person and cushioned seating | 15 |
| `smear_butter` | Holding and spreading soft food | 14 |
| `step_on` | Supporting body weight and elevated stepping | 16 |
| `wine` | Holding, opening, pouring, and serving wine | 16 |

The current validated KG contains 481 graph nodes, 841 graph edges, 140 object classes, 34 affordances, 33 categories, and 215 detector aliases. `task_affordance_kg.yaml` is the authoritative source for the complete candidate lists and relevance priors.

---

## Ranking & Deduplication

### Scoring Formula
When visual detection confidence ($OD_{conf}$) is returned:
$$\text{Score} = w_{od} \cdot OD_{conf} + w_{kg} \cdot KG_{prior}$$
By default: $w_{od} = 0.7$, $w_{kg} = 0.3$.

When Florence returns bounding boxes without explicit confidence scores ($OD_{conf} = \text{null}$), no numeric final score is invented and KG relevance is not treated as visual confidence. An unscored box is accepted only when the canonical family query detects it, or when at least two aliases from the same semantic family overlap at IoU 0.70 or higher. The output records `score_basis: unscored_text_conditioned_florence`.

KG relevance remains semantic suitability only. It is not treated as visual confidence.

### Cross-Query Clustering
1. Florence runs precise KG-provided queries such as `bucket`, `pail`, and `water bucket`.
2. Only queries in the same semantic family may support one another. Unrelated concepts such as `bucket` and `sprinkler` never vote together.
3. Same-family query boxes with IoU 0.70 or higher become one physical-object cluster and preserve all supporting queries.
4. Standard Florence `<OD>` may add `reference_od_supported: true` and stronger evidence, but disagreement or omission does not reject an otherwise supported cluster.
5. Related bucket subtypes are conservatively returned as `bucket` unless independent evidence supports a subtype.
6. Only clearly pathological boxes covering more than 90% of the image are rejected by area. Large or tall valid objects are not rejected.

---

## Controlled Ablations

Four pre-configured ablations are available in `three_stage/ablations/`:

1. `no_affordances.json` (`use_affordances=False`): Bypasses the functional affordance gate. Candidate objects listed explicitly under a role are retrieved regardless of affordance validation.
2. `no_hierarchy.json` (`use_hierarchy=False`): Disables `IS_A` category traversal. Restricts node expansion to direct object relations.
3. `no_kg_weights.json` (`use_kg_weights=False`): Ignores KG relevance priors in final ranking ($w_{kg} = 0$). Unscored Florence detections remain unscored (`score = null`).
4. `no_aliases.json` (`use_detector_aliases=False`): Disables detector alias expansion. Queries Florence using canonical concept names only (e.g. `box cutter` instead of `utility knife`).

The current YAML lists candidates and affordances explicitly. Therefore, the affordance and hierarchy ablations mainly change graph gating or the active explanation subgraph; they may not change predictions for every task. This must be measured rather than assumed.

---

## Usage Commands

For a simple explanation of every command and flags such as `--start 20 --end 30`, see [COMMAND_GUIDE.md](COMMAND_GUIDE.md).

The short commands below use the convenience launcher in the current `RM_project` workspace. The Git-contained launcher is also available as `Anshdeep_RM/Aryan/three_stage_affordance_kg/run_pipeline.sh`.

### Knowledge Graph Tools
```bash
# Validate KG schema, node consistency, and edge references
./run_three_stage.sh validate-kg

# Query KG traversal for a canonical task
./run_three_stage.sh query-kg --task open_parcel
```

### Running Pipeline Inferences
```bash
# Single image run with task name override (SLM bypass)
./run_three_stage.sh run --image Dataset/open_parcel/1.jpg --task open_parcel --run-name parcel_test

# Full three-stage run from natural language user prompt
./run_three_stage.sh run --image Dataset/extnigh_fire/1.jpg --prompt "Find something to extinguish this fire"

# Run with full debug trace printed to terminal
./run_three_stage.sh run --image Dataset/open_parcel/1.jpg --task open_parcel --debug-trace

# Run numeric image filenames 20.jpg through 30.jpg (inclusive)
./run_three_stage.sh run --input-dir Dataset/open_parcel --start 20 --end 30 --task open_parcel --run-name parcel_20_to_30 --debug-trace

# One representative image from every recognized task folder; models load once
./run_three_stage.sh run-suite --input-dir Dataset --limit-per-task 1 --run-name all_tasks_test

# Controlled ablation run
./run_three_stage.sh --config Anshdeep_RM/Aryan/three_stage_affordance_kg/three_stage/ablations/no_affordances.json run --image Dataset/open_parcel/1.jpg --task open_parcel
```

### Running Unit Test Suite
```bash
# Run all model-free unit tests
Anshdeep_RM/Aryan/three_stage_affordance_kg/run_tests.sh
```

### Benchmarking

Every run creates `annotation_template.json`. Manually replace `relevant_objects: null` with visible, verified concept/role labels, then run:

```bash
./run_three_stage.sh benchmark \
  --run-dir results/three_stage_affordance_kg/parcel_test \
  --ground-truth results/three_stage_affordance_kg/parcel_test/annotation_template.json
```

Without completed manual labels, accuracy values remain `null`. Timing and graph-size values are still reported.

---

## Output Artifacts

Each pipeline run creates a directory under `results/three_stage_affordance_kg/<run-name>/`:
- `result.json`: Full machine-readable execution trace containing:
  - `stage1`: Normalized task, confidence, parser mode.
  - `stage2`: Subgraph summary, required affordances, roles, candidate objects, detector queries.
  - `stage3`: Raw Florence detection outputs and query mapping.
  - `final`: Status (`success` / `partial` / `no_suitable_object_detected`), per-role ranked objects, bounding boxes, IoU merged aliases, timing breakdowns.
- `visualization.jpg`: Red bounding boxes with the canonical concept, role, and available scores.

`success` means Florence returned a task-conditioned candidate for every required role. It does not prove that every box is correct. Results stay marked `pending_manual_review` until they are checked against manual ground truth.

Florence-2 in this setup returns boxes but no calibrated object confidence. The JSON therefore stores `od_confidence: null`, and `final_score` remains `null`. Accepted detections record `canonical_concept`, `supporting_queries`, `alias_support_count`, `reference_od_supported`, and `acceptance_reason`. Do not present KG relevance as visual confidence.

---

## Verification & Test Suite

The test suite in `three_stage/tests/test_pipeline.py` verifies:
- YAML loading and duplicate key rejection.
- Dangling object references, unknown categories, invalid affordances, invalid role names.
- Missing detector alias checks and relevance prior range validation $[0.0, 1.0]$.
- Graph connectivity and circular `IS_A` path detection.
- Required affordance gating and role requirement policies.
- Ablation switch behaviors (`use_affordances=False`, `use_hierarchy=False`, `use_kg_weights=False`, `use_detector_aliases=False`).
- Ranking metrics, same-family clustering, optional reference evidence, conservative bucket canonicalization, and `false_positives_per_image` evaluation metric.

The current model-free suite has 37 tests. Run it locally to verify the exact current count and result.

---

## Current Validation Status

- KG validation passes with no errors or warnings.
- The 37 model-free tests pass.
- A real `extinguish_fire` batch was started after the optional-reference change, but it was manually stopped after image 1 because CPU inference is slow. This partial run is not reported as a completed benchmark.
- Every visual result remains `pending_manual_review` until a person checks the box against ground truth.
