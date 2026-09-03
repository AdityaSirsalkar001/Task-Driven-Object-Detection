# Aryan - Two-Stage Task-Object Benchmarking

This folder contains my work on two different two-stage pipelines for finding task-relevant objects in images.

## Objective

Given an image and a task prompt, identify useful visible objects and save grounded bounding boxes for manual checking.

## Pipeline 1: Florence -> Qwen Selection

Folder: `florence_then_qwen/`

1. Florence-2 detects visible objects and creates all candidate boxes.
2. Qwen2.5-VL-3B receives the image, task vector, and Florence candidate IDs.
3. Qwen selects only task-relevant IDs.
4. The final result saves selected objects in red boxes.

This pipeline is useful when we want every final box to come directly from Florence detection.

## Pipeline 2: Qwen -> Florence Grounding

Folder: `qwen_then_florence_baseline/`

This is an adapted batch version of Anshdeep Singh's `basic_pipeline` approach.

1. Qwen2.5-VL-3B reads the image and user task prompt.
2. Qwen returns object names, such as `knife`.
3. Florence-2 open-vocabulary detection searches the image for those exact names.
4. Florence saves red grounded boxes.

This is a no-KG baseline. It is useful for comparing Qwen-first reasoning against Florence-first detection.

## Current Result Snapshot

Folder: `results/anshdeep_extinguish_firee_first_100_partial/`

- Dataset folder used locally: `sample_photos/extinguish_firee/`
- Prompt: `Identify only the visible items that can be used to cool or extinguish fire.`
- Completed at snapshot time: 17 images.
- `boxed_images/`: red-box result for each completed image.
- `per_image_records/`: one JSON record per completed image.

The original 100-image local run was still in progress when this snapshot was prepared. This repository stores only completed result files, not model weights or source images.

## Run Commands

From the project root, after downloading model weights and creating `qwen_env`:

```bash
./run_task_vector_benchmark.sh --input-dir sample_photos/extinguish_firee --task-vector "extinguish_fire" --prompt "Identify only the items required to cool fire in the {task_vector}." --limit 100 --output-root output --run-name florence_then_qwen_fire_100
```

```bash
./run_anshdeep_basic_pipeline.sh --input-dir sample_photos/extinguish_firee --prompt "Identify only the visible items that can be used to cool or extinguish fire." --limit 100 --output-root output --run-name qwen_then_florence_fire_100
```

## Notes

- Qwen2.5-VL-3B and Florence run locally on CPU in this setup.
- CPU execution is slow, so runs are designed for controlled benchmark samples and manual review.
- Each JSON file records the input prompt, model output, detected labels, box coordinates, and timing.
