# Two-Stage Task-Vector Benchmark

## What it does

1. Florence-2 detects objects and creates bounding boxes.
2. Qwen2.5-VL-3B sees the original image, task vector, and Florence object IDs.
3. Qwen selects only IDs required for the task.
4. The final image shows only selected objects in red boxes.

## Add images

Create `sample_photos/` in the project folder and add `.jpg`, `.jpeg`, `.png`, `.webp`, or `.bmp` images.

## Quick test

```bash
./run_task_vector_benchmark.sh --input-dir sample_photos --task-vector "open_parcel" --limit 1
```

## Full benchmark

```bash
./run_task_vector_benchmark.sh --input-dir sample_photos --task-vector "open_parcel" --run-name open_parcel_benchmark
```

## Output

Each run is saved in `benchmark_outputs/<run-name>/`.

- `stage1_florence_boxes/`: all Florence detections in green.
- `final_task_boxes/`: only Qwen-selected task items in red.
- `per_image_records/`: full evidence for each image.
- `benchmark_report.md`: report to inspect and submit.
- `manual_review_template.json`: fill this after checking the final red boxes.
