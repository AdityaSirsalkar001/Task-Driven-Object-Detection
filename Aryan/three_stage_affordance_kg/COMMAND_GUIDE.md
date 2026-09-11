# Three-Stage Pipeline Command Guide

Run all commands from the repository root containing `run_three_stage.sh`.
The launcher expects `qwen_env`, `qwen_models`, and `Dataset` beside that repository in the parent project folder, matching the current local setup.

## 1. Test One Image

Use `--task` when you already know the task. This skips Qwen and directly tests the KG and Florence.

```bash
./run_three_stage.sh run \
  --image Dataset/open_parcel/5.jpg \
  --task open_parcel \
  --run-name open_parcel_image_5 \
  --debug-trace
```

Use `--prompt` to test all three stages, including Qwen.

```bash
./run_three_stage.sh run \
  --image Dataset/open_parcel/5.jpg \
  --prompt "What can I use to open this parcel?" \
  --run-name open_parcel_prompt_test \
  --debug-trace
```

## 2. Test Images 20 Through 30

This includes both `20.jpg` and `30.jpg`, so it processes 11 images.

```bash
./run_three_stage.sh run \
  --input-dir Dataset/open_parcel \
  --start 20 \
  --end 30 \
  --task open_parcel \
  --run-name open_parcel_20_to_30 \
  --debug-trace
```

The range works with numeric names such as `20.jpg`, `000020.jpg`, and `30.png`.

To process only five images from that range, also add:

```bash
--limit 5
```

## 3. Test the First N Images

This processes the first 10 images in natural filename order:

```bash
./run_three_stage.sh run \
  --input-dir Dataset/open_parcel \
  --limit 10 \
  --task open_parcel \
  --run-name open_parcel_first_10
```

## 4. Test Every Task Folder

Process one image from every recognized dataset folder:

```bash
./run_three_stage.sh run-suite \
  --input-dir Dataset \
  --limit-per-task 1 \
  --run-name all_tasks_one_image \
  --debug-trace
```

Process five images from every task folder:

```bash
./run_three_stage.sh run-suite \
  --input-dir Dataset \
  --limit-per-task 5 \
  --run-name all_tasks_five_images
```

## 5. Supported Task Names

Use one of these values with `--task`:

```text
dig_hole
extinguish_fire
open_parcel
place_flowers
pour_sugar
sit_comfortably
smear_butter
step_on
wine
```

The spelling `extingh_fire` is automatically normalized to `extinguish_fire`.

## 6. Flag Meanings

| Flag | Simple meaning |
| --- | --- |
| `run` | Run the pipeline on one image or one folder. |
| `run-suite` | Run matching tasks across all dataset task folders. |
| `--image PATH` | Process one exact image. |
| `--input-dir PATH` | Process images inside a folder. |
| `--task NAME` | Use a known task and skip Qwen. |
| `--prompt "TEXT"` | Ask Qwen to convert normal text into a task. |
| `--start N` | Start from numeric image filename N. Only for `--input-dir`. |
| `--end N` | Stop at numeric image filename N. Only for `--input-dir`. |
| `--limit N` | Process at most N selected images. |
| `--limit-per-task N` | Process at most N images from every task folder in `run-suite`. |
| `--run-name NAME` | Name the result folder. It must be new for every run. |
| `--debug-trace` | Print Qwen, KG, Florence, accepted boxes, and rejected boxes. |
| `--no-visualization` | Do not create boxed output images. JSON is still saved. |
| `--config FILE.json` | Use a custom configuration or ablation file. Place it before the command. |

Do not use `--task` and `--prompt` together. Do not use `--image` and `--input-dir` together.

## 7. Check the Knowledge Graph

Validate the full KG:

```bash
./run_three_stage.sh validate-kg
```

See candidates and aliases for one task:

```bash
./run_three_stage.sh query-kg --task open_parcel
```

Other examples:

```bash
./run_three_stage.sh query-kg --task extinguish_fire
./run_three_stage.sh query-kg --task dig_hole
./run_three_stage.sh query-kg --task place_flowers
```

## 8. Run Tests

These tests do not load the large Qwen or Florence models.

```bash
Anshdeep_RM/Aryan/three_stage_affordance_kg/run_tests.sh
```

## 9. Test an Ablation

The `--config` flag must come before `run`:

```bash
./run_three_stage.sh \
  --config Anshdeep_RM/Aryan/three_stage_affordance_kg/three_stage/ablations/no_affordances.json \
  run \
  --image Dataset/open_parcel/5.jpg \
  --task open_parcel \
  --run-name open_parcel_no_affordances
```

Available ablation files are:

```text
no_affordances.json
no_hierarchy.json
no_kg_weights.json
no_aliases.json
```

## 10. Find the Results

Results are stored here:

```text
results/three_stage_affordance_kg/<run-name>/
```

Important files:

| File | Meaning |
| --- | --- |
| `result.json` | Full output from Qwen, KG, Florence, filtering, and ranking. |
| `visualization.jpg` | Final accepted objects shown with boxes. |
| `manifest.json` | Run settings, image list, model versions, and run status. |
| `benchmark.json` | Timing and available evaluation values. |
| `annotation_template.json` | File to fill with manual correct answers for accuracy testing. |

Each image gets its own subfolder. The terminal prints the exact result folder after finishing.

## 11. Run a Benchmark

First manually complete `annotation_template.json`. Then run:

```bash
./run_three_stage.sh benchmark \
  --run-dir results/three_stage_affordance_kg/open_parcel_20_to_30 \
  --ground-truth results/three_stage_affordance_kg/open_parcel_20_to_30/annotation_template.json
```

Without manual correct labels, the program can report timing but cannot honestly calculate precision, recall, or F1.

## 12. Get Command Help

```bash
./run_three_stage.sh --help
./run_three_stage.sh run --help
./run_three_stage.sh run-suite --help
./run_three_stage.sh benchmark --help
```
