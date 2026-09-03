"""Two-stage task-vector benchmark: Florence boxes, then Qwen2.5-VL selection."""

import argparse
import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "sample_photos"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "qwen_models" / "Qwen2.5-VL-3B-Instruct"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "benchmark_outputs"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FLORENCE_PYTHON = PROJECT_ROOT / "edge_project" / "florence_env" / "bin" / "python"
FLORENCE_STAGE_SCRIPT = PROJECT_ROOT / "florence_benchmark_stage.py"


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def natural_path_key(path: Path) -> list[Any]:
    """Sort sequential filenames naturally: 2.jpg before 10.jpg."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.as_posix())]


def list_images(input_dir: Path) -> list[Path]:
    images = [path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(images, key=lambda path: natural_path_key(path.relative_to(input_dir)))


def build_candidates(parsed_od: dict[str, Any]) -> list[dict[str, Any]]:
    detection_data = parsed_od.get("<OD>", {})
    candidates = []
    for index, (label, bbox) in enumerate(
        zip(detection_data.get("labels", []), detection_data.get("bboxes", [])), start=1
    ):
        candidates.append({"id": f"object_{index}", "label": str(label), "bbox": [float(value) for value in bbox]})
    return candidates


def build_selection_prompt(task_vector: str, task_prompt: str, candidates: list[dict[str, Any]]) -> str:
    candidate_lines = "\n".join(f"- {item['id']}: {item['label']}" for item in candidates)
    rendered_prompt = task_prompt.replace("{task_vector}", task_vector)
    return f"""You are stage 2 of a task-object benchmark.
Task vector: {task_vector}
User task prompt: {rendered_prompt}

Look at the image and select only detected objects that answer the user task prompt.
You may select only IDs from the candidate list. Do not invent objects or IDs.
Select an object only when it is visibly supported and task-relevant. If there is not enough evidence, select no objects.

Candidates from stage 1:
{candidate_lines or '- No candidates were detected.'}

Return exactly one JSON object, with no Markdown:
{{"task_vector":"{task_vector}","selected_ids":["object_1"],"reason":"short visual reason"}}
"""


def extract_json_object(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


def select_candidates(response: str, candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str | None]:
    data = extract_json_object(response)
    if not data or not isinstance(data.get("selected_ids"), list):
        return [], "Qwen response did not contain valid selected_ids JSON."
    by_id = {item["id"]: item for item in candidates}
    selected = [by_id[item_id] for item_id in data["selected_ids"] if item_id in by_id]
    return selected, None


def draw_selected_boxes(image_path: Path, selected: list[dict[str, Any]], output_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font_size = min(20, max(14, image.width // 80))
    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
    except OSError:
        font = ImageFont.load_default()

    for item in selected:
        x1, y1, x2, y2 = item["bbox"]
        label = f"{item['id']}: {item['label']}"
        draw.rectangle([x1, y1, x2, y2], outline="red", width=5)
        text_box = draw.textbbox((0, 0), label, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        text_x = x1 + 4
        text_y = max(y1 - text_height - 10, 0)
        draw.rectangle([text_x - 4, text_y - 4, text_x + text_width + 4, text_y + text_height + 4], fill="red")
        draw.text((text_x, text_y), label, fill="white", font=font)
    image.save(output_path)


def run_florence_stage(image_path: Path, record_path: Path, boxed_path: Path) -> tuple[str, dict[str, Any]]:
    """Use Florence's original virtual environment to avoid package conflicts."""
    command = [
        str(FLORENCE_PYTHON),
        str(FLORENCE_STAGE_SCRIPT),
        str(image_path),
        "--json-output",
        str(record_path),
        "--boxed-output",
        str(boxed_path),
    ]
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)
    stage_data = json.loads(record_path.read_text())
    return stage_data["raw"], stage_data["parsed"]


def load_qwen(model_path: Path):
    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(
        model_path,
        min_pixels=256 * 28 * 28,
        max_pixels=640 * 28 * 28,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to("cpu").eval()
    return model, processor, process_vision_info


def ask_qwen(model: Any, processor: Any, process_vision_info: Any, image_path: Path, prompt: str, max_new_tokens: int) -> str:
    import torch

    messages = [{
        "role": "user",
        "content": [
            # qwen-vl-utils accepts local paths directly. Using a file URI here
            # breaks paths containing spaces because its loader does not unquote %20.
            {"type": "image", "image": str(image_path.resolve())},
            {"type": "text", "text": prompt},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to("cpu")
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed_ids = [output[len(input_ids):] for input_ids, output in zip(inputs.input_ids, generated_ids)]
    return processor.batch_decode(trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def write_report(run_dir: Path, task_vector: str, task_prompt: str, model_path: Path, results: list[dict[str, Any]]) -> None:
    total_stage1 = sum(item.get("stage1_seconds", 0) for item in results)
    total_stage2 = sum(item.get("stage2_seconds", 0) for item in results)
    selected_total = sum(len(item.get("selected_items", [])) for item in results)
    report = {
        "benchmark_name": run_dir.name,
        "task_vector": task_vector,
        "task_prompt": task_prompt,
        "stage_1": "Florence-2 object detection and bounding boxes",
        "stage_2": "Qwen2.5-VL-3B task-aware selection from Florence candidates",
        "model_path": str(model_path),
        "image_count": len(results),
        "selected_item_count": selected_total,
        "total_stage1_seconds": round(total_stage1, 3),
        "total_stage2_seconds": round(total_stage2, 3),
        "results": results,
    }
    (run_dir / "benchmark_report.json").write_text(json.dumps(report, indent=2))

    markdown = [
        "# Two-Stage Task-Vector Benchmark Report",
        "",
        f"- Task vector: `{task_vector}`",
        f"- User task prompt: `{task_prompt}`",
        "- Stage 1: Florence-2 detects objects and creates grounded bounding boxes.",
        "- Stage 2: Qwen2.5-VL-3B receives the image and Florence candidate IDs, then selects task-required IDs.",
        f"- Images processed: {len(results)}",
        f"- Selected items: {selected_total}",
        f"- Stage 1 total time: {total_stage1:.2f} seconds",
        f"- Stage 2 total time: {total_stage2:.2f} seconds",
        "",
        "## Manual Review",
        "Open `final_task_boxes/` and check whether each red box is a correct required item for the task vector.",
        "",
        "| Image | Florence candidates | Qwen selected items | Stage 1 sec | Stage 2 sec | Manual verdict | Notes |",
        "|---|---|---|---:|---:|---|---|",
    ]
    for item in results:
        candidates = ", ".join(candidate["label"] for candidate in item.get("florence_candidates", [])) or "none"
        selected = ", ".join(candidate["label"] for candidate in item.get("selected_items", [])) or "none"
        markdown.append(
            f"| {item['image_id']} | {candidates} | {selected} | {item.get('stage1_seconds', 0):.2f} | "
            f"{item.get('stage2_seconds', 0):.2f} | pending | |"
        )
    (run_dir / "benchmark_report.md").write_text("\n".join(markdown) + "\n")

    review_template = [
        {
            "image_id": item["image_id"],
            "final_boxed_image": item["final_boxed_image"],
            "manual_verdict": "pending",
            "correct_selected_ids": [],
            "false_positive_ids": [],
            "false_negative_description": "",
            "notes": "",
        }
        for item in results
    ]
    (run_dir / "manual_review_template.json").write_text(json.dumps(review_template, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Florence -> Qwen2.5-VL task-vector benchmark.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Folder containing benchmark images.")
    parser.add_argument("--task-vector", required=True, help="Task prompt, for example: open_parcel.")
    parser.add_argument(
        "--prompt",
        default="Identify only the items required to perform {task_vector}.",
        help="Task instruction for Qwen. Use {task_vector} as an optional placeholder.",
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", help="Optional output folder name.")
    parser.add_argument("--limit", type=int, help="Process only the first N images for a quick test.")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    images = list_images(args.input_dir)
    if args.limit is not None:
        images = images[:args.limit]
    if not images:
        raise FileNotFoundError(f"No supported images found in {args.input_dir}")
    if not args.model_path.exists():
        raise FileNotFoundError(f"Qwen2.5-VL model not found at {args.model_path}")
    if not FLORENCE_PYTHON.exists():
        raise FileNotFoundError(f"Florence environment not found at {FLORENCE_PYTHON}")

    run_name = args.run_name or f"{slugify(args.task_vector)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = args.output_root / run_name
    stage1_dir = run_dir / "stage1_florence_boxes"
    final_dir = run_dir / "final_task_boxes"
    records_dir = run_dir / "per_image_records"
    for directory in (stage1_dir, final_dir, records_dir):
        directory.mkdir(parents=True, exist_ok=True)

    print(f"Loading Qwen2.5-VL-3B from {args.model_path} on CPU...")
    model, processor, process_vision_info = load_qwen(args.model_path)
    results = []
    for image_path in images:
        image_id = slugify(str(image_path.relative_to(args.input_dir).with_suffix("")))
        print(f"Processing {image_path.name}")

        stage1_start = time.perf_counter()
        florence_stage_record = records_dir / f"{image_id}_florence.json"
        florence_boxed_path = stage1_dir / f"{image_id}_florence_od.jpg"
        florence_raw, florence_parsed = run_florence_stage(image_path, florence_stage_record, florence_boxed_path)
        candidates = build_candidates(florence_parsed)
        stage1_seconds = time.perf_counter() - stage1_start

        selection_prompt = build_selection_prompt(args.task_vector, args.prompt, candidates)
        stage2_start = time.perf_counter()
        qwen_response = ask_qwen(model, processor, process_vision_info, image_path, selection_prompt, args.max_new_tokens)
        stage2_seconds = time.perf_counter() - stage2_start
        selected, parse_error = select_candidates(qwen_response, candidates)
        final_boxed_path = final_dir / f"{image_id}_task_items.jpg"
        draw_selected_boxes(image_path, selected, final_boxed_path)

        record = {
            "image_id": image_id,
            "source_image": str(image_path.resolve()),
            "task_vector": args.task_vector,
            "user_task_prompt": args.prompt.replace("{task_vector}", args.task_vector),
            "stage1_seconds": round(stage1_seconds, 3),
            "stage2_seconds": round(stage2_seconds, 3),
            "florence_raw": florence_raw,
            "florence_candidates": candidates,
            "qwen_prompt": selection_prompt,
            "qwen_response": qwen_response,
            "selected_items": selected,
            "selection_parse_error": parse_error,
            "florence_boxed_image": str(florence_boxed_path.resolve()),
            "final_boxed_image": str(final_boxed_path.resolve()),
        }
        (records_dir / f"{image_id}.json").write_text(json.dumps(record, indent=2))
        results.append(record)

    write_report(run_dir, args.task_vector, args.prompt.replace("{task_vector}", args.task_vector), args.model_path, results)
    print(f"\nBenchmark complete: {run_dir}")
    print(f"Final task boxes: {final_dir}")
    print(f"Report: {run_dir / 'benchmark_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
