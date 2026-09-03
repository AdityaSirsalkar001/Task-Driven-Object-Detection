"""Batch version of Anshdeep's two-stage Qwen VLM -> Florence OVD pipeline."""

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw, ImageFont
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForMultimodalLM, AutoProcessor, Florence2ForConditionalGeneration


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_QWEN_PATH = PROJECT_ROOT / "qwen_models" / "Qwen2.5-VL-3B-Instruct"
DEFAULT_FLORENCE_PATH = PROJECT_ROOT / "qwen_models" / "Florence-2-base-ft-community"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output" / "anshdeep_basic_pipeline"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
OVD_TASK = "<OPEN_VOCABULARY_DETECTION>"


def natural_key(path: Path) -> list[Any]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.as_posix())]


def list_images(input_dir: Path) -> list[Path]:
    images = [path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(images, key=lambda path: natural_key(path.relative_to(input_dir)))


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def parse_object_list(response: str) -> list[str]:
    """Turn Qwen's intended comma-separated reply into safe OVD queries."""
    response = response.replace("\n", ",")
    candidates = []
    for value in response.split(","):
        cleaned = re.sub(r"^[\s\-•\d.]+|[\s.!?]+$", "", value).strip(" '\"")
        if cleaned and len(cleaned) <= 80:
            candidates.append(cleaned)
    return list(dict.fromkeys(candidates))


def load_models(qwen_path: Path, florence_path: Path):
    qwen_processor = AutoProcessor.from_pretrained(
        qwen_path,
        min_pixels=256 * 28 * 28,
        max_pixels=640 * 28 * 28,
    )
    qwen_model = AutoModelForMultimodalLM.from_pretrained(
        qwen_path,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to("cpu").eval()
    florence_processor = AutoProcessor.from_pretrained(florence_path)
    florence_model = Florence2ForConditionalGeneration.from_pretrained(
        florence_path,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).to("cpu").eval()
    return qwen_model, qwen_processor, florence_model, florence_processor


def qwen_propose_objects(model: Any, processor: Any, image_path: Path, user_prompt: str, max_new_tokens: int) -> tuple[str, list[str]]:
    instruction = (
        f"Look at the image. {user_prompt} "
        "Identify only the visible objects that answer this request. "
        "Return only a comma-separated list of short object noun phrases. "
        "Do not include explanations, numbering, Markdown, or any other text."
    )
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": str(image_path.resolve())},
            {"type": "text", "text": instruction},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to("cpu")
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    raw_response = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    return raw_response, parse_object_list(raw_response)


def florence_ground_objects(model: Any, processor: Any, image: Image.Image, object_queries: list[str]) -> list[dict[str, Any]]:
    detections = []
    for query in object_queries:
        inputs = processor(text=OVD_TASK + query, images=image, return_tensors="pt").to("cpu")
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                do_sample=False,
                num_beams=3,
            )
        generated_text = processor.batch_decode(output_ids, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(generated_text, task=OVD_TASK, image_size=image.size)
        result = parsed.get(OVD_TASK, {})
        if isinstance(result, str):
            continue
        labels = result.get("bboxes_labels", result.get("labels", []))
        for label, bbox in zip(labels, result.get("bboxes", [])):
            detections.append({"query": query, "label": str(label), "bbox": [float(value) for value in bbox]})
    return detections


def draw_boxes(image: Image.Image, detections: list[dict[str, Any]], output_path: Path) -> None:
    rendered = image.copy()
    draw = ImageDraw.Draw(rendered)
    font_size = min(20, max(14, rendered.width // 80))
    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
    except OSError:
        font = ImageFont.load_default()
    for item in detections:
        x1, y1, x2, y2 = item["bbox"]
        label = item["label"]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=5)
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_x = x1 + 4
        text_y = max(y1 - (text_bbox[3] - text_bbox[1]) - 10, 0)
        draw.rectangle([text_x - 4, text_y - 4, text_x + (text_bbox[2] - text_bbox[0]) + 4, text_y + (text_bbox[3] - text_bbox[1]) + 4], fill="red")
        draw.text((text_x, text_y), label, fill="white", font=font)
    rendered.save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Anshdeep's Qwen -> Florence two-stage basic pipeline in batch mode.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--prompt", required=True, help="Question given to Qwen for every image.")
    parser.add_argument("--qwen-path", type=Path, default=DEFAULT_QWEN_PATH)
    parser.add_argument("--florence-path", type=Path, default=DEFAULT_FLORENCE_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    images = list_images(args.input_dir)
    if args.limit is not None:
        images = images[:args.limit]
    if not images:
        raise FileNotFoundError(f"No supported images found in {args.input_dir}")
    for required_path in (args.qwen_path, args.florence_path):
        if not required_path.exists():
            raise FileNotFoundError(f"Model not found at {required_path}")

    run_name = args.run_name or f"{slugify(args.prompt)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = args.output_root / run_name
    boxed_dir = run_dir / "boxed_images"
    records_dir = run_dir / "per_image_records"
    boxed_dir.mkdir(parents=True, exist_ok=True)
    records_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Qwen2.5-VL and Florence community models on CPU...")
    qwen_model, qwen_processor, florence_model, florence_processor = load_models(args.qwen_path, args.florence_path)
    records = []
    for image_path in images:
        image_id = slugify(str(image_path.relative_to(args.input_dir).with_suffix("")))
        print(f"Processing {image_path.name}")
        image = Image.open(image_path).convert("RGB")

        stage1_start = time.perf_counter()
        qwen_response, qwen_objects = qwen_propose_objects(qwen_model, qwen_processor, image_path, args.prompt, args.max_new_tokens)
        stage1_seconds = time.perf_counter() - stage1_start
        stage2_start = time.perf_counter()
        detections = florence_ground_objects(florence_model, florence_processor, image, qwen_objects)
        stage2_seconds = time.perf_counter() - stage2_start

        boxed_path = boxed_dir / f"{image_id}_boxed.jpg"
        draw_boxes(image, detections, boxed_path)
        record = {
            "image_id": image_id,
            "source_image": str(image_path.resolve()),
            "prompt": args.prompt,
            "qwen_raw_response": qwen_response,
            "qwen_object_queries": qwen_objects,
            "florence_detections": detections,
            "qwen_stage_seconds": round(stage1_seconds, 3),
            "florence_stage_seconds": round(stage2_seconds, 3),
            "boxed_image": str(boxed_path.resolve()),
        }
        (records_dir / f"{image_id}.json").write_text(json.dumps(record, indent=2))
        records.append(record)

    report = {
        "pipeline": "Anshdeep basic pipeline: Qwen2.5-VL -> Florence open-vocabulary detection",
        "prompt": args.prompt,
        "image_count": len(records),
        "total_qwen_seconds": round(sum(item["qwen_stage_seconds"] for item in records), 3),
        "total_florence_seconds": round(sum(item["florence_stage_seconds"] for item in records), 3),
        "records": records,
    }
    (run_dir / "pipeline_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nComplete. Boxed images: {boxed_dir}")
    print(f"Per-image JSON: {records_dir}")
    print(f"Report: {run_dir / 'pipeline_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
