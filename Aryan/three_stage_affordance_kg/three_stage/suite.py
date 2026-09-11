"""Batch runner that pairs recognized dataset folders with canonical tasks."""
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

from .config import discover_samples
from .evaluation import benchmark_folder
from .pipeline import ThreeStagePipeline


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


def _natural_key(path):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r'(\d+)', str(path))]


def run_suite(args, config, save_json, visualize, debug_trace):
    """Run the first N images in every folder whose name is a KG task alias."""
    if args.limit_per_task < 1:
        raise ValueError('--limit-per-task must be positive')
    dataset = (args.input_dir or discover_samples(config)).resolve()
    pipeline = ThreeStagePipeline(config)
    task_folders = []
    for folder in sorted((p for p in dataset.iterdir() if p.is_dir()), key=_natural_key):
        task = pipeline.kg.normalize_task(folder.name)
        if task != 'unknown':
            task_folders.append((task, folder))
    if not task_folders:
        raise FileNotFoundError(f'No recognized task folders found under {dataset}')

    run_name = args.run_name or datetime.now(timezone.utc).strftime('suite_%Y%m%dT%H%M%S_%f')
    if Path(run_name).name != run_name or run_name in ('.', '..'):
        raise ValueError('--run-name must be a simple folder name')
    run_dir = Path(config['results_path']) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'dataset': str(dataset),
        'limit_per_task': args.limit_per_task,
        'status': 'running',
        'cases': [],
    }
    save_json(run_dir / 'manifest.json', manifest)
    print('Loading Florence once for the complete task suite...', flush=True)
    pipeline.prepare(use_slm=False)
    annotations = []
    errors = 0
    for task, folder in task_folders:
        images = sorted(
            (p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
            key=_natural_key,
        )[:args.limit_per_task]
        for image in images:
            image = image.resolve()
            image_key = image.stem + '_' + hashlib.sha256(str(image).encode()).hexdigest()[:10]
            case_dir = run_dir / task / image_key
            print(f'{task}: {image.name}', flush=True)
            try:
                result = pipeline.run(task, image, task)
                result['requested_task'] = task
                save_json(case_dir / 'result.json', result)
                if config['visualize'] and not args.no_visualization:
                    visualize(result, case_dir / 'visualization.jpg')
                if args.debug_trace:
                    debug_trace(result)
                annotations.append({
                    'image': str(image),
                    'image_sha256': result['image_sha256'],
                    'prompt': task,
                    'task': task,
                    'relevant_objects': None,
                    'annotation_status': 'pending_manual_review',
                })
                manifest['cases'].append({'task': task, 'image': str(image), 'status': result['final']['status']})
            except Exception as exc:
                errors += 1
                save_json(case_dir / 'error.json', {'task': task, 'image': str(image), 'error': str(exc), 'type': type(exc).__name__})
                manifest['cases'].append({'task': task, 'image': str(image), 'status': 'error', 'error': str(exc)})
                print(f'  FAILED: {exc}', flush=True)
    save_json(run_dir / 'annotation_template.json', annotations)
    manifest.update(status='completed_with_errors' if errors else 'completed', errors=errors, model_load_ms=pipeline.load_ms)
    save_json(run_dir / 'manifest.json', manifest)
    save_json(run_dir / 'benchmark.json', benchmark_folder(run_dir))
    print(f'Results: {run_dir}', flush=True)
    return 1 if errors else 0
