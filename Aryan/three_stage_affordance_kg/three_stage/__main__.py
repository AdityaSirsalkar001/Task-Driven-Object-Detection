"""CLI: run samples, validate/query the KG, and benchmark saved results."""
import argparse
import hashlib
import json
import platform
import re
from datetime import datetime, timezone
from pathlib import Path

from .config import ROOT, discover_samples, load_config
from .knowledge.affordance_graph import TaskGraph
from .pipeline import ThreeStagePipeline


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False))
    temp.replace(path)


def select_images(directory, limit=None, start=None, end=None):
    """Select naturally sorted images, optionally by numeric filename range."""
    images = sorted(
        (
            path.resolve()
            for path in directory.rglob('*')
            if path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp', '.bmp'} and path.is_file()
        ),
        key=lambda path: [
            int(part) if part.isdigit() else part.lower()
            for part in re.split(r'(\d+)', str(path))
        ],
    )
    if start is not None or end is not None:
        ranged = []
        for image in images:
            if not image.stem.isdigit():
                continue
            number = int(image.stem)
            if (start is None or number >= start) and (end is None or number <= end):
                ranged.append(image)
        images = ranged
    if limit is not None:
        images = images[:limit]
    return images


def debug_trace(result):
    """Print a structured trace of all pipeline stages to terminal."""
    s1 = result['stage1']
    s2 = result['stage2']
    final = result['final']
    print(f"  ── Stage 1 (SLM) ──")
    print(f"     task: {s1['task']}  confidence: {s1.get('task_confidence', 'N/A')}  source: {s1.get('source', s1.get('confidence_source', 'slm'))}")
    if s1.get('validation_error'):
        print(f"     ⚠ validation_error: {s1['validation_error']}")
    print(f"  ── Stage 2 (KG) ──")
    if s2.get('required_affordances'):
        print(f"     required affordances: {', '.join(s2['required_affordances'])}")
    for role, candidates in s2.get('roles', {}).items():
        names = [f"{c['concept']}({c['kg_relevance']:.2f})" for c in candidates]
        print(f"     role [{role}]: {', '.join(names) if names else '(empty)'}")
    vocab = s2.get('detector_vocabulary', {})
    if vocab:
        print(f"     detector queries ({len(vocab)}): {', '.join(sorted(vocab)[:15])}{'...' if len(vocab) > 15 else ''}")
    print(f"  ── Stage 3 (Florence) ──")
    raw = result['stage3'].get('raw_detections', [])
    print(f"     raw detections: {len(raw)}")
    for det in raw[:10]:
        bbox_str = ','.join(f'{x:.0f}' if isinstance(x, float) else str(x) for x in det.get('bbox', []))
        print(f"       [{bbox_str}] query={det.get('query')!r}  label={det.get('label')!r}  od_conf={det.get('od_confidence')}")
    if len(raw) > 10:
        print(f"       ... and {len(raw) - 10} more")
    reference = result['stage3'].get('reference_detections', [])
    print(f"     standard OD reference detections: {len(reference)}")
    for item in reference[:10]:
        print(f"       label={item.get('label')!r}  bbox={item.get('bbox')}")
    print(f"  ── Final ──")
    print(f"     status: {final['status']}")
    for role, values in final.get('roles', {}).items():
        sel = values.get('selected', [])
        status_icon = 'context' if values['status'] == 'context_only' else ('✓' if values['status'] == 'found' else '✗')
        print(f"     role [{role}] {status_icon} {values['status']}  (KG candidates: {', '.join(values.get('kg_candidates', []))})")
        for item in sel:
            score_str = f"{item['final_score']:.3f}" if item['final_score'] is not None else 'N/A'
            print(
                f"       → {item['concept']} score={score_str} ({item['score_basis']}) "
                f"queries={item.get('supporting_queries', [])} "
                f"alias_support={item.get('alias_support_count')} "
                f"reference_od={item.get('reference_od_supported')} "
                f"acceptance={item.get('acceptance_reason')} bbox={item['bbox']}"
            )
    rejected = final.get('rejected', [])
    if rejected:
        reasons = {}
        for r in rejected:
            reasons[r['reason']] = reasons.get(r['reason'], 0) + 1
        print(f"     rejected: {sum(reasons.values())} ({', '.join(f'{k}:{v}' for k, v in sorted(reasons.items()))})")
        for item in rejected:
            metrics = item.get('bbox_metrics', {})
            def ratio(name):
                value = metrics.get(name)
                return 'N/A' if value is None else f'{value:.4f}'
            query = item.get('detection', {}).get('query') or ','.join(item.get('supporting_queries', []))
            print(
                f"       REJECT query={query!r} bbox={item.get('bbox')} "
                f"area_ratio={ratio('area_ratio')} width_ratio={ratio('width_ratio')} "
                f"height_ratio={ratio('height_ratio')} rule={item.get('rejection_rule')}"
            )
    print(f"     timing: SLM={result['timing_ms']['slm']:.0f}ms  KG={result['timing_ms']['kg']:.0f}ms  OD={result['timing_ms']['od']:.0f}ms  rank={result['timing_ms']['ranking']:.0f}ms  total={result['timing_ms']['total']/1000:.1f}s")


def visualize(result, destination):
    from PIL import Image
    from anshdeep_basic_pipeline_runner import draw_boxes
    labels = []
    def fmt(v):
        return 'NA' if v is None else f'{v:.2f}'
    for role, values in result['final']['roles'].items():
        for item in values['selected']:
            label = f"{item['concept']} ({role})"
            if item['od_confidence'] is not None or item['final_score'] is not None:
                label += f" OD:{fmt(item['od_confidence'])} KG:{fmt(item['kg_relevance'])} Final:{fmt(item['final_score'])}"
            labels.append({'bbox': item['bbox'], 'label': label})
    with Image.open(result['image']) as source:
        draw_boxes(source.convert('RGB'), labels, destination)


def main():
    parser = argparse.ArgumentParser(description='SLM -> task KG -> Florence; deterministic ranking')
    parser.add_argument('--config', help='JSON overrides; relative resource paths use the project root')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('validate-kg')
    query = sub.add_parser('query-kg')
    query.add_argument('--task', required=True)
    run = sub.add_parser('run')
    inputs = run.add_mutually_exclusive_group()
    inputs.add_argument('--image', type=Path)
    inputs.add_argument('--input-dir', type=Path)
    tasks = run.add_mutually_exclusive_group(required=True)
    tasks.add_argument('--prompt', action='append', help='Repeat to run several prompts per image')
    tasks.add_argument('--task', action='append', help='Repeat canonical tasks; bypasses the SLM')
    run.add_argument('--limit', type=int, help='Maximum images to process (default: 1 without a range)')
    run.add_argument('--start', type=int, help='First numeric image filename to include, e.g. 20')
    run.add_argument('--end', type=int, help='Last numeric image filename to include, e.g. 30')
    run.add_argument('--run-name')
    run.add_argument('--no-visualization', action='store_true')
    run.add_argument('--debug-trace', action='store_true', help='Print all stages, candidates, detections and scores')
    suite = sub.add_parser('run-suite', help='Run canonical tasks from all recognized dataset folders')
    suite.add_argument('--input-dir', type=Path)
    suite.add_argument('--limit-per-task', type=int, default=1)
    suite.add_argument('--run-name')
    suite.add_argument('--no-visualization', action='store_true')
    suite.add_argument('--debug-trace', action='store_true')
    bench = sub.add_parser('benchmark')
    bench.add_argument('--run-dir', type=Path, required=True)
    bench.add_argument('--ground-truth', type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == 'validate-kg':
        kg = TaskGraph(config['kg_path'], config)
        report = kg.validation
        print(json.dumps(report, indent=2))
        if report['errors']:
            return 1
        missing = [t for t in kg.tasks if any(not c for c in kg.query_task(t)['roles'].values())]
        if missing:
            print('No active candidate path for tasks:', missing)
            return 1
        print('TaskAffordanceKG valid')
        return 0
    if args.command == 'query-kg':
        print(json.dumps(TaskGraph(config['kg_path'], config).query_task(args.task), indent=2))
        return 0
    if args.command == 'benchmark':
        from .evaluation import benchmark_folder
        report = benchmark_folder(args.run_dir, args.ground_truth)
        save_json(args.run_dir / 'benchmark.json', report)
        print(json.dumps(report, indent=2))
        return 0
    if args.command == 'run-suite':
        from .suite import run_suite
        return run_suite(args, config, save_json, visualize, debug_trace)
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    if args.start is not None and args.start < 0:
        parser.error('--start must be zero or positive')
    if args.end is not None and args.end < 0:
        parser.error('--end must be zero or positive')
    if args.start is not None and args.end is not None and args.start > args.end:
        parser.error('--start cannot be greater than --end')
    if args.image and (args.start is not None or args.end is not None):
        parser.error('--start and --end can only be used with --input-dir')
    if args.image:
        images = [args.image.resolve()]
    else:
        directory = args.input_dir or discover_samples(config)
        default_limit = None if args.start is not None or args.end is not None else 1
        images = select_images(directory, args.limit if args.limit is not None else default_limit, args.start, args.end)
    if not images:
        parser.error('No images found')
    name = args.run_name or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%f')
    if Path(name).name != name or name in ('.', '..'):
        parser.error('--run-name must be a simple folder name')
    run_dir = Path(config['results_path']) / name
    run_dir.mkdir(parents=True, exist_ok=False)
    prompts = args.task or args.prompt
    pipeline = ThreeStagePipeline(config)
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(), 'config': config, 'images': [str(p) for p in images], 'prompts': prompts, 'slm_bypassed': bool(args.task), 'hardware': platform.uname()._asdict(), 'status': 'running', 'notes': 'Model load and image export excluded from per-case inference timing; no prompt cache.'}
    import importlib.metadata
    manifest['versions'] = {p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'networkx', 'pillow')}
    save_json(run_dir / 'manifest.json', manifest)
    print('Loading models once...', flush=True)
    try:
        pipeline.prepare(use_slm=not args.task)
    except Exception as exc:
        manifest.update(status='model_load_failed', error=str(exc))
        save_json(run_dir / 'manifest.json', manifest)
        print(f'Model load failed: {exc}. Check local model paths in the config.')
        return 1
    errors, templates = 0, []
    for image in images:
        image_key = image.stem + '_' + hashlib.sha256(str(image).encode()).hexdigest()[:10]
        for index, prompt in enumerate(prompts):
            case_dir = run_dir / image_key / f'case_{index+1}'
            task = prompt if args.task else None
            print(f'{image.name}: {prompt}', flush=True)
            try:
                result = pipeline.run(prompt, image, task)
                result['requested_task'] = task
                save_json(case_dir / 'result.json', result)
                if config['visualize'] and not args.no_visualization:
                    visualize(result, case_dir / 'visualization.jpg')
                templates.append({'image': str(image), 'image_sha256': result['image_sha256'], 'prompt': prompt, 'task': task, 'relevant_objects': None, 'annotation_status': 'pending_manual_review'})
                if args.debug_trace:
                    debug_trace(result)
                print(f"  ── summary: task={result['stage1']['task']}; candidates={result['stage2']['candidate_concept_count']}; detected={len(result['final']['all_ranked_objects'])}; time={result['timing_ms']['total']/1000:.1f}s", flush=True)
            except Exception as exc:
                errors += 1
                save_json(case_dir / 'error.json', {'image': str(image), 'prompt': prompt, 'error': str(exc), 'type': type(exc).__name__})
                print(f'  FAILED: {exc}', flush=True)
    save_json(run_dir / 'annotation_template.json', templates)
    manifest.update(status='completed_with_errors' if errors else 'completed', model_load_ms=pipeline.load_ms, errors=errors)
    save_json(run_dir / 'manifest.json', manifest)
    from .evaluation import benchmark_folder
    save_json(run_dir / 'benchmark.json', benchmark_folder(run_dir))
    print(f'Results: {run_dir}', flush=True)
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
