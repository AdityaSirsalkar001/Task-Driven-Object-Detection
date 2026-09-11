"""Concept/role metrics with explicit, manually supplied ground truth."""
import json
from pathlib import Path
from statistics import mean

QUALITY = ['task_parsing_accuracy', 'precision', 'recall', 'f1', 'recall_at_1', 'recall_at_3', 'end_to_end_success', 'false_positives_per_image']


def evaluate(results, annotations=None):
    """Evaluate sets of (role, concept), not localization accuracy or instances."""
    annotations = annotations or []
    keys = [(a['image_sha256'], a['task'], a.get('prompt')) for a in annotations]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate annotation keys')
    measurements = {key: [] for key in QUALITY}
    count = 0
    for result in results:
        matches = [a for a in annotations if a.get('image_sha256') == result['image_sha256'] and a.get('task') == result.get('requested_task', result['stage1']['task']) and (a.get('prompt') is None or a['prompt'] == result['prompt'])]
        # Prompt-only experiments must match a prompt-specific label, even when
        # parsing is wrong; never select ground truth using a predicted task.
        if result.get('requested_task') is None:
            matches = [a for a in annotations if a.get('image_sha256') == result['image_sha256'] and a.get('prompt') == result['prompt']]
        if len(matches) > 1:
            raise ValueError('Ambiguous ground truth for a result')
        if not matches:
            continue
        annotation = matches[0]
        if not annotation.get('task'):
            continue
        if result['stage1'].get('source') != 'canonical_task_bypass':
            measurements['task_parsing_accuracy'].append(int(annotation['task'] == result['stage1']['task']))
        if annotation.get('relevant_objects') is None:
            continue
        count += 1
        gt = {(o['role'], o['concept']) for o in annotation['relevant_objects']}
        predicted = {(r, o['concept']) for r, values in result['final']['roles'].items() for o in values['selected']}
        tp = len(predicted & gt)
        precision = tp/len(predicted) if predicted else (1.0 if not gt else 0.0)
        recall = tp/len(gt) if gt else (1.0 if not predicted else 0.0)
        measurements['precision'].append(precision)
        measurements['recall'].append(recall)
        measurements['f1'].append(2*precision*recall/(precision+recall) if precision+recall else 0)
        measurements['end_to_end_success'].append(int(predicted == gt and result['stage1']['task'] == annotation['task']))
        fp = len(predicted - gt)
        measurements['false_positives_per_image'].append(fp)
        for k in (1, 3):
            top = set()
            for role, values in result['final']['roles'].items():
                unique = list(dict.fromkeys(o['concept'] for o in values['detections']))
                top.update((role, concept) for concept in unique[:k])
            measurements[f'recall_at_{k}'].append(len(top & gt)/len(gt) if gt else float(not top))
    report = {'result_count': len(results), 'annotated_object_cases': count, 'metrics': {k: mean(v) if v else None for k, v in measurements.items()}, 'metric_sample_counts': {k: len(v) for k, v in measurements.items()}, 'definition': 'Macro per-case concept/role set precision, recall, F1; Recall@K uses unique concepts within each role; success is exact concept/role set AND correct task. No box IoU accuracy without box annotations.', 'quality_status': 'manual_ground_truth_used' if count else 'unavailable_without_ground_truth', 'mean_timing_ms': {}, 'mean_graph_stats': {}}
    if results:
        report['mean_timing_ms'] = {k: mean(r['timing_ms'][k] for r in results) for k in ('slm', 'kg', 'od', 'ranking', 'total')}
        report['mean_graph_stats'] = {k: mean(r['stage2'][k] for r in results) for k in ('candidate_concept_count', 'detector_query_count', 'total_kg_nodes', 'active_kg_nodes', 'active_subgraph_reduction_ratio')}
    return report


def benchmark_folder(run_dir, ground_truth=None):
    results = [json.loads(p.read_text()) for p in sorted(Path(run_dir).rglob('result.json'))]
    annotations = json.loads(Path(ground_truth).read_text()) if ground_truth else []
    report = evaluate(results, annotations)
    errors = [str(p.relative_to(run_dir)) for p in Path(run_dir).rglob('error.json')]
    report['error_cases'] = errors
    report['failed_case_count'] = len(errors)
    report['comparison_ready'] = not errors and report['annotated_object_cases'] == len(results) and bool(results)
    report['failure_note'] = 'Quality metrics cover saved successful cases only; inspect errors before comparison.'
    return report
