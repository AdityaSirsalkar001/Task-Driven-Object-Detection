"""Central configuration; paths resolve from the existing project root."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None):
    config = json.loads((ROOT / 'three_stage/config.json').read_text())
    if path:
        override = json.loads(Path(path).read_text())
        unknown = set(override) - set(config)
        if unknown:
            raise ValueError(f'Unknown configuration keys: {sorted(unknown)}')
        config.update(override)
    for key in ('alpha', 'beta', 'od_confidence_threshold', 'near_full_image_area_ratio', 'cross_query_cluster_iou'):
        v = config[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1:
            raise ValueError(f'{key} must be between 0 and 1')
    if not math.isclose(config['alpha'] + config['beta'], 1):
        raise ValueError('alpha + beta must equal 1')
    for key in ('top_k', 'threads', 'max_new_tokens'):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if config['missing_confidence_policy'] not in ('sanity_filter', 'reject'):
        raise ValueError('missing_confidence_policy must be sanity_filter or reject')
    for key in ('kg_path', 'slm_path', 'od_path', 'results_path', 'sample_image_path'):
        if config[key]:
            config[key] = str((ROOT / config[key]).resolve())
    return config


def discover_samples(config):
    if config['sample_image_path']:
        path = Path(config['sample_image_path'])
    else:
        candidates = [p for p in ROOT.iterdir() if p.is_dir() and p.name.lower().startswith('sample')]
        if len(candidates) != 1:
            raise ValueError(f'Specify --input-dir; discovered sample folders: {candidates}')
        path = candidates[0]
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path
