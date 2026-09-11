"""Three inspectable stages; models are initialized at most once per instance."""
import hashlib
import time
from pathlib import Path

from .knowledge.affordance_graph import TaskGraph
from .knowledge.detector_vocab import detector_vocabulary
from .ranking import rank


class ThreeStagePipeline:
    def __init__(self, config, parser=None, detector=None):
        self.config = config
        start = time.perf_counter()
        self.kg = TaskGraph(config['kg_path'], config)
        self.kg_hash = hashlib.sha256(Path(config['kg_path']).read_bytes()).hexdigest()
        self.parser, self.detector = parser, detector
        self.load_ms = {'kg': (time.perf_counter()-start)*1000, 'slm': 0, 'od': 0}

    def prepare(self, use_slm=True):
        if use_slm and self.parser is None:
            from .parser import TaskParser
            start = time.perf_counter()
            self.parser = TaskParser(self.config, self.kg.tasks)
            self.load_ms['slm'] = (time.perf_counter()-start)*1000
        if self.detector is None:
            from .detector import FlorenceDetector
            start = time.perf_counter()
            self.detector = FlorenceDetector(self.config)
            self.load_ms['od'] = (time.perf_counter()-start)*1000

    def parse_prompt(self, prompt):
        return self.parser.parse(prompt)

    def retrieve_kg_candidates(self, task):
        return self.kg.query_task(task)

    def detect_objects(self, image_path, vocabulary):
        return self.detector.detect(image_path, list(vocabulary)) if vocabulary else []

    def detect_reference_objects(self, image_path, vocabulary):
        if not vocabulary or not hasattr(self.detector, 'detect_reference'):
            return []
        return self.detector.detect_reference(image_path)

    def rank_candidates(self, raw, vocabulary, roles, image_size, reference):
        return rank(raw, vocabulary, roles, self.config, image_size, reference)

    def run(self, prompt, image_path, task=None):
        image_path = Path(image_path).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        from PIL import Image
        with Image.open(image_path) as source:
            image_size = source.size
        if task is not None:
            task = self.kg.normalize_task(task)
        if task is not None and task not in self.kg.tasks and task != 'unknown':
            raise ValueError(f'Unsupported canonical task: {task}')
        self.prepare(use_slm=task is None)
        start = time.perf_counter()
        stage1 = self.parse_prompt(prompt) if task is None else {'raw_prompt': prompt, 'task': task, 'task_confidence': None, 'constraints': {}, 'source': 'canonical_task_bypass'}
        t1 = time.perf_counter()
        stage2 = self.retrieve_kg_candidates(stage1['task'])
        vocabulary = detector_vocabulary(stage2, self.config['use_detector_aliases'])
        stage2['detector_vocabulary'] = vocabulary
        stage2['detector_query_count'] = len(vocabulary)
        t2 = time.perf_counter()
        raw = self.detect_objects(image_path, vocabulary)
        reference = self.detect_reference_objects(image_path, vocabulary)
        t3 = time.perf_counter()
        final = self.rank_candidates(raw, vocabulary, stage2['roles'], image_size, reference)
        t4 = time.perf_counter()
        return {'schema_version': 1, 'pipeline': 'three_stage', 'prompt': prompt, 'image': str(image_path), 'image_size': list(image_size), 'image_sha256': hashlib.sha256(image_path.read_bytes()).hexdigest(), 'kg_sha256': self.kg_hash, 'config': self.config, 'stage1': stage1, 'stage2': stage2, 'stage3': {'raw_detections': raw, 'reference_detections': reference, 'detections': final['all_ranked_objects'], 'confidence_note': 'Florence has no object confidence; null is not zero. Standard OD is optional supporting evidence, not a hard veto.'}, 'final': final, 'model_load_ms': dict(self.load_ms), 'timing_ms': {'slm': (t1-start)*1000, 'kg': (t2-t1)*1000, 'od': (t3-t2)*1000, 'ranking': (t4-t3)*1000, 'total': (t4-start)*1000}}
