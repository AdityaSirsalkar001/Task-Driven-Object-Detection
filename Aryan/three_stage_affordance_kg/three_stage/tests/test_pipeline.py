"""Offline tests for the nine-task affordance pipeline."""
import tempfile
import unittest
from pathlib import Path

from three_stage.config import load_config
from three_stage.evaluation import evaluate
from three_stage.knowledge.affordance_graph import TaskGraph, load_yaml, validate
from three_stage.knowledge.detector_vocab import detector_vocabulary
from three_stage.parser import validate_response
from three_stage.pipeline import ThreeStagePipeline
from three_stage.ranking import rank


EXPECTED = {
    'dig_hole': ('instrument', {'shovel', 'spade', 'garden_trowel', 'pickaxe', 'hoe'}),
    'extinguish_fire': ('instrument', {'fire_extinguisher', 'fire_blanket'}),
    'open_parcel': ('instrument', {'box_cutter', 'scissors', 'letter_opener', 'knife'}),
    'place_flowers': ('destination', {'vase', 'flower_pot', 'jar'}),
    'pour_sugar': ('instrument', {'sugar_dispenser', 'sugar_bowl', 'jar', 'spoon', 'measuring_spoon'}),
    'sit_comfortably': ('support', {'armchair', 'sofa', 'chair', 'bench', 'stool'}),
    'smear_butter': ('instrument', {'butter_knife', 'spreader', 'small_spatula', 'knife'}),
    'step_on': ('support', {'step_stool', 'ladder', 'stairs', 'platform', 'stool', 'chair'}),
    'wine': ('container', {'wine_glass', 'wine_bottle', 'decanter'}),
}


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.kg = TaskGraph(self.config['kg_path'], self.config)

    def test_kg_loads_and_has_nine_tasks(self):
        self.assertFalse(self.kg.validation['errors'])
        self.assertEqual(set(self.kg.tasks), set(EXPECTED))

    def test_every_task_returns_expected_candidates(self):
        for task, (role, expected) in EXPECTED.items():
            with self.subTest(task=task):
                result = self.kg.query_task(task)
                actual = {x['concept'] for x in result['roles'][role]}
                self.assertTrue(expected.issubset(actual))
                all_candidates = {x['concept'] for values in result['roles'].values() for x in values}
                self.assertGreaterEqual(len(all_candidates), 10)
                self.assertLessEqual(len(all_candidates), 20)
                self.assertLess(result['active_kg_nodes'], result['total_kg_nodes'])

    def test_spelling_alias_normalizes(self):
        self.assertEqual(self.kg.normalize_task('extingh_fire'), 'extinguish_fire')
        self.assertEqual(self.kg.normalize_task('put out the fire'), 'extinguish_fire')

    def test_task_aliases_are_graph_nodes(self):
        synonym_edges = [
            data for _, _, data in self.kg.graph.edges(data=True)
            if data['relation'] == 'SYNONYM_OF'
        ]
        self.assertTrue(synonym_edges)
        self.assertTrue(any(edge['target'] == 'task:extinguish_fire' for edge in synonym_edges))

    def test_alias_maps_to_canonical_concept(self):
        vocab = detector_vocabulary(self.kg.query_task('open_parcel'))
        self.assertEqual(vocab['utility knife'][0]['concept'], 'box_cutter')
        self.assertEqual(vocab['letter opener'][0]['concept'], 'letter_opener')

    def test_open_parcel_avoids_ambiguous_queries(self):
        kg = self.kg.query_task('open_parcel')
        vocab = detector_vocabulary(kg)
        self.assertTrue({'box cutter', 'utility knife', 'scissors', 'letter opener', 'knife'}.issubset(vocab))
        self.assertTrue({'box', 'cutter', 'package'}.isdisjoint(vocab))
        self.assertFalse(kg['roles']['target'][0]['output'])

    def test_all_weights_are_valid(self):
        for task in self.kg.tasks:
            for values in self.kg.query_task(task)['roles'].values():
                for candidate in values:
                    self.assertTrue(0 <= candidate['kg_relevance'] <= 1)

    def test_unknown_task_is_safe(self):
        self.assertEqual(self.kg.query_task('unsupported task')['roles'], {})

    def test_malformed_slm_output_is_safe(self):
        for raw in ('not json', '[]', '{"task":"made_up","confidence":0.9}', '{"task":"open_parcel","confidence":4}'):
            self.assertEqual(validate_response(raw, 'test', self.kg.tasks)['task'], 'unknown')

    def test_slm_alias_is_validated(self):
        result = validate_response('{"task":"extingh_fire","task_confidence":0.9}', 'test', self.kg.tasks)
        self.assertEqual(result['task'], 'extinguish_fire')

    def test_ranking_formula(self):
        kg = self.kg.query_task('open_parcel')
        raw = [{'label': 'box cutter', 'query': 'box cutter', 'bbox': [0, 0, 20, 5], 'od_confidence': .8}]
        result = rank(raw, detector_vocabulary(kg), kg['roles'], self.config, (100, 100))
        self.assertAlmostEqual(result['roles']['instrument']['selected'][0]['final_score'], .86)

    def test_empty_detections_are_not_found(self):
        kg = self.kg.query_task('open_parcel')
        result = rank([], detector_vocabulary(kg), kg['roles'], self.config)
        self.assertEqual(result['roles']['instrument']['status'], 'not_found')
        self.assertEqual(result['roles']['target']['status'], 'context_only')
        self.assertEqual(result['status'], 'no_suitable_object_detected')

    def test_cross_query_clustering_merges_aliases(self):
        kg = self.kg.query_task('open_parcel')
        raw = [
            {'label': 'box cutter', 'query': 'box cutter', 'bbox': [0, 0, 20, 5], 'od_confidence': None},
            {'label': 'utility knife', 'query': 'utility knife', 'bbox': [0, 0, 20, 5], 'od_confidence': None},
        ]
        reference = [{'label': 'box cutter', 'bbox': [0, 0, 20, 5]}]
        items = rank(raw, detector_vocabulary(kg), kg['roles'], self.config, (100, 100), reference)['roles']['instrument']['detections']
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]['supporting_queries'], ['box cutter', 'utility knife'])
        self.assertEqual(items[0]['concept'], 'box_cutter')

    def test_validator_summary(self):
        report = validate(load_yaml(self.config['kg_path']))
        self.assertEqual(report['tasks'], 9)
        self.assertGreaterEqual(report['objects'], 100)
        self.assertGreaterEqual(report['affordances'], 30)
        self.assertFalse(report['errors'])

    def test_candidate_metadata_is_complete(self):
        for task in self.kg.tasks:
            result = self.kg.query_task(task)
            for role, candidates in result['roles'].items():
                for candidate in candidates:
                    source = self.kg.data['objects'][candidate['concept']]
                    self.assertEqual(candidate['role'], role)
                    self.assertTrue(0 <= candidate['kg_relevance'] <= 1)
                    self.assertIn('conditional', source)
                    self.assertIn('preferred', source)
                    self.assertTrue(source['detector_aliases'])

    def test_mock_pipeline_without_models(self):
        class Parser:
            def parse(self, prompt):
                return {'raw_prompt': prompt, 'task': 'open_parcel', 'task_confidence': .9, 'constraints': {}}

        class Detector:
            def detect(self, image, vocabulary):
                return [{'query': 'scissors', 'label': 'scissors', 'bbox': [0, 0, 20, 10], 'od_confidence': .9}]

        pipeline = ThreeStagePipeline(self.config, Parser(), Detector())
        with tempfile.TemporaryDirectory() as folder:
            image = Path(folder) / 'mock.jpg'
            from PIL import Image
            Image.new('RGB', (100, 100)).save(image)
            result = pipeline.run('open this', image)
        self.assertEqual(result['stage1']['task'], 'open_parcel')
        self.assertEqual(result['final']['roles']['instrument']['selected'][0]['concept'], 'scissors')
        self.assertIsNone(evaluate([result])['metrics']['precision'])

    def test_duplicate_yaml_keys_rejected(self):
        content = "objects:\n  obj1: {}\n  obj1: {}\n"
        with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as f:
            f.write(content)
            temp_path = f.name
        try:
            with self.assertRaisesRegex(ValueError, 'Duplicate YAML key'):
                load_yaml(temp_path)
        finally:
            Path(temp_path).unlink()

    def _make_minimal_yaml(self, **overrides):
        import yaml
        base = {
            'tasks': {'test_task': {'roles': {'target': {'candidates': {'obj1': 1.0}}}}},
            'objects': {'obj1': {'is_a': ['cat1'], 'affordances': ['aff1'], 'detector_aliases': ['obj1']}},
            'affordances': {'aff1': {}},
            'categories': {'cat1': {}},
            'roles': ['target', 'instrument']
        }
        for k, v in overrides.items():
            base[k] = v
        with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as f:
            yaml.dump(base, f)
            return f.name

    def test_dangling_object_reference_fails_validation(self):
        path = self._make_minimal_yaml(tasks={'test_task': {'roles': {'target': {'candidates': {'missing_obj': 1.0}}}}})
        try:
            report = validate(load_yaml(path))
            self.assertTrue(any('Unknown object missing_obj' in e for e in report['errors']))
        finally:
            Path(path).unlink()

    def test_invalid_category_on_object_fails(self):
        path = self._make_minimal_yaml(objects={'obj1': {'is_a': ['nonexistent_category'], 'detector_aliases': ['obj1']}})
        try:
            report = validate(load_yaml(path))
            self.assertTrue(any('Unknown category nonexistent_category' in e for e in report['errors']))
        finally:
            Path(path).unlink()

    def test_invalid_affordance_on_object_fails(self):
        path = self._make_minimal_yaml(objects={'obj1': {'affordances': ['fake_affordance'], 'detector_aliases': ['obj1']}})
        try:
            report = validate(load_yaml(path))
            self.assertTrue(any('Invalid affordance fake_affordance' in e for e in report['errors']))
        finally:
            Path(path).unlink()

    def test_invalid_role_name_fails(self):
        path = self._make_minimal_yaml(tasks={'test_task': {'roles': {'weapon': {'candidates': {'obj1': 1.0}}}}})
        try:
            report = validate(load_yaml(path))
            self.assertTrue(any('Invalid role weapon' in e for e in report['errors']))
        finally:
            Path(path).unlink()

    def test_unknown_relation_type_fails(self):
        path = self._make_minimal_yaml(relations={'MADE_UP_RELATION': 'invalid'})
        try:
            report = validate(load_yaml(path))
            self.assertTrue(any('Unknown relation types' in e for e in report['errors']))
        finally:
            Path(path).unlink()

    def test_missing_detector_aliases_fails(self):
        path = self._make_minimal_yaml(objects={'obj1': {'detector_aliases': []}})
        try:
            report = validate(load_yaml(path))
            self.assertTrue(any('Missing detector aliases: obj1' in e for e in report['errors']))
        finally:
            Path(path).unlink()

    def test_relevance_prior_out_of_range_fails(self):
        path = self._make_minimal_yaml(tasks={'test_task': {'roles': {'target': {'candidates': {'obj1': 1.5}}}}})
        try:
            report = validate(load_yaml(path))
            self.assertTrue(any('Relevance outside [0,1]' in e for e in report['errors']))
        finally:
            Path(path).unlink()

    def test_circular_category_is_a_fails(self):
        path = self._make_minimal_yaml(categories={'cat1': {'is_a': ['cat2']}, 'cat2': {'is_a': ['cat1']}})
        try:
            report = validate(load_yaml(path))
            self.assertTrue(any('Invalid circular IS_A path' in e for e in report['errors']))
        finally:
            Path(path).unlink()

    def test_object_missing_required_affordance_gate_fails(self):
        path = self._make_minimal_yaml(
            tasks={'test_task': {'roles': {'target': {'required_affordance': 'aff1', 'candidates': {'obj1': 1.0}}}}},
            objects={'obj1': {'detector_aliases': ['obj1']}}
        )
        try:
            report = validate(load_yaml(path))
            self.assertTrue(any('lacks required affordance aff1' in e for e in report['errors']))
        finally:
            Path(path).unlink()

    def test_use_affordances_false_returns_candidates(self):
        config = self.config.copy()
        config['use_affordances'] = False
        kg = TaskGraph(self.config['kg_path'], config)
        result = kg.query_task('open_parcel')
        self.assertIn('box_cutter', {x['concept'] for x in result['roles']['instrument']})

    def test_use_hierarchy_false_changes_subgraph(self):
        config_true = self.config.copy()
        config_true['use_hierarchy'] = True
        kg_true = TaskGraph(self.config['kg_path'], config_true)
        result_true = kg_true.query_task('open_parcel')

        config_false = self.config.copy()
        config_false['use_hierarchy'] = False
        kg_false = TaskGraph(self.config['kg_path'], config_false)
        result_false = kg_false.query_task('open_parcel')

        self.assertLess(result_false['active_kg_nodes'], result_true['active_kg_nodes'])

    def test_optional_target_absence_not_failure(self):
        kg = self.kg.query_task('open_parcel')
        raw = [{'label': 'scissors', 'query': 'scissors', 'bbox': [0, 0, 20, 10], 'od_confidence': 0.9}]
        result = rank(raw, detector_vocabulary(kg), kg['roles'], self.config, (100, 100))
        self.assertEqual(result['status'], 'success')

    def test_required_role_absence_is_failure(self):
        kg = self.kg.query_task('open_parcel')
        raw = [{'label': 'parcel', 'query': 'parcel', 'bbox': [0, 0, 10, 10], 'od_confidence': 0.9}]
        result = rank(raw, detector_vocabulary(kg), kg['roles'], self.config)
        self.assertEqual(result['status'], 'no_suitable_object_detected')

    def test_all_optional_task_succeeds_when_candidate_is_detected(self):
        kg = self.kg.query_task('wine')
        raw = [{'label': 'wine glass', 'query': 'wine glass', 'bbox': [0, 0, 10, 10], 'od_confidence': 0.9}]
        result = rank(raw, detector_vocabulary(kg), kg['roles'], self.config, (100, 100))
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['verification_status'], 'pending_manual_review')

    def test_standard_od_is_optional_support(self):
        kg = self.kg.query_task('open_parcel')
        raw = [
            {'query': 'letter opener', 'label': 'letter opener', 'bbox': [0, 0, 639, 479], 'od_confidence': None},
            {'query': 'scissors', 'label': 'scissors', 'bbox': [486, 326, 575, 451], 'od_confidence': None},
        ]
        result = rank(raw, detector_vocabulary(kg), kg['roles'], self.config, (640, 480), [])
        self.assertEqual(result['status'], 'success')
        self.assertEqual([item['concept'] for item in result['all_ranked_objects']], ['scissors'])
        scissors = result['roles']['instrument']['selected'][0]
        self.assertIsNone(scissors['final_score'])
        self.assertEqual(scissors['score_basis'], 'unscored_text_conditioned_florence')
        self.assertFalse(scissors['reference_od_supported'])
        self.assertEqual(scissors['acceptance_reason'], 'canonical_query_detected')
        self.assertEqual(result['roles']['target']['status'], 'context_only')
        self.assertEqual(result['roles']['target']['selected'], [])
        reasons = {item['reason'] for item in result['rejected']}
        self.assertIn('near_full_image_bbox', reasons)
        self.assertNotIn('no_reference_detection_support', reasons)

    def test_large_cross_query_scissors_cluster_survives(self):
        kg = self.kg.query_task('open_parcel')
        bbox = [144, 34, 347, 546]
        raw = [
            {'query': 'box cutter', 'label': 'box cutter', 'bbox': bbox, 'od_confidence': None},
            {'query': 'scissors', 'label': 'scissors', 'bbox': bbox, 'od_confidence': None},
            {'query': 'utility knife', 'label': 'utility knife', 'bbox': bbox, 'od_confidence': None},
        ]
        reference = [{'label': 'scissors', 'bbox': bbox}]
        result = rank(raw, detector_vocabulary(kg), kg['roles'], self.config, (427, 640), reference)
        self.assertEqual(result['status'], 'success')
        item = next(item for item in result['all_ranked_objects'] if item['concept'] == 'scissors')
        self.assertEqual(item['concept'], 'scissors')
        self.assertEqual(item['supporting_queries'], ['scissors'])
        self.assertGreater(item['visual_support_iou'], 0.99)
        self.assertIsNone(item['final_score'])

    def test_no_plausible_instrument_returns_not_found(self):
        kg = self.kg.query_task('open_parcel')
        raw = [{'query': 'utility knife', 'label': 'utility knife', 'bbox': [410, 211, 639, 479], 'od_confidence': None}]
        result = rank(raw, detector_vocabulary(kg), kg['roles'], self.config, (640, 480))
        self.assertEqual(result['status'], 'no_suitable_object_detected')
        self.assertEqual(result['rejected'][0]['reason'], 'insufficient_text_conditioned_support')

    def test_bucket_canonical_query_needs_no_reference_od(self):
        kg = self.kg.query_task('extinguish_fire')
        raw = [{'query': 'bucket', 'label': 'bucket', 'bbox': [20, 20, 100, 120], 'od_confidence': None}]
        result = rank(raw, detector_vocabulary(kg), kg['roles'], self.config, (200, 200), [])
        bucket = next(item for item in result['all_ranked_objects'] if item['concept'] == 'bucket')
        self.assertFalse(bucket['reference_od_supported'])
        self.assertEqual(bucket['alias_support_count'], 1)
        self.assertEqual(bucket['acceptance_reason'], 'canonical_query_detected')

    def test_bucket_alias_family_votes_and_canonicalizes(self):
        kg = self.kg.query_task('extinguish_fire')
        raw = [
            {'query': 'metal pail', 'label': 'metal pail', 'bbox': [20, 20, 120, 120], 'od_confidence': None},
            {'query': 'water bucket', 'label': 'water bucket', 'bbox': [35, 20, 135, 120], 'od_confidence': None},
        ]
        result = rank(raw, detector_vocabulary(kg), kg['roles'], self.config, (200, 200), [])
        self.assertEqual(len(result['all_ranked_objects']), 1)
        bucket = result['all_ranked_objects'][0]
        self.assertEqual(bucket['concept'], 'bucket')
        self.assertEqual(bucket['alias_support_count'], 2)
        self.assertEqual(bucket['acceptance_reason'], 'same_family_alias_agreement')

    def test_unrelated_concepts_cannot_vote_together(self):
        kg = self.kg.query_task('extinguish_fire')
        bbox = [20, 20, 100, 120]
        raw = [
            {'query': 'metal pail', 'label': 'metal pail', 'bbox': bbox, 'od_confidence': None},
            {'query': 'fire suppressant canister', 'label': 'fire suppressant canister', 'bbox': bbox, 'od_confidence': None},
        ]
        result = rank(raw, detector_vocabulary(kg), kg['roles'], self.config, (200, 200), [])
        self.assertEqual(result['status'], 'no_suitable_object_detected')
        self.assertEqual(len(result['rejected']), 2)
        self.assertTrue(all(item['reason'] == 'insufficient_text_conditioned_support' for item in result['rejected']))


if __name__ == '__main__':
    unittest.main()
