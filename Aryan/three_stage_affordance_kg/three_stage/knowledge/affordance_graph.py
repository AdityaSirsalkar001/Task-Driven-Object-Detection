"""Typed task-affordance graph loaded from the human-editable YAML seed."""
from collections import Counter
from pathlib import Path
import math
import re

import networkx as nx
import yaml


RELATIONS = {
    'REQUIRES_ROLE', 'REQUIRES_AFFORDANCE', 'HAS_AFFORDANCE',
    'VALID_FOR_ROLE', 'IS_A', 'DETECTOR_ALIAS', 'SYNONYM_OF',
    'HAS_ATTRIBUTE', 'INCOMPATIBLE_WITH', 'OPTIONAL_CONTEXT',
}
ROLE_NAMES = {'target', 'instrument', 'material', 'destination', 'support', 'container', 'consumable', 'context', 'object'}


class UniqueKeyLoader(yaml.SafeLoader):
    """YAML loader that rejects duplicate mapping keys instead of losing data."""


def _mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f'Duplicate YAML key: {key}')
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def normalize(value):
    return re.sub(r'[^a-z0-9]+', '_', str(value).lower()).strip('_')


def load_yaml(path):
    data = yaml.load(Path(path).read_text(), Loader=UniqueKeyLoader)
    if not isinstance(data, dict):
        raise ValueError('KG root must be a mapping')
    return data


def validate(data):
    """Validate references, roles, aliases, priors, and useful task paths."""
    errors, warnings = [], []
    required = ('tasks', 'objects', 'affordances', 'categories', 'roles')
    if any(not isinstance(data.get(k), (dict, list)) for k in required):
        return {'errors': ['KG is missing tasks, objects, affordances, categories, or roles'], 'warnings': []}
    tasks, objects = data['tasks'], data['objects']
    affordances, categories = data['affordances'], data['categories']
    declared_relations = set(data.get('relations', {}))
    unknown_relations = declared_relations - RELATIONS
    if unknown_relations:
        errors.append(f'Unknown relation types: {sorted(unknown_relations)}')
    declared_roles = set(data['roles'])
    if not declared_roles.issubset(ROLE_NAMES):
        errors.append(f'Invalid role names: {sorted(declared_roles - ROLE_NAMES)}')
    aliases = {}
    for name, obj in objects.items():
        if not isinstance(obj, dict):
            errors.append(f'Invalid object: {name}')
            continue
        if not obj.get('detector_aliases'):
            errors.append(f'Missing detector aliases: {name}')
        for category in obj.get('is_a', []):
            if category not in categories:
                errors.append(f'Unknown category {category} on object {name}')
        for affordance in obj.get('affordances', []):
            if affordance not in affordances:
                errors.append(f'Invalid affordance {affordance} on object {name}')
        for alias in obj.get('detector_aliases', []):
            aliases.setdefault(normalize(alias), set()).add(name)
    for alias, owners in aliases.items():
        if len(owners) > 1:
            warnings.append(f'Detector alias {alias} maps to {sorted(owners)}')
    task_aliases = {}
    for task, spec in tasks.items():
        if not isinstance(spec, dict) or not spec.get('roles'):
            errors.append(f'Task without useful candidate path: {task}')
            continue
        for alias in [task, *spec.get('aliases', [])]:
            key = normalize(alias)
            if key in task_aliases and task_aliases[key] != task:
                errors.append(f'Task synonym collision: {alias}')
            task_aliases[key] = task
        useful = False
        for role, role_spec in spec['roles'].items():
            if role not in declared_roles:
                errors.append(f'Invalid role {role} on task {task}')
                continue
            groups = list(role_spec.get('candidate_groups', []))
            if role_spec.get('candidates'):
                groups.append({'affordance': role_spec.get('required_affordance'), 'candidates': role_spec['candidates']})
            required_affordance = role_spec.get('required_affordance')
            if required_affordance and required_affordance not in affordances:
                errors.append(f'Invalid affordance {required_affordance} on {task}/{role}')
            for group in groups:
                gate = group.get('affordance')
                if gate and gate not in affordances:
                    errors.append(f'Invalid affordance {gate} on {task}/{role}')
                for obj, prior in group.get('candidates', {}).items():
                    useful = True
                    if obj not in objects:
                        errors.append(f'Unknown object {obj} on {task}/{role}')
                        continue
                    if isinstance(prior, bool) or not isinstance(prior, (int, float)) or not math.isfinite(prior) or not 0 <= prior <= 1:
                        errors.append(f'Relevance outside [0,1] for {task}/{role}/{obj}')
                    if gate and gate not in objects[obj].get('affordances', []):
                        errors.append(f'Object {obj} lacks required affordance {gate} for {task}/{role}')
        if not useful:
            errors.append(f'Task without useful candidate path: {task}')
    hierarchy = nx.DiGraph()
    for name, spec in categories.items():
        for parent in (spec or {}).get('is_a', []):
            if parent not in categories:
                errors.append(f'Unknown parent category {parent} on {name}')
            hierarchy.add_edge(name, parent)
    if not nx.is_directed_acyclic_graph(hierarchy):
        errors.append('Invalid circular IS_A path')
    return {
        'errors': errors, 'warnings': warnings, 'tasks': len(tasks),
        'objects': len(objects), 'affordances': len(affordances),
        'categories': len(categories), 'aliases': sum(len(x.get('detector_aliases', [])) for x in objects.values()),
        'relation_types': len(declared_relations),
    }


class TaskGraph:
    """Build once and return a small task-conditioned active subgraph."""
    def __init__(self, path, config):
        self.data = load_yaml(path)
        self.validation = validate(self.data)
        if self.validation['errors']:
            raise ValueError('\n'.join(self.validation['errors']))
        self.config = config
        self.graph = nx.MultiDiGraph()
        self.tasks = {}
        self.task_aliases = {}
        self._build()
        self.validation['graph_nodes'] = len(self.graph)
        self.validation['graph_edges'] = self.graph.number_of_edges()

    def _node(self, node_id, node_type, name, **properties):
        self.graph.add_node(node_id, id=node_id, type=node_type, name=name, **properties)

    def _edge(self, source, target, relation, **properties):
        self.graph.add_edge(source, target, key=relation, source=source, target=target, relation=relation, **properties)

    def _build(self):
        data = self.data
        for name, spec in data['affordances'].items():
            self._node(f'affordance:{name}', 'Affordance', name, **(spec or {}))
        for name, spec in data['categories'].items():
            self._node(f'category:{name}', 'ObjectCategory', name)
            for affordance in (spec or {}).get('affordances', []):
                self._edge(f'category:{name}', f'affordance:{affordance}', 'HAS_AFFORDANCE')
        for name, spec in data['categories'].items():
            for parent in (spec or {}).get('is_a', []):
                self._edge(f'category:{name}', f'category:{parent}', 'IS_A')
        for name, spec in data['objects'].items():
            oid = f'object:{name}'
            self._node(oid, 'ObjectClass', name)
            for category in spec.get('is_a', []):
                self._edge(oid, f'category:{category}', 'IS_A')
            for affordance in spec.get('affordances', []):
                self._edge(oid, f'affordance:{affordance}', 'HAS_AFFORDANCE')
            for index, attribute in enumerate(spec.get('attributes', [])):
                aid = f'attribute:{normalize(attribute)}'
                if aid not in self.graph:
                    self._node(aid, 'Attribute', normalize(attribute))
                self._edge(oid, aid, 'HAS_ATTRIBUTE')
            for index, alias in enumerate(spec.get('detector_aliases', [])):
                aid = f'alias:{name}:{index}'
                self._node(aid, 'DetectorAlias', alias, label=alias, detector='florence')
                self._edge(oid, aid, 'DETECTOR_ALIAS')
        for task, spec in data['tasks'].items():
            tid = f'task:{task}'
            self._node(tid, 'Task', task, aliases=spec.get('aliases', []))
            self.tasks[task] = self.graph.nodes[tid]
            for index, alias in enumerate(spec.get('aliases', [])):
                alias_id = f'task_alias:{task}:{index}'
                self._node(alias_id, 'Alias', alias)
                self._edge(alias_id, tid, 'SYNONYM_OF')
            for alias in [task, *spec.get('aliases', [])]:
                self.task_aliases[normalize(alias)] = task
            for role, role_spec in spec['roles'].items():
                rid = f'role:{task}:{role}'
                self._node(rid, 'Role', role, required=role_spec.get('required', False), output=role_spec.get('output', True), task=task)
                self._edge(tid, rid, 'REQUIRES_ROLE')
                groups = list(role_spec.get('candidate_groups', []))
                if role_spec.get('candidates'):
                    groups.append({'affordance': role_spec.get('required_affordance'), 'candidates': role_spec['candidates']})
                for group in groups:
                    affordance = group.get('affordance')
                    if affordance:
                        self._edge(rid, f'affordance:{affordance}', 'REQUIRES_AFFORDANCE')
                    for obj, prior in group.get('candidates', {}).items():
                        self._edge(f'object:{obj}', rid, 'VALID_FOR_ROLE', relevance_prior=float(prior), affordance=affordance)

    def normalize_task(self, task):
        return self.task_aliases.get(normalize(task), 'unknown')

    def query_task(self, task):
        task = self.normalize_task(task)
        result = {'task': task, 'roles': {}, 'affordances': [], 'required_affordances': [], 'active_subgraph': {'nodes': [], 'edges': []}, 'total_kg_nodes': len(self.graph)}
        visited = set()
        if task == 'unknown':
            return self._finish(result, visited)
        tid = f'task:{task}'
        visited.add(tid)
        required_affordance_ids = set()
        for _, rid, relation in sorted(self.graph.out_edges(tid, keys=True)):
            if relation != 'REQUIRES_ROLE':
                continue
            visited.add(rid)
            role_node = self.graph.nodes[rid]
            gates = {target for _, target, rel in self.graph.out_edges(rid, keys=True) if rel == 'REQUIRES_AFFORDANCE'}
            visited.update(gates)
            required_affordance_ids.update(gates)
            candidates = []
            for oid, _, relation, edge in self.graph.in_edges(rid, keys=True, data=True):
                if relation != 'VALID_FOR_ROLE':
                    continue
                visited.add(oid)
                gate = edge.get('affordance')
                object_affordances = {target for _, target, rel in self.graph.out_edges(oid, keys=True) if rel == 'HAS_AFFORDANCE'}
                if self.config['use_affordances'] and gate and f'affordance:{gate}' not in object_affordances:
                    continue
                if self.config['use_affordances']:
                    visited.update(object_affordances)
                if self.config['use_hierarchy']:
                    visited.update(target for _, target, rel in self.graph.out_edges(oid, keys=True) if rel == 'IS_A')
                alias_nodes = [target for _, target, rel in self.graph.out_edges(oid, keys=True) if rel == 'DETECTOR_ALIAS']
                visited.update(alias_nodes)
                candidates.append({'concept': self.graph.nodes[oid]['name'], 'node_id': oid, 'role': role_node['name'], 'aliases': [self.graph.nodes[a]['label'] for a in alias_nodes], 'kg_relevance': edge['relevance_prior'], 'required': role_node['required'], 'output': role_node['output']})
            result['roles'][role_node['name']] = sorted(candidates, key=lambda c: (-c['kg_relevance'], c['concept']))
        affordances = sorted(self.graph.nodes[n]['name'] for n in required_affordance_ids)
        result['affordances'] = affordances
        result['required_affordances'] = affordances
        return self._finish(result, visited)

    def _finish(self, result, visited):
        edges = [dict(data) for node in sorted(visited) for _, target, _, data in self.graph.out_edges(node, keys=True, data=True) if target in visited]
        result['active_subgraph'] = {'nodes': [dict(self.graph.nodes[n]) for n in sorted(visited)], 'edges': edges}
        result['active_kg_nodes'] = len(visited)
        result['active_edges'] = len(edges)
        result['active_subgraph_reduction_ratio'] = 1 - len(visited) / max(1, len(self.graph))
        result['candidate_concept_count'] = len({c['concept'] for values in result['roles'].values() for c in values})
        return result

    def get_active_subgraph(self, task):
        return self.query_task(task)['active_subgraph']

    def get_candidates(self, task):
        result = self.query_task(task)
        return [candidate for values in result['roles'].values() for candidate in values]

    def get_detector_labels(self, task):
        return sorted({alias for candidate in self.get_candidates(task) for alias in candidate['aliases']})
