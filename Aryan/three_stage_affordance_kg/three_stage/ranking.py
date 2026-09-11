"""Cross-query clustering, conservative box checks, and role-wise ranking."""
import math
import re


# Visually similar KG concepts that should vote together without claiming a
# subtype that Florence did not independently verify.
CONCEPT_FAMILIES = {
    'bucket': {
        'bucket', 'pail', 'water_bucket', 'sand_bucket', 'bucket_of_sand',
        'fire_bucket',
    },
}


def _concept_family(concept):
    for canonical, members in CONCEPT_FAMILIES.items():
        if concept in members:
            return canonical
    return concept


def iou(a, b):
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0


def _normalize(value):
    return re.sub(r'[^a-z0-9]+', '_', str(value).lower()).strip('_')


def bbox_metrics(bbox, image_size):
    if len(bbox) != 4 or image_size is None or len(image_size) != 2:
        return {'area_ratio': None, 'width_ratio': None, 'height_ratio': None}
    image_width, image_height = image_size
    if image_width <= 0 or image_height <= 0:
        return {'area_ratio': None, 'width_ratio': None, 'height_ratio': None}
    width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    return {
        'area_ratio': width * height / (image_width * image_height),
        'width_ratio': width / image_width,
        'height_ratio': height / image_height,
    }


def _rejected(reason, rule, bbox, image_size, **extra):
    return {
        'reason': reason,
        'rejection_rule': rule,
        'bbox': bbox,
        'bbox_metrics': bbox_metrics(bbox, image_size),
        **extra,
    }


def _cluster_detections(detections, threshold):
    """Union detections transitively when their boxes strongly overlap."""
    parents = list(range(len(detections)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left, right):
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left in range(len(detections)):
        for right in range(left + 1, len(detections)):
            if iou(detections[left]['bbox'], detections[right]['bbox']) >= threshold:
                union(left, right)
    clusters = {}
    for index, detection in enumerate(detections):
        clusters.setdefault(find(index), []).append(detection)
    return list(clusters.values())


def rank(raw, vocabulary, roles, config, image_size=None, reference_detections=None):
    reference_detections = reference_detections or []
    groups = {role: [] for role in roles}
    rejected = []
    valid = []
    near_full_limit = config['near_full_image_area_ratio']
    cluster_iou = config['cross_query_cluster_iou']

    for detection in raw:
        bbox = detection.get('bbox', [])
        confidence = detection.get('od_confidence')
        if len(bbox) != 4 or any(type(x) not in (int, float) or not math.isfinite(x) for x in bbox) or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            rejected.append(_rejected('invalid_bbox', 'bbox must contain four finite coordinates with x2>x1 and y2>y1', bbox, image_size, detection=detection))
            continue
        if image_size is None or len(image_size) != 2 or image_size[0] <= 0 or image_size[1] <= 0:
            rejected.append(_rejected('missing_image_size_for_sanity', 'valid image width and height are required', bbox, image_size, detection=detection))
            continue
        if bbox[0] < 0 or bbox[1] < 0 or bbox[2] > image_size[0] or bbox[3] > image_size[1]:
            rejected.append(_rejected('bbox_outside_image', '0 <= x1 < x2 <= image_width and 0 <= y1 < y2 <= image_height', bbox, image_size, detection=detection))
            continue
        if bbox_metrics(bbox, image_size)['area_ratio'] > near_full_limit:
            rejected.append(_rejected('near_full_image_bbox', f'bbox_area_ratio > {near_full_limit:.2f}', bbox, image_size, detection=detection))
            continue
        if confidence is not None and (type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1):
            rejected.append(_rejected('invalid_confidence', 'OD confidence must be null or within [0,1]', bbox, image_size, detection=detection))
            continue
        if confidence is None and config['missing_confidence_policy'] == 'reject':
            rejected.append(_rejected('confidence_unavailable', 'missing_confidence_policy=reject', bbox, image_size, detection=detection))
            continue
        if confidence is not None and confidence < config['od_confidence_threshold']:
            rejected.append(_rejected('below_confidence_threshold', f'od_confidence < {config["od_confidence_threshold"]}', bbox, image_size, detection=detection))
            continue
        if detection.get('query') not in vocabulary:
            rejected.append(_rejected('unrequested_query', 'query must exist in the active KG detector vocabulary', bbox, image_size, detection=detection))
            continue
        valid.append(detection)

    alias_to_concepts = {}
    for query, candidates in vocabulary.items():
        for candidate in candidates:
            for label in [query, candidate['concept'].replace('_', ' '), *candidate.get('aliases', [])]:
                alias_to_concepts.setdefault(_normalize(label), set()).add(candidate['concept'])

    # A visual vote is meaningful only inside one semantic family. For example,
    # bucket and pail may corroborate each other; bucket and sprinkler may not.
    family_detections = []
    for detection in valid:
        concepts = {candidate['concept'] for candidate in vocabulary[detection['query']]}
        for family in sorted({_concept_family(concept) for concept in concepts}):
            family_detections.append({
                **detection,
                '_family': family,
                '_family_concepts': sorted(
                    concept for concept in concepts if _concept_family(concept) == family
                ),
            })

    clusters = []
    alias_agreement_iou = min(cluster_iou, 0.7)
    for family in sorted({item['_family'] for item in family_detections}):
        members = [item for item in family_detections if item['_family'] == family]
        clusters.extend(_cluster_detections(members, alias_agreement_iou))

    for cluster_index, members in enumerate(clusters, start=1):
        bbox = members[0]['bbox']
        family = members[0]['_family']
        supporting_queries = sorted({item['query'] for item in members})
        supporting_labels = sorted({item.get('label', item['query']) for item in members})
        candidate_records = {}
        for member in members:
            for candidate in vocabulary[member['query']]:
                if _concept_family(candidate['concept']) == family:
                    candidate_records.setdefault(candidate['concept'], []).append(candidate)
        # Keep the conservative family concept available even when only subtype
        # aliases fired (for example pail + water bucket -> bucket).
        if family in CONCEPT_FAMILIES and family not in candidate_records:
            family_candidates = [
                candidate
                for candidates in vocabulary.values()
                for candidate in candidates
                if candidate['concept'] == family
            ]
            if family_candidates:
                candidate_records[family] = family_candidates
        candidate_concepts = sorted(candidate_records)

        evidence = []
        for reference in reference_detections:
            overlap = iou(bbox, reference.get('bbox', [])) if len(reference.get('bbox', [])) == 4 else 0
            if overlap < cluster_iou:
                continue
            for concept in alias_to_concepts.get(_normalize(reference.get('label', '')), set()):
                if concept in candidate_records and _concept_family(concept) == family:
                    evidence.append((overlap, concept, reference))

        confidences = [item['od_confidence'] for item in members if item.get('od_confidence') is not None]
        canonical_query_present = any(_normalize(query) == _normalize(family) for query in supporting_queries)
        alias_support_count = len(supporting_queries)
        if not confidences and not canonical_query_present and alias_support_count < 2:
            rejected.append(_rejected(
                'insufficient_text_conditioned_support',
                'unscored detection requires the canonical family query or >=2 same-family aliases at IoU >= 0.70',
                bbox,
                image_size,
                cluster_id=cluster_index,
                canonical_concept=family,
                supporting_queries=supporting_queries,
                alias_support_count=alias_support_count,
                reference_od_supported=bool(evidence),
                candidate_concepts=candidate_concepts,
                reference_labels=sorted({item[2]['label'] for item in evidence}),
            ))
            continue

        if confidences:
            best_confidence = max(confidences)
            supported = {
                candidate['concept']
                for member in members if member.get('od_confidence') == best_confidence
                for candidate in vocabulary[member['query']]
                if _concept_family(candidate['concept']) == family
            }
            if len(supported) != 1:
                rejected.append(_rejected('ambiguous_confident_cluster', 'highest OD confidence supports more than one canonical concept', bbox, image_size, cluster_id=cluster_index, supporting_queries=supporting_queries, candidate_concepts=candidate_concepts))
                continue
            chosen_concept = next(iter(supported))
            resolution = 'od_confidence'
        else:
            # Standard OD may prove a specific subtype. Otherwise use the
            # conservative family concept (for example water_bucket -> bucket).
            reference_concepts = {item[1] for item in evidence}
            if len(reference_concepts) == 1:
                chosen_concept = next(iter(reference_concepts))
                resolution = 'standard_od_subtype_support'
            elif family in candidate_records:
                chosen_concept = family
                resolution = 'canonical_family'
            else:
                chosen_concept = max(
                    candidate_concepts,
                    key=lambda concept: max(item['kg_relevance'] for item in candidate_records[concept]),
                )
                resolution = 'same_family_alias_consensus'

        best_reference_iou = max((item[0] for item in evidence), default=None)
        reference_labels = sorted({item[2]['label'] for item in evidence})
        reference_supported = bool(evidence)
        if canonical_query_present and alias_support_count >= 2:
            acceptance_reason = 'canonical_query_plus_alias_agreement'
        elif canonical_query_present:
            acceptance_reason = 'canonical_query_detected'
        elif alias_support_count >= 2:
            acceptance_reason = 'same_family_alias_agreement'
        else:
            acceptance_reason = 'scored_visual_detection'
        evidence_strength = (
            'strong_reference_supported' if reference_supported
            else ('moderate_alias_consensus' if alias_support_count >= 2 else 'basic_canonical_query')
        )

        confidence = max(confidences) if confidences else None
        for candidate in {item['role']: item for item in candidate_records[chosen_concept]}.values():
            prior = candidate['kg_relevance']
            if confidence is None:
                score, basis = None, 'unscored_text_conditioned_florence'
            elif config['use_kg_weights']:
                score, basis = config['alpha'] * confidence + config['beta'] * prior, 'weighted_od_kg'
            else:
                score, basis = confidence, 'od_only'
            groups[candidate['role']].append({
                **candidate,
                'detector_label': chosen_concept.replace('_', ' '),
                'bbox': bbox,
                'od_confidence': confidence,
                'final_score': score,
                'score_basis': basis,
                'cluster_id': cluster_index,
                'candidate_concepts': candidate_concepts,
                'supporting_queries': supporting_queries,
                'supporting_detector_labels': supporting_labels,
                'reference_labels': reference_labels,
                'visual_support_iou': best_reference_iou,
                'canonical_concept': chosen_concept,
                'alias_support_count': alias_support_count,
                'reference_od_supported': reference_supported,
                'acceptance_reason': acceptance_reason,
                'evidence_strength': evidence_strength,
                'concept_resolution': resolution,
            })

    final = {}
    for role, items in groups.items():
        output_role = any(candidate.get('output', True) for candidate in roles[role])
        items.sort(key=lambda item: (-(item['final_score'] if item['final_score'] is not None else -1), -(item['visual_support_iou'] or 0), item['concept'], item['bbox']))
        if output_role:
            final[role] = {'output': True, 'status': 'found' if items else 'not_found', 'kg_candidates': [item['concept'] for item in roles[role]], 'detections': items, 'selected': items[:config['top_k']]}
        else:
            final[role] = {'output': False, 'status': 'context_only', 'kg_candidates': [item['concept'] for item in roles[role]], 'detections': [], 'selected': []}

    required_roles = [role for role, candidates in roles.items() if any(item.get('required') and item.get('output', True) for item in candidates)]
    found_required = [role for role in required_roles if final[role]['status'] == 'found']
    output_values = [values for values in final.values() if values['output']]
    any_found = any(values['status'] == 'found' for values in output_values)
    if required_roles:
        status = 'success' if len(found_required) == len(required_roles) else ('partial' if found_required else 'no_suitable_object_detected')
    else:
        status = 'success' if any_found else 'no_suitable_object_detected'
    all_ranked = [item for values in output_values for item in values['detections']]
    return {'status': status, 'verification_status': 'pending_manual_review' if any_found else 'not_applicable', 'roles': final, 'all_ranked_objects': all_ranked, 'rejected': rejected}
