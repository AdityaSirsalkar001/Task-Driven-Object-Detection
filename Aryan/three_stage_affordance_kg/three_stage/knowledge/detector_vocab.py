"""Translate semantic concepts into queries independently of OD implementation."""


def detector_vocabulary(kg_result, use_aliases=True, supported_labels=None):
    mapping = {}
    for candidates in kg_result['roles'].values():
        for candidate in candidates:
            if not candidate.get('output', True):
                continue
            aliases = candidate['aliases'] if use_aliases else [candidate['concept'].replace('_', ' ')]
            for alias in aliases:
                if supported_labels is not None and alias not in supported_labels:
                    continue
                entries = mapping.setdefault(alias, [])
                if candidate not in entries:
                    entries.append(candidate)
    return dict(sorted(mapping.items()))
