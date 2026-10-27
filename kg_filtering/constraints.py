import json
from collections import defaultdict
from pathlib import Path

import torch


class DomainConstraints:
    def __init__(self, path, context, id2ent, id2rel):
        if not path:
            raise ValueError('The TMF rule baseline requires rules_path')
        payload = json.loads(Path(path).read_text(encoding='utf-8-sig'))
        self.types = {entity: set(values) for entity, values in payload.get('entity_types', {}).items()}
        self.domains = {relation: set(values) for relation, values in payload.get('relation_domains', {}).items()}
        self.ranges = {relation: set(values) for relation, values in payload.get('relation_ranges', {}).items()}
        self.hierarchical = set(payload.get('hierarchical_relations', []))
        self.id2ent, self.id2rel = id2ent, id2rel
        self.triples = context.triples

    def probabilities(self, triples):
        values = []
        for h, r, t in triples:
            head, relation, tail = self.id2ent[h], self.id2rel[r], self.id2ent[t]
            checks = []
            if relation in self.domains:
                checks.append(bool(self.types.get(head, set()) & self.domains[relation]))
            if relation in self.ranges:
                checks.append(bool(self.types.get(tail, set()) & self.ranges[relation]))
            if relation in self.hierarchical:
                adjacency = defaultdict(set)
                for left, predicate, right in self.triples:
                    if (left, predicate, right) != (h, r, t) and self.id2rel[predicate] == relation:
                        adjacency[left].add(right)
                frontier, seen = [t], set()
                cycle = h == t
                while frontier and not cycle:
                    node = frontier.pop()
                    if node in seen:
                        continue
                    seen.add(node)
                    cycle = h in adjacency[node]
                    frontier.extend(adjacency[node] - seen)
                checks.append(not cycle)
            if not checks:
                raise ValueError(f'No applicable domain constraints for {relation}')
            values.append(float(all(checks)))
        return torch.tensor(values)
