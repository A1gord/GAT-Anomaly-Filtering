from dataclasses import dataclass, field
from typing import Callable

import torch

from .attention import RelationAwareAttention


@dataclass
class AttentionCapture:
    model: object
    records: list[dict] = field(default_factory=list)
    previous: list[tuple[RelationAwareAttention, Callable | None]] = field(default_factory=list)

    def __enter__(self):
        for name, module in self.model.named_modules():
            if isinstance(module, RelationAwareAttention):
                self.previous.append((module, module.trace_sink))
                module.trace_sink = self._receiver(name)
        return self

    def _receiver(self, name):
        def receive(edge_index, probabilities, gate):
            self.records.append({'layer': name, 'edge_index': edge_index.detach().cpu(),
                'attention': probabilities.detach().cpu(),
                'gate_mean': None if gate is None else gate.detach().mean(-1).cpu()})
        return receive

    def __exit__(self, error_type, error, traceback):
        for module, previous in self.previous:
            module.trace_sink = previous
        self.previous.clear()


def relation_label(identifier, id2rel):
    count = len(id2rel)
    if identifier == 2 * count:
        return {'kind': 'self', 'relation': None}
    return {'kind': 'inverse' if identifier >= count else 'forward',
        'relation': id2rel[identifier % count]}


@torch.no_grad()
def explain_candidates(model, encoded, batcher, ent2id, rel2id, device, count, top_k):
    id2ent = {value: key for key, value in ent2id.items()}
    id2rel = {value: key for key, value in rel2id.items()}
    explanations = []
    if not any(isinstance(module, RelationAwareAttention) for module in model.modules()):
        return {'available': False, 'reason': 'this model has no relation-aware attention layers', 'candidates': []}
    was_training = model.training
    model.eval()
    try:
        for row, candidate in enumerate(encoded[:count]):
            batch = batcher.collate([candidate])
            with AttentionCapture(model) as capture:
                probability = float(model(batch.to(device)).sigmoid()[0])
            layers = []
            local_endpoints = batch.candidates[0, [0, 2]].tolist()
            for record in capture.records:
                sources, targets = record['edge_index']
                endpoints = []
                for endpoint in dict.fromkeys(local_endpoints):
                    available = torch.nonzero(targets == endpoint).flatten()
                    heads = []
                    for head in range(record['attention'].size(1)):
                        weights = record['attention'][available, head]
                        order = torch.argsort(weights, descending=True, stable=True)[:top_k]
                        edges = []
                        for local_index in order.tolist():
                            index = int(available[local_index])
                            edges.append({'source': id2ent[int(batch.node_ids[sources[index]])],
                                **relation_label(int(batch.edge_types[index]), id2rel),
                                'attention': float(record['attention'][index, head]),
                                'mean_gate': None if record['gate_mean'] is None else float(record['gate_mean'][index, head])})
                        heads.append({'head': head, 'incoming': edges,
                            'retained_attention_mass': sum(edge['attention'] for edge in edges)})
                    endpoints.append({'entity': id2ent[int(batch.node_ids[endpoint])], 'heads': heads})
                layers.append({'layer': record['layer'], 'endpoints': endpoints})
            head, relation, tail = candidate
            explanations.append({'row': row, 'triple': [id2ent[head], id2rel[relation], id2ent[tail]],
                'valid_probability': probability, 'layers': layers,
                'context_nodes': len(batch.node_ids), 'context_edges': len(batch.edge_types)})
    finally:
        model.train(was_training)
    return {'available': True, 'interpretation': 'attention and gate summaries are descriptive, not causal attributions',
        'candidate_selection': 'first encoded input occurrences', 'candidates': explanations}
