import json
from pathlib import Path

import torch
from torch import nn


def load_semantic_features(path, ent2id, dimension):
    if not path:
        if dimension:
            raise ValueError('semantic_dim requires entity_features_path')
        return torch.empty(len(ent2id), 0)
    payload = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if payload.keys() - ent2id.keys():
        raise ValueError('Feature file contains unknown entities')
    if not payload:
        raise ValueError('Semantic feature file is empty')
    inferred = len(next(iter(payload.values())))
    if inferred < 1 or dimension and dimension != inferred:
        raise ValueError('Invalid semantic feature dimension')
    matrix = torch.zeros(len(ent2id), inferred)
    for entity, values in payload.items():
        if len(values) != inferred:
            raise ValueError(f'Inconsistent feature dimension for {entity}')
        matrix[ent2id[entity]] = torch.tensor(values, dtype=torch.float32)
    if not torch.isfinite(matrix).all():
        raise ValueError('Semantic features must be finite')
    return matrix


class EntityFeatures(nn.Module):
    def __init__(self, num_entities, dimension, semantic_dim, enabled=True):
        super().__init__()
        self.embedding = nn.Embedding(num_entities, dimension)
        self.register_buffer('semantic', torch.zeros(num_entities, semantic_dim))
        self.projection = nn.Linear(semantic_dim, dimension, bias=False) if semantic_dim and enabled else None
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, node_ids):
        result = self.embedding(node_ids)
        return result if self.projection is None else result + self.projection(self.semantic[node_ids])
