from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATv2Conv, GCNConv, RGCNConv

from .features import EntityFeatures
from .model import EdgeValidityModel


class CompositionalLayer(nn.Module):
    def __init__(self, dimension, relation_dim):
        super().__init__()
        self.relation = nn.Linear(relation_dim, dimension, bias=False)
        self.directions = nn.ModuleList(nn.Linear(dimension, dimension, bias=False) for _ in range(3))
        self.residual = nn.Linear(dimension, dimension)

    def forward(self, x, edge_index, edge_types, relation_embedding):
        source, target = edge_index
        relation_count = (relation_embedding.num_embeddings - 1) // 2
        directions = torch.where(edge_types == 2 * relation_count, 2,
            torch.where(edge_types >= relation_count, 1, 0))
        composed = x[source] * self.relation(relation_embedding(edge_types))
        messages = torch.zeros_like(composed)
        for index, projection in enumerate(self.directions):
            mask = directions == index
            messages[mask] = projection(composed[mask])
        degree = torch.bincount(target, minlength=x.size(0)).clamp_min(1).to(x.dtype)
        return self.residual(x) + torch.zeros_like(x).index_add(0, target, messages) / degree[:, None]


class BaselineEncoder(nn.Module):
    def __init__(self, num_entities, num_relations, config):
        super().__init__()
        self.features = EntityFeatures(num_entities, config.embedding_dim,
            config.semantic_dim, config.use_semantic_features)
        self.name, self.dropout = config.model_name, config.dropout
        dimension = config.embedding_dim
        heads = config.num_heads if config.use_multi_head else 1
        layers = []
        for index in range(config.num_layers):
            if self.name == 'gcn':
                layer = GCNConv(dimension, dimension, add_self_loops=False)
            elif self.name == 'gatv2':
                final = index + 1 == config.num_layers
                layer = GATv2Conv(dimension, dimension if final else dimension // heads,
                    heads=heads, concat=not final, dropout=config.dropout, add_self_loops=False)
            elif self.name == 'rgcn':
                layer = RGCNConv(dimension, dimension, 2 * num_relations + 1,
                    num_bases=min(30, 2 * num_relations + 1))
            elif self.name == 'compgcn':
                layer = CompositionalLayer(dimension, config.relation_dim)
            else:
                raise ValueError(f'Unknown neural baseline: {self.name}')
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

    def forward(self, batch, relation_embedding):
        x = self.features(batch.node_ids)
        for index, layer in enumerate(self.layers):
            if self.name == 'rgcn':
                x = layer(x, batch.edge_index, batch.edge_types)
            elif self.name == 'compgcn':
                x = layer(x, batch.edge_index, batch.edge_types, relation_embedding)
            else:
                x = layer(x, batch.edge_index)
            if index + 1 < len(self.layers):
                x = F.dropout(F.elu(x), self.dropout, self.training)
        return x


class TransEClassifier(nn.Module):
    def __init__(self, num_entities, num_relations, config):
        super().__init__()
        self.entities = nn.Embedding(num_entities, config.embedding_dim)
        self.relations = nn.Embedding(num_relations, config.embedding_dim)
        self.bias = nn.Parameter(torch.zeros(()))
        self.log_scale = nn.Parameter(torch.zeros(()))
        nn.init.xavier_uniform_(self.entities.weight)
        nn.init.xavier_uniform_(self.relations.weight)

    def forward(self, batch):
        h, r, t = batch.candidates.unbind(1)
        entities = F.normalize(self.entities(batch.node_ids), dim=-1)
        distance = (entities[h] + self.relations(r) - entities[t]).norm(p=1, dim=-1)
        return self.bias - self.log_scale.exp() * distance


def build_model(num_entities, num_relations, config):
    if config.model_name == 'transe':
        return TransEClassifier(num_entities, num_relations, config)
    model = EdgeValidityModel(num_entities, num_relations, config)
    if config.model_name != 'proposed':
        model.encoder = BaselineEncoder(num_entities, num_relations, config)
    return model


class StructuralFeatures:
    def __init__(self, context):
        self.triples = set(context.triples)
        self.adjacent = defaultdict(set)
        self.pair_relations = defaultdict(set)
        for h, r, t in self.triples:
            self.adjacent[h].add(t)
            self.adjacent[t].add(h)
            self.pair_relations[h, t].add(r)

    def transform(self, triples):
        rows = []
        for h, r, t in triples:
            removed = (h, r, t) in self.triples
            left, right = set(self.adjacent[h]), set(self.adjacent[t])
            if removed and len(self.pair_relations[h, t]) == 1 and not self.pair_relations.get((t, h)):
                left.discard(t)
                right.discard(h)
            common, union = len(left & right), len(left | right)
            clustering = []
            for neighbors in (left, right):
                count = len(neighbors)
                links = sum(len(self.adjacent[node] & neighbors) for node in neighbors)
                clustering.append(links / (count * (count - 1)) if count > 1 else 0.0)
            rows.append((len(left), len(right), *clustering, common,
                common / union if union else 0.0,
                common / min(len(left), len(right)) if left and right else 0.0))
        return rows


def fit_random_forest(encoded, labels, context, config):
    from sklearn.ensemble import RandomForestClassifier

    features = StructuralFeatures(context)
    classifier = RandomForestClassifier(n_estimators=config.forest_estimators,
        class_weight='balanced', random_state=config.seed, n_jobs=-1)
    classifier.fit(features.transform(encoded['train']), labels['train'])
    positive = list(classifier.classes_).index(1) if 1 in classifier.classes_ else None
    return {name: torch.tensor(classifier.predict_proba(features.transform(triples))[:, positive],
        dtype=torch.float32) if positive is not None else torch.zeros(len(triples))
        for name, triples in encoded.items()}

