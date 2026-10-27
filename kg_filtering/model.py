import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .attention import RelationAwareAttention
from .features import EntityFeatures
from .config import Config
from .types import CandidateBatch


class EntityEncoder(nn.Module):
    def __init__(self, num_entities: int, config: Config) -> None:
        super().__init__()
        self.features = EntityFeatures(num_entities, config.embedding_dim,
            config.semantic_dim, config.use_semantic_features)
        self.dropout = config.dropout
        heads = config.num_heads if config.use_multi_head else 1
        self.layers = nn.ModuleList(RelationAwareAttention(config.embedding_dim,
            config.relation_dim, config.embedding_dim if final else config.embedding_dim // heads,
            heads, not final, config) for final in
            (index == config.num_layers - 1 for index in range(config.num_layers)))

    def forward(self, batch: CandidateBatch, relation_embedding: nn.Embedding) -> Tensor:
        x = self.features(batch.node_ids)
        relations = relation_embedding(batch.edge_types)
        for index, layer in enumerate(self.layers):
            x = layer(x, batch.edge_index, relations)
            if index + 1 < len(self.layers):
                x = F.dropout(F.elu(x), self.dropout, self.training)
        return x


class EdgeValidityModel(nn.Module):
    def __init__(self, num_entities: int, num_relations: int, config: Config) -> None:
        super().__init__()
        self.encoder = EntityEncoder(num_entities, config)
        self.relation_embedding = nn.Embedding(2 * num_relations + 1, config.relation_dim)
        self.classifier = nn.Sequential(
            nn.Linear(2 * config.embedding_dim + config.relation_dim, config.classifier_hidden_dim),
            nn.ReLU(), nn.Linear(config.classifier_hidden_dim, 1))
        nn.init.xavier_uniform_(self.relation_embedding.weight)

    def forward(self, batch: CandidateBatch) -> Tensor:
        entities = self.encoder(batch, self.relation_embedding)
        h, r, t = batch.candidates.unbind(1)
        return self.classifier(torch.cat((entities[h], self.relation_embedding(r), entities[t]), -1)).squeeze(-1)
