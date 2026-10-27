import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.utils import softmax

from .config import Config


class HeadProjection(nn.Module):
    def __init__(self, inputs: int, heads: int, channels: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(heads, inputs, channels))
        for head in self.weight:
            nn.init.xavier_uniform_(head)

    def forward(self, values: Tensor) -> Tensor:
        return torch.einsum('ni,kio->nko', values, self.weight)


class RelationAwareAttention(nn.Module):
    def __init__(self, inputs: int, relation_dim: int, channels: int, heads: int, concat: bool, config: Config) -> None:
        super().__init__()
        self.trace_sink = None
        self.concat = concat
        self.dynamic = config.use_gatv2
        self.relation_attention = config.use_relation_attention
        self.relation_messages = config.use_relation_messages
        self.dropout = config.dropout
        self.query = HeadProjection(inputs, heads, channels)
        self.key = HeadProjection(inputs, heads, channels)
        self.relation = HeadProjection(relation_dim, heads, channels)
        self.attention_relation = nn.Parameter(torch.empty(heads, channels, channels))
        self.gate = nn.Parameter(torch.empty(heads, channels, channels))
        self.message_relation = nn.Parameter(torch.empty(heads, channels, channels))
        self.neighbor = HeadProjection(inputs, heads, channels)
        self.residual = HeadProjection(inputs, heads, channels)
        self.attention = nn.Parameter(torch.empty(heads, channels))
        for parameter in (self.attention_relation, self.gate, self.message_relation):
            for head in parameter:
                nn.init.xavier_uniform_(head)
        nn.init.xavier_uniform_(self.attention)

    def forward(self, x: Tensor, edge_index: Tensor, relation_features: Tensor) -> Tensor:
        source, target = edge_index
        projected = self.relation(relation_features)
        attention_input = self.query(x)[target] + self.key(x)[source]
        if self.relation_attention:
            attention_input = attention_input + torch.einsum('nki,kio->nko', projected, self.attention_relation)
        if self.dynamic:
            scores = (F.leaky_relu(attention_input, 0.2) * self.attention).sum(-1)
        else:
            scores = F.leaky_relu((attention_input * self.attention).sum(-1), 0.2)
        alpha = softmax(scores, target, num_nodes=x.size(0))
        messages = self.neighbor(x)[source]
        gate = None
        if self.relation_messages:
            gate = torch.sigmoid(torch.einsum('nki,kio->nko', projected, self.gate))
            messages = gate * messages + torch.einsum('nki,kio->nko', projected, self.message_relation)
        if self.trace_sink is not None:
            self.trace_sink(edge_index, alpha, gate)
        weighted = messages * F.dropout(alpha, self.dropout, self.training).unsqueeze(-1)
        output = self.residual(x).index_add(0, target, weighted)
        return output.flatten(1) if self.concat else output.mean(1)
