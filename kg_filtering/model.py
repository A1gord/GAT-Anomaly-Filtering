import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

from .config import Config


class CustomGATLayer(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int,
        concat: bool,
        dropout: float,
        negative_slope: float,
        use_gatv2: bool,
        use_edge_weight_awareness: bool,
    ) -> None:
        super().__init__(aggr="add")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.use_gatv2 = use_gatv2
        self.use_edge_weight_awareness = use_edge_weight_awareness
        if use_gatv2:
            self.lin_src = nn.Linear(in_channels, heads * out_channels, bias=False)
            self.lin_dst = nn.Linear(in_channels, heads * out_channels, bias=False)
            self.att = nn.Parameter(torch.empty(1, heads, out_channels))
            if use_edge_weight_awareness:
                self.lin_edge = nn.Linear(1, heads * out_channels, bias=False)
        else:
            self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
            self.att_src = nn.Parameter(torch.empty(1, heads, out_channels))
            self.att_dst = nn.Parameter(torch.empty(1, heads, out_channels))
            if use_edge_weight_awareness:
                self.lin_edge = nn.Linear(1, heads * out_channels, bias=False)
                self.att_edge = nn.Parameter(torch.empty(1, heads, out_channels))
        bias_dim = heads * out_channels if concat else out_channels
        self.bias = nn.Parameter(torch.empty(bias_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.use_gatv2:
            nn.init.xavier_uniform_(self.lin_src.weight)
            nn.init.xavier_uniform_(self.lin_dst.weight)
            nn.init.xavier_uniform_(self.att)
            if self.use_edge_weight_awareness:
                nn.init.xavier_uniform_(self.lin_edge.weight)
        else:
            nn.init.xavier_uniform_(self.lin.weight)
            nn.init.xavier_uniform_(self.att_src)
            nn.init.xavier_uniform_(self.att_dst)
            if self.use_edge_weight_awareness:
                nn.init.xavier_uniform_(self.lin_edge.weight)
                nn.init.xavier_uniform_(self.att_edge)
        nn.init.zeros_(self.bias)

    def forward(
        self, x: Tensor, edge_index: Tensor, edge_weight: Tensor | None = None
    ) -> Tensor:
        if self.use_gatv2:
            x_src = self.lin_src(x).view(-1, self.heads, self.out_channels)
            x_dst = self.lin_dst(x).view(-1, self.heads, self.out_channels)
            out = self.propagate(edge_index, x=(x_src, x_dst), edge_weight=edge_weight)
        else:
            x_lin = self.lin(x).view(-1, self.heads, self.out_channels)
            out = self.propagate(edge_index, x=(x_lin, x_lin), edge_weight=edge_weight)
        if self.concat:
            out = out.reshape(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        out = out + self.bias
        return out

    def message(
        self,
        x_i: Tensor,
        x_j: Tensor,
        edge_weight: Tensor | None,
        index: Tensor,
        ptr: Tensor | None,
        size_i: int | None,
    ) -> Tensor:
        if self.use_gatv2:
            alpha_input = x_i + x_j
            if self.use_edge_weight_awareness and edge_weight is not None:
                edge_term = self.lin_edge(edge_weight).view(
                    -1, self.heads, self.out_channels
                )
                alpha_input = alpha_input + edge_term
            alpha = (F.leaky_relu(alpha_input, self.negative_slope) * self.att).sum(
                dim=-1
            )
        else:
            alpha = (x_i * self.att_src).sum(dim=-1) + (x_j * self.att_dst).sum(dim=-1)
            if self.use_edge_weight_awareness and edge_weight is not None:
                edge_term = self.lin_edge(edge_weight).view(
                    -1, self.heads, self.out_channels
                )
                alpha = alpha + (edge_term * self.att_edge).sum(dim=-1)
            alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)


class EntityEncoder(nn.Module):
    def __init__(
        self,
        num_entities: int,
        embedding_dim: int,
        num_layers: int,
        num_heads: int,
        use_multi_head: bool,
        use_gatv2: bool,
        use_edge_weight_awareness: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        self.entity_embedding = nn.Embedding(num_entities, embedding_dim)
        self.dropout = dropout
        heads = num_heads if use_multi_head else 1
        self.layers = nn.ModuleList()
        if (
            use_multi_head
            and heads > 1
            and embedding_dim % heads != 0
            and num_layers > 1
        ):
            raise ValueError(
                "embedding_dim must be divisible by num_heads for multi-head intermediate layers"
            )
        in_channels = embedding_dim
        for layer_idx in range(num_layers):
            is_final = layer_idx == num_layers - 1
            concat = bool(use_multi_head and heads > 1 and not is_final)
            out_channels = (
                embedding_dim
                if is_final
                else (embedding_dim // heads if concat else embedding_dim)
            )
            layer = CustomGATLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                heads=heads,
                concat=concat,
                dropout=dropout,
                negative_slope=0.2,
                use_gatv2=use_gatv2,
                use_edge_weight_awareness=use_edge_weight_awareness,
            )
            self.layers.append(layer)
            in_channels = embedding_dim

    def forward(self, edge_index: Tensor, edge_weight: Tensor | None = None) -> Tensor:
        x = self.entity_embedding.weight
        for idx, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_weight)
            if idx < len(self.layers) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class EdgeValidityModel(nn.Module):
    def __init__(self, num_entities: int, num_relations: int, config: Config) -> None:
        super().__init__()
        self.encoder = EntityEncoder(
            num_entities=num_entities,
            embedding_dim=config.embedding_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            use_multi_head=config.use_multi_head,
            use_gatv2=config.use_gatv2,
            use_edge_weight_awareness=config.use_edge_weight_awareness,
            dropout=config.dropout,
        )
        self.relation_embedding = nn.Embedding(num_relations, config.embedding_dim)
        self.classifier = nn.Sequential(
            nn.Linear(config.embedding_dim * 3, config.classifier_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.classifier_hidden_dim, 1),
        )

    def forward(
        self, triples: Tensor, edge_index: Tensor, edge_weight: Tensor | None = None
    ) -> Tensor:
        entity_repr = self.encoder(edge_index, edge_weight)
        h = entity_repr[triples[:, 0]]
        r = self.relation_embedding(triples[:, 1])
        t = entity_repr[triples[:, 2]]
        edge_repr = torch.cat([h, r, t], dim=-1)
        logits = self.classifier(edge_repr).squeeze(-1)
        return logits
