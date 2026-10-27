from dataclasses import dataclass, field

import torch

from .neighborhood import Neighborhood
from .types import CandidateBatch, TripleId


class ContextBudgetExceeded(MemoryError):
    def __init__(self, row: int, nodes: int, edges: int):
        self.row, self.nodes, self.edges = row, nodes, edges
        super().__init__(f'Candidate row {row} requires {nodes} nodes and {edges} edges; increase the exact-context budget')


@dataclass(frozen=True, slots=True)
class BatchLimits:
    candidates: int
    nodes: int = 0
    edges: int = 0

    def accepts(self, candidates: int, nodes: int, edges: int) -> bool:
        return candidates <= self.candidates and (not self.nodes or nodes <= self.nodes) and \
            (not self.edges or edges <= self.edges)


@dataclass
class CandidateAssembler:
    node_ids: list[int] = field(default_factory=list)
    sources: list[int] = field(default_factory=list)
    targets: list[int] = field(default_factory=list)
    edge_types: list[int] = field(default_factory=list)
    candidates: list[TripleId] = field(default_factory=list)
    row_indices: list[int] = field(default_factory=list)

    def fits(self, neighborhood: Neighborhood, limits: BatchLimits) -> bool:
        return limits.accepts(len(self.candidates) + 1, len(self.node_ids) + len(neighborhood.nodes),
            len(self.edge_types) + len(neighborhood.edges))

    def append(self, row: int, candidate: TripleId, neighborhood: Neighborhood) -> None:
        offset = len(self.node_ids)
        local = {node: index + offset for index, node in enumerate(neighborhood.nodes)}
        head, relation, tail = candidate
        self.candidates.append((local[head], relation, local[tail]))
        self.row_indices.append(row)
        for source, predicate, target in neighborhood.edges:
            self.sources.append(local[source])
            self.targets.append(local[target])
            self.edge_types.append(predicate)
        self.node_ids.extend(neighborhood.nodes)

    def materialize(self) -> CandidateBatch:
        if not self.candidates:
            raise ValueError('Cannot materialize an empty candidate graph batch')
        return CandidateBatch(torch.tensor(self.node_ids, dtype=torch.long),
            torch.tensor([self.sources, self.targets], dtype=torch.long),
            torch.tensor(self.edge_types, dtype=torch.long),
            torch.tensor(self.candidates, dtype=torch.long))
