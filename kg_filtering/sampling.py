import random
from collections.abc import Iterator, Sequence

import torch

from .graph import RelationalContext
from .types import CandidateBatch, TripleId


class CandidateBatcher:
    def __init__(self, context: RelationalContext, depth: int, batch_size: int) -> None:
        self.context = context
        self.depth = depth
        self.batch_size = batch_size

    def collate(self, candidates: Sequence[TripleId]) -> CandidateBatch:
        node_ids, edges, edge_types, local_candidates = [], [], [], []
        for candidate in candidates:
            nodes, typed_edges = self.context.neighborhood(tuple(candidate), self.depth)
            mapping = {node: index + len(node_ids) for index, node in enumerate(nodes)}
            h, r, t = candidate
            local_candidates.append((mapping[h], r, mapping[t]))
            for source, relation, target in typed_edges:
                edges.append((mapping[source], mapping[target]))
                edge_types.append(relation)
            node_ids.extend(nodes)
        return CandidateBatch(torch.tensor(node_ids, dtype=torch.long),
            torch.tensor(edges, dtype=torch.long).reshape(-1, 2).T.contiguous(),
            torch.tensor(edge_types, dtype=torch.long),
            torch.tensor(local_candidates, dtype=torch.long).reshape(-1, 3))

    def batches(self, triples: Sequence[TripleId], shuffle: bool = False, seed: int = 0) -> Iterator[tuple[list[int], CandidateBatch]]:
        indices = list(range(len(triples)))
        if shuffle:
            random.Random(seed).shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            selection = indices[start:start + self.batch_size]
            yield selection, self.collate([triples[index] for index in selection])
