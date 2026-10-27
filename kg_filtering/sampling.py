import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, asdict

from .assembly import BatchLimits, CandidateAssembler, ContextBudgetExceeded
from .graph import RelationalContext
from .types import CandidateBatch, TripleId


@dataclass
class BatchCounters:
    batches: int = 0
    candidates: int = 0
    nodes: int = 0
    edges: int = 0
    peak_nodes: int = 0
    peak_edges: int = 0

    def observe(self, assembler: CandidateAssembler) -> None:
        self.batches += 1
        self.candidates += len(assembler.candidates)
        self.nodes += len(assembler.node_ids)
        self.edges += len(assembler.edge_types)
        self.peak_nodes = max(self.peak_nodes, len(assembler.node_ids))
        self.peak_edges = max(self.peak_edges, len(assembler.edge_types))


class CandidateBatcher:
    def __init__(self, context: RelationalContext, depth: int, batch_size: int,
                 max_nodes: int = 0, max_edges: int = 0) -> None:
        if min(depth, batch_size) < 1 or min(max_nodes, max_edges) < 0:
            raise ValueError('Invalid candidate batch limits')
        self.context, self.depth, self.batch_size = context, depth, batch_size
        self.limits = BatchLimits(batch_size, max_nodes, max_edges)
        self.counters = BatchCounters()

    @classmethod
    def from_config(cls, context, depth, config):
        return cls(context, depth, config.batch_size, config.batch_max_nodes, config.batch_max_edges)

    def collate(self, candidates: Sequence[TripleId]) -> CandidateBatch:
        assembler = CandidateAssembler()
        for row, candidate in enumerate(candidates):
            neighborhood = self.context.extract(tuple(candidate), self.depth)
            if not assembler.fits(neighborhood, self.limits):
                raise ContextBudgetExceeded(row, len(assembler.node_ids) + len(neighborhood.nodes),
                    len(assembler.edge_types) + len(neighborhood.edges))
            assembler.append(row, candidate, neighborhood)
        self.counters.observe(assembler)
        return assembler.materialize()

    def batches(self, triples: Sequence[TripleId], shuffle: bool = False,
                seed: int = 0) -> Iterator[tuple[list[int], CandidateBatch]]:
        indices = list(range(len(triples)))
        if shuffle:
            random.Random(seed).shuffle(indices)
        assembler = CandidateAssembler()
        for row in indices:
            candidate = tuple(triples[row])
            neighborhood = self.context.extract(candidate, self.depth)
            if assembler.candidates and not assembler.fits(neighborhood, self.limits):
                self.counters.observe(assembler)
                yield assembler.row_indices, assembler.materialize()
                assembler = CandidateAssembler()
            if not assembler.fits(neighborhood, self.limits):
                raise ContextBudgetExceeded(row, len(neighborhood.nodes), len(neighborhood.edges))
            assembler.append(row, candidate, neighborhood)
        if assembler.candidates:
            self.counters.observe(assembler)
            yield assembler.row_indices, assembler.materialize()

    def describe(self) -> dict:
        return {**asdict(self.counters), 'limits': asdict(self.limits), 'depth': self.depth}
