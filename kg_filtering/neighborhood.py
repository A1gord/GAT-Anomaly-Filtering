from dataclasses import dataclass
from sys import getsizeof

from .indices import RelationalIndex
from .types import TripleId


@dataclass(frozen=True, slots=True)
class Neighborhood:
    nodes: tuple[int, ...]
    edges: tuple[TripleId, ...]
    frontier_sizes: tuple[int, ...]
    excluded_fact: int | None

    @property
    def accounted_bytes(self) -> int:
        return getsizeof(self) + getsizeof(self.frontier_sizes) + sum(map(getsizeof, self.frontier_sizes)) + getsizeof(self.nodes) + sum(map(getsizeof, self.nodes)) + \
            getsizeof(self.edges) + sum(getsizeof(edge) + sum(map(getsizeof, edge)) for edge in self.edges)


class IncomingExpansion:
    def __init__(self, index: RelationalIndex):
        self.index = index

    def extract(self, candidate: TripleId, depth: int) -> Neighborhood:
        head, relation, tail = candidate
        if depth < 1 or not 0 <= relation < self.index.num_relations:
            raise ValueError('Invalid expansion depth or candidate relation')
        if not all(0 <= node < self.index.num_entities for node in (head, tail)):
            raise ValueError('Candidate entity is outside the context vocabulary')
        excluded = self.index.fact_identity(candidate)
        frontier, nodes = {head, tail}, {head, tail}
        edges, widths = set(), []
        for _ in range(depth):
            widths.append(len(frontier))
            if not frontier:
                break
            following = set()
            for target in sorted(frontier):
                for arc in self.index.arcs(target):
                    if excluded is not None and arc.origin == excluded:
                        continue
                    edges.add((arc.source, arc.relation, arc.target))
                    following.add(arc.source)
            frontier = following - nodes
            nodes.update(following)
        self_relation = 2 * self.index.num_relations
        edges.update((node, self_relation, node) for node in nodes)
        return Neighborhood(tuple(sorted(nodes)), tuple(sorted(edges)), tuple(widths), excluded)
