from collections import defaultdict
from collections.abc import Sequence

from .types import TripleId


class RelationalContext:
    def __init__(self, triples: Sequence[TripleId], num_entities: int, num_relations: int, inverse: bool = True) -> None:
        self.triples = tuple(sorted(set(tuple(triple) for triple in triples)))
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.inverse = inverse
        self.self_relation = 2 * num_relations
        self.incoming = defaultdict(list)
        for h, r, t in self.triples:
            self.incoming[t].append((h, r, t, (h, r, t)))
            self.incoming[h].append((t, r + num_relations if inverse else r, h, (h, r, t)))

    def neighborhood(self, candidate: TripleId, depth: int) -> tuple[list[int], list[TripleId]]:
        h, r, t = candidate
        frontier = {h, t}
        nodes = set(frontier)
        edges = set()
        for _ in range(depth):
            following = set()
            for target in sorted(frontier):
                for source, relation, destination, origin in self.incoming[target]:
                    if origin == candidate:
                        continue
                    edges.add((source, relation, destination))
                    following.add(source)
            frontier = following - nodes
            nodes.update(following)
        edges.update((node, self.self_relation, node) for node in nodes)
        return sorted(nodes), sorted(edges)
