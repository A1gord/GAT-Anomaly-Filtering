from collections.abc import Sequence

from .cache import BudgetedCache
from .indices import RelationalIndex
from .neighborhood import IncomingExpansion, Neighborhood
from .types import TripleId


class RelationalContext:
    def __init__(self, triples: Sequence[TripleId], num_entities: int, num_relations: int,
                 inverse: bool = True, cache_entries: int = 128, cache_bytes: int = 67108864) -> None:
        self.index = RelationalIndex(triples, num_entities, num_relations, inverse)
        self.triples = self.index.facts
        self.num_entities, self.num_relations = num_entities, num_relations
        self.inverse, self.self_relation = inverse, 2 * num_relations
        self.expansion = IncomingExpansion(self.index)
        self.cache: BudgetedCache[tuple[TripleId, int], Neighborhood] = BudgetedCache(cache_entries, cache_bytes)

    def extract(self, candidate: TripleId, depth: int) -> Neighborhood:
        key = tuple(candidate), depth
        result = self.cache.get(key)
        if result is None:
            result = self.expansion.extract(key[0], depth)
            if self.cache.max_entries and self.cache.max_bytes:
                self.cache.put(key, result, result.accounted_bytes)
        return result

    def neighborhood(self, candidate: TripleId, depth: int) -> tuple[list[int], list[TripleId]]:
        result = self.extract(candidate, depth)
        return list(result.nodes), list(result.edges)

    def describe(self) -> dict:
        return {'index': self.index.describe(), 'cache': self.cache.describe()}
