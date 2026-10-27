from array import array
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass

from .types import TripleId


@dataclass(frozen=True, slots=True)
class TypedArc:
    source: int
    relation: int
    target: int
    origin: int


class RelationalIndex:
    def __init__(self, triples, num_entities: int, num_relations: int, inverse: bool):
        if num_entities < 1 or num_relations < 1:
            raise ValueError('A relational index requires nonempty entity and relation vocabularies')
        self.facts = tuple(sorted(set(tuple(triple) for triple in triples)))
        self.fact_ids = {fact: index for index, fact in enumerate(self.facts)}
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.inverse = inverse
        ordered = []
        for identity, (head, relation, tail) in enumerate(self.facts):
            if not 0 <= head < num_entities or not 0 <= tail < num_entities or not 0 <= relation < num_relations:
                raise ValueError(f'Fact {identity} refers to an identifier outside its vocabulary')
            ordered.append((tail, head, relation, identity))
            ordered.append((head, tail, relation + num_relations if inverse else relation, identity))
        ordered.sort()
        counts = Counter(target for target, _, _, _ in ordered)
        self.offsets = array('q', [0])
        for node in range(num_entities):
            self.offsets.append(self.offsets[-1] + counts[node])
        self.sources = array('q', (source for _, source, _, _ in ordered))
        self.relations = array('q', (relation for _, _, relation, _ in ordered))
        self.origins = array('q', (identity for _, _, _, identity in ordered))
        self.relation_counts = dict(Counter(self.relations))

    def arcs(self, target: int) -> Iterator[TypedArc]:
        if not 0 <= target < self.num_entities:
            raise IndexError(f'Entity {target} is outside the context vocabulary')
        for offset in range(self.offsets[target], self.offsets[target + 1]):
            yield TypedArc(self.sources[offset], self.relations[offset], target, self.origins[offset])

    def degree(self, node: int) -> int:
        if not 0 <= node < self.num_entities:
            raise IndexError(node)
        return self.offsets[node + 1] - self.offsets[node]

    def fact_identity(self, triple: TripleId) -> int | None:
        return self.fact_ids.get(tuple(triple))

    def describe(self) -> dict:
        degrees = [self.degree(node) for node in range(self.num_entities)]
        return {'entities': self.num_entities, 'unique_facts': len(self.facts),
            'directed_arcs_without_self_loops': len(self.sources),
            'isolated_entities': sum(value == 0 for value in degrees),
            'maximum_typed_in_degree': max(degrees, default=0),
            'mean_typed_in_degree': sum(degrees) / self.num_entities,
            'relation_arc_counts': self.relation_counts,
            'packed_array_bytes': sum(values.itemsize * len(values) for values in (
                self.offsets, self.sources, self.relations, self.origins))}
