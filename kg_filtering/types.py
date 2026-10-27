from dataclasses import dataclass

from torch import Tensor

TripleId = tuple[int, int, int]
TripleStr = tuple[str, str, str]


@dataclass(frozen=True)
class LabeledPartition:
    triples: tuple[TripleStr, ...]
    labels: tuple[int, ...]
    anomaly_types: tuple[str, ...]

    def __post_init__(self):
        if not len(self.triples) == len(self.labels) == len(self.anomaly_types):
            raise ValueError('Partition columns must have equal lengths')
        if any(label not in (0, 1) for label in self.labels):
            raise ValueError('Labels must be binary')


@dataclass(frozen=True)
class ReferenceSplits:
    train: tuple[TripleStr, ...]
    valid: tuple[TripleStr, ...]
    test: tuple[TripleStr, ...]

    @property
    def positives(self):
        return frozenset(self.train + self.valid + self.test)


@dataclass(frozen=True)
class CandidateBatch:
    node_ids: Tensor
    edge_index: Tensor
    edge_types: Tensor
    candidates: Tensor

    def to(self, device):
        return CandidateBatch(*(value.to(device) for value in (
            self.node_ids, self.edge_index, self.edge_types, self.candidates)))
