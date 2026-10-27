import math
import random
from collections import Counter

from .types import LabeledPartition


def allocate_counts(total, weights):
    scale = sum(weights)
    if not scale:
        return (0,) * len(weights)
    exact = [total * weight / scale for weight in weights]
    counts = [math.floor(value) for value in exact]
    order = sorted(range(len(weights)), key=lambda i: (-(exact[i] - counts[i]), i))
    for index in order[:total - sum(counts)]:
        counts[index] += 1
    return tuple(counts)


def corrupt_partition(clean, known, entities, relations, config, seed, occupied):
    rng = random.Random(seed)
    total = round(len(clean) * config.noise_ratio)
    if config.dataset.lower() in {'fb15k-237', 'wn18rr', 'nell-995'}:
        total -= total % 2
    targets = allocate_counts(total, (
        config.spurious_fraction, config.substitution_fraction, config.redundant_fraction))
    samples = [(triple, 1, 'valid') for triple in clean]
    for family, target in zip(('entity_corruption', 'relation_substitution', 'redundant_assertion'), targets):
        accepted = 0
        if target and family == 'relation_substitution' and len(relations) < 2:
            raise ValueError('Relation substitution requires at least two relations')
        for _ in range(max(1000, target * 1000)):
            if accepted == target:
                break
            h, r, t = rng.choice(clean)
            if family == 'redundant_assertion':
                candidate = h, r, t
            elif family == 'relation_substitution':
                candidate = h, rng.choice(relations), t
                if candidate[1] == r:
                    continue
            elif rng.getrandbits(1):
                candidate = rng.choice(entities), r, t
                if candidate[0] == h:
                    continue
            else:
                candidate = h, r, rng.choice(entities)
                if candidate[2] == t:
                    continue
            if family != 'redundant_assertion':
                if candidate in known or candidate in occupied:
                    continue
                occupied.add(candidate)
            samples.append((candidate, 0, family))
            accepted += 1
        if accepted != target:
            raise ValueError(f'Cannot generate {target} unique {family} anomalies; obtained {accepted}')
    rng.shuffle(samples)
    return LabeledPartition(*(tuple(column) for column in zip(*samples)))


def build_corrupted_splits(reference, config):
    entities = sorted({v for h, _, t in reference.positives for v in (h, t)})
    relations = sorted({r for _, r, _ in reference.positives})
    occupied = set()
    clean_splits = reference.train, reference.valid, reference.test
    partitions = tuple(corrupt_partition(clean, reference.positives, entities, relations,
        config, config.seed + 1009 * index, occupied) for index, clean in enumerate(clean_splits))
    stats = {name: {'size': len(partition.triples),
        'anomaly_counts': dict(Counter(partition.anomaly_types)),
        'injected_over_clean': (len(partition.triples) - len(clean)) / len(clean)}
        for name, partition, clean in zip(('train', 'valid', 'test'), partitions, clean_splits)}
    return partitions, stats

