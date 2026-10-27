from pathlib import Path

from .data import generate_synthetic_clean_triples, load_triples, split_indices
from .types import ReferenceSplits


def load_reference_splits(config):
    if config.dataset_dir:
        root = Path(config.dataset_dir)
        partitions = []
        for aliases in (('train',), ('valid', 'validation', 'dev'), ('test',)):
            paths = [root / f'{name}.{suffix}' for name in aliases for suffix in ('txt', 'tsv')]
            found = next((path for path in paths if path.is_file()), None)
            if found is None:
                raise FileNotFoundError(f'Missing canonical {aliases[0]} partition in {root}')
            partitions.append(tuple(load_triples(str(found))))
    else:
        path = Path(config.clean_graph_path)
        if path.is_file():
            triples = load_triples(str(path))
        elif config.synthetic_if_missing:
            triples = generate_synthetic_clean_triples(config.synthetic_num_entities,
                config.synthetic_num_relations, config.synthetic_num_triples, config.seed)
        else:
            raise FileNotFoundError(f'Reference graph not found: {path}')
        triples = sorted(set(triples))
        partitions = [tuple(triples[i] for i in indices) for indices in split_indices(
            len(triples), config.train_ratio, config.val_ratio, config.test_ratio, config.reference_split_seed)]
    sets = list(map(set, partitions))
    if any(len(partition) != len(unique) for partition, unique in zip(partitions, sets)):
        raise ValueError('Clean reference partitions contain duplicate assertions')
    if any(sets[i] & sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError('Reference partitions overlap')
    return ReferenceSplits(*partitions)

