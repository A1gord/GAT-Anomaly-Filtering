import random
from pathlib import Path


def parse_triple_line(line):
    parts = line.rstrip('\r\n').split('\t')
    if len(parts) == 1:
        parts = line.split()
    if len(parts) != 3 or any(not part for part in parts):
        raise ValueError('Each triple must contain exactly head, relation, tail')
    return tuple(parts)


def load_triples(path):
    triples = []
    with Path(path).open(encoding='utf-8-sig') as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    triples.append(parse_triple_line(line))
                except ValueError as error:
                    raise ValueError(f'{path}:{number}: {error}') from error
    if not triples:
        raise ValueError(f'Triple file is empty: {path}')
    return triples


def build_entity_relation_vocab(triples):
    return sorted({v for h, _, t in triples for v in (h, t)}), sorted({r for _, r, _ in triples})


def build_mappings(entities, relations):
    ent2id = {value: index for index, value in enumerate(entities)}
    rel2id = {value: index for index, value in enumerate(relations)}
    return ent2id, rel2id, dict(enumerate(entities)), dict(enumerate(relations))


def encode_triples(triples, ent2id, rel2id):
    return [(ent2id[h], rel2id[r], ent2id[t]) for h, r, t in triples]


def encode_triples_with_mask(triples, ent2id, rel2id, drop_unknown):
    encoded, indices = [], []
    for index, (h, r, t) in enumerate(triples):
        if h not in ent2id or t not in ent2id or r not in rel2id:
            if drop_unknown:
                continue
            raise ValueError(f'Unknown token at row {index + 1}: {(h, r, t)}')
        encoded.append((ent2id[h], rel2id[r], ent2id[t]))
        indices.append(index)
    return encoded, indices


def split_indices(num_items, train_ratio, val_ratio, test_ratio, seed):
    if num_items < 3:
        raise ValueError('At least three unique reference triples are required')
    indices = list(range(num_items))
    random.Random(seed).shuffle(indices)
    train = min(max(1, int(num_items * train_ratio)), num_items - 2)
    valid = min(max(1, round(num_items * val_ratio)), num_items - train - 1)
    return indices[:train], indices[train:train + valid], indices[train + valid:]


def generate_synthetic_clean_triples(num_entities, num_relations, num_triples, seed):
    if min(num_entities, num_relations, num_triples) < 1 or num_triples > num_entities ** 2 * num_relations:
        raise ValueError('Invalid synthetic graph dimensions')
    ids = random.Random(seed).sample(range(num_entities ** 2 * num_relations), num_triples)
    return [(f'e{x // (num_relations * num_entities)}', f'r{(x // num_entities) % num_relations}',
             f'e{x % num_entities}') for x in ids]
