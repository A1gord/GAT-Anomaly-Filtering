import json
from pathlib import Path

import torch

from .storage import atomic_destination, atomic_text, write_json_atomic


def ensure_parent_dir(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def save_json(path, payload):
    write_json_atomic(path, payload)


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def export_filtered_graph(triples, probabilities, threshold, output_path):
    if len(triples) != len(probabilities):
        raise ValueError('Triple and probability counts differ')
    kept = dict.fromkeys(triple for triple, probability in zip(triples, probabilities) if probability >= threshold)
    with atomic_text(output_path) as handle:
        for head, relation, tail in kept:
            handle.write(f'{head}\t{relation}\t{tail}\n')
    return len(kept)


def save_checkpoint(path, payload):
    with atomic_destination(path) as temporary:
        torch.save(payload, temporary)


def load_checkpoint(path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError('Checkpoint must be a dictionary')
    return payload


def save_mappings(path, ent2id, rel2id):
    save_json(path, {'ent2id': ent2id, 'rel2id': rel2id})


def load_mappings(path):
    payload = load_json(path)
    for key in ('ent2id', 'rel2id'):
        values = payload.get(key)
        if not isinstance(values, dict) or sorted(values.values()) != list(range(len(values))):
            raise ValueError(f'Invalid contiguous mapping: {key}')
    return payload['ent2id'], payload['rel2id']
