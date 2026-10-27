import json
from pathlib import Path

import torch


def ensure_parent_dir(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def save_json(path, payload):
    ensure_parent_dir(path)
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def export_filtered_graph(triples, probabilities, threshold, output_path):
    if len(triples) != len(probabilities):
        raise ValueError('Triple and probability counts differ')
    ensure_parent_dir(output_path)
    kept = dict.fromkeys(triple for triple, probability in zip(triples, probabilities) if probability >= threshold)
    with Path(output_path).open('w', encoding='utf-8') as handle:
        for h, r, t in kept:
            handle.write(f'{h}\t{r}\t{t}\n')
    return len(kept)


def save_checkpoint(path, payload):
    ensure_parent_dir(path)
    torch.save(payload, path)


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
