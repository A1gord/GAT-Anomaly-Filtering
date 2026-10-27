import hashlib
import json
from dataclasses import asdict

from .io_utils import load_checkpoint, save_checkpoint

CHECKPOINT_VERSION = 2


def dataset_fingerprint(reference):
    payload = {'train': reference.train, 'valid': reference.valid, 'test': reference.test}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def write_model_checkpoint(path, model, config, threshold, mappings, context, fingerprint, best_epoch):
    save_checkpoint(path, {'format_version': CHECKPOINT_VERSION,
        'model_state_dict': {key: value.detach().cpu() for key, value in model.state_dict().items()},
        'config': asdict(config), 'best_threshold': threshold, 'best_epoch': best_epoch,
        'ent2id': mappings[0], 'rel2id': mappings[1], 'context_triples': list(context.triples),
        'dataset_fingerprint': fingerprint})


def read_model_checkpoint(path):
    payload = load_checkpoint(path)
    if payload.get('format_version') != CHECKPOINT_VERSION:
        raise ValueError('Incompatible checkpoint; retrain using the relation-aware model')
    for key in ('model_state_dict', 'config', 'ent2id', 'rel2id', 'context_triples', 'best_threshold'):
        if key not in payload:
            raise ValueError(f'Checkpoint is missing {key}')
    return payload
