import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from .environment import runtime_descriptor


def digest_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while block := handle.read(1048576):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(payload) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False,
        separators=(',', ':')).encode()
    return hashlib.sha256(encoded).hexdigest()


def source_inventory(config) -> dict:
    paths = {}
    if config.dataset_dir:
        root = Path(config.dataset_dir)
        for label, aliases in (('train', ('train',)), ('valid', ('valid', 'validation', 'dev')), ('test', ('test',))):
            candidates = [root / f'{name}.{suffix}' for name in aliases for suffix in ('txt', 'tsv')]
            selected = next((path for path in candidates if path.is_file()), None)
            if selected is None:
                raise FileNotFoundError(f'Missing {label} source partition in {root}')
            paths[f'reference_{label}'] = selected
    elif Path(config.clean_graph_path).is_file():
        paths['reference'] = Path(config.clean_graph_path)
    elif not config.synthetic_if_missing:
        raise FileNotFoundError(config.clean_graph_path)
    for label in ('entity_features_path', 'rules_path', 'external_scores_path'):
        if label == 'rules_path' and config.model_name != 'rule_based':
            continue
        if label == 'external_scores_path' and config.model_name != 'caged':
            continue
        value = getattr(config, label)
        if value:
            paths[label] = Path(value)
    return {name: {'path': str(path.resolve()), 'sha256': digest_file(path), 'bytes': path.stat().st_size}
        for name, path in sorted(paths.items())}


def execution_signature(config) -> tuple[str, dict]:
    inventory = source_inventory(config)
    settings = asdict(config)
    settings.pop('experiment_resume', None)
    source_root = Path(__file__).parent
    source_hashes = {path.name: digest_file(path) for path in sorted(source_root.glob('*.py'))}
    return canonical_digest({'config': settings, 'inputs': inventory, 'source': source_hashes, 'runtime': runtime_descriptor()}), inventory
