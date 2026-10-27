from pathlib import Path

from .provenance import digest_file


def run_artifact_inventory(config, result):
    root = Path(config.output_dir)
    paths = {'metrics': config.output_metrics_path, 'scores': config.output_scores_path,
        'history': config.output_history_path, 'dataset_stats': config.output_dataset_stats_path,
        'filtered_graph': config.output_filtered_path, 'mappings': config.mappings_path}
    for partition in ('train', 'valid', 'test'):
        for directory in ('reference', 'candidates', 'restored'):
            paths[f'{directory}_{partition}'] = root / directory / f'{partition}.tsv'
    for field in ('diagnostics_path', 'explanations_path'):
        if result.get(field):
            paths[field] = result[field]
    if result['checkpoint_available']:
        paths['checkpoint'] = config.checkpoint_path
    if result.get('batching') is not None and config.record_telemetry:
        paths['training_events'] = root / 'training_events.jsonl'
    inventory = {}
    for name, path in paths.items():
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f'Completed run is missing its {name} artifact: {path}')
        inventory[name] = {'path': str(path.resolve()), 'sha256': digest_file(path), 'bytes': path.stat().st_size}
    return inventory
