from dataclasses import replace
from pathlib import Path

from .config import Config

ARTIFACT_NAMES = {
    'output_filtered_path': 'filtered_triples.tsv',
    'output_metrics_path': 'metrics.json',
    'output_history_path': 'history.json',
    'output_dataset_stats_path': 'dataset_stats.json',
    'output_scores_path': 'scores.tsv',
    'checkpoint_path': 'best_model.pt',
    'mappings_path': 'mappings.json',
}


def with_output_directory(config: Config, directory: str | Path) -> Config:
    root = Path(directory)
    return replace(config, output_dir=str(root), **{
        field: str(root / filename) for field, filename in ARTIFACT_NAMES.items()})


def resolve_artifact_paths(config: Config) -> Config:
    defaults = Config()
    return replace(config, **{field: str(Path(config.output_dir) / filename)
        for field, filename in ARTIFACT_NAMES.items()
        if getattr(config, field) == getattr(defaults, field)})
