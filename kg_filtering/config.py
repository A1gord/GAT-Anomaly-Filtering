import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

import torch


@dataclass
class Config:
    mode: str = "train"
    clean_graph_path: str = "data/clean_triples.tsv"
    inference_graph_path: str = "data/inference_triples.tsv"
    output_filtered_path: str = "outputs/filtered_triples.tsv"
    output_metrics_path: str = "outputs/metrics.json"
    output_history_path: str = "outputs/history.json"
    output_dataset_stats_path: str = "outputs/dataset_stats.json"
    checkpoint_path: str = "outputs/best_model.pt"
    mappings_path: str = "outputs/mappings.json"
    embedding_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    classifier_hidden_dim: int = 256
    dropout: float = 0.2
    lr: float = 5e-4
    weight_decay: float = 1e-5
    max_epochs: int = 300
    patience: int = 20
    noise_ratio: float = 0.2
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    threshold: float = 0.5
    optimize_threshold: bool = True
    threshold_min: float = 0.05
    threshold_max: float = 0.95
    threshold_steps: int = 37
    spurious_fraction: float = 0.34
    substitution_fraction: float = 0.33
    redundant_fraction: float = 0.33
    seed: int = 42
    use_multi_head: bool = True
    use_gatv2: bool = True
    use_edge_weight_awareness: bool = False
    drop_unknown_in_predict: bool = True
    save_checkpoint: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    synthetic_if_missing: bool = True
    synthetic_num_entities: int = 500
    synthetic_num_relations: int = 20
    synthetic_num_triples: int = 5000


def load_config(path: str | None) -> Config:
    cfg = Config()
    if path is None:
        return cfg
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    for key, value in payload.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def apply_overrides(config: Config, overrides: Dict[str, Any]) -> Config:
    for key, value in overrides.items():
        if value is None:
            continue
        if hasattr(config, key):
            setattr(config, key, value)
    return config


def config_to_dict(config: Config) -> Dict[str, Any]:
    return asdict(config)
