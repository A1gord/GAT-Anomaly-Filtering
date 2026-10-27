import json
import math
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

from .registry import MODEL_REGISTRY


@dataclass
class Config:
    mode: str = 'train'
    dataset: str = 'custom'
    dataset_dir: str | None = None
    clean_graph_path: str = 'data/clean_triples.tsv'
    inference_graph_path: str = 'data/inference_triples.tsv'
    output_dir: str = 'outputs'
    output_filtered_path: str = 'outputs/filtered_triples.tsv'
    output_metrics_path: str = 'outputs/metrics.json'
    output_history_path: str = 'outputs/history.json'
    output_dataset_stats_path: str = 'outputs/dataset_stats.json'
    output_scores_path: str = 'outputs/scores.tsv'
    checkpoint_path: str = 'outputs/best_model.pt'
    mappings_path: str = 'outputs/mappings.json'
    entity_features_path: str | None = None
    rules_path: str | None = None
    external_scores_path: str | None = None
    embedding_dim: int = 128
    relation_dim: int = 64
    semantic_dim: int = 0
    num_layers: int = 2
    num_heads: int = 8
    classifier_hidden_dim: int = 256
    batch_size: int = 256
    dropout: float = 0.2
    lr: float = 5e-4
    weight_decay: float = 1e-5
    max_epochs: int = 300
    patience: int = 20
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    noise_ratio: float = 0.1
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    threshold: float = 0.5
    optimize_threshold: bool = True
    threshold_min: float = 0.05
    threshold_max: float = 0.95
    threshold_steps: int = 37
    spurious_fraction: float = 0.5
    substitution_fraction: float = 0.5
    redundant_fraction: float = 0.0
    seed: int = 13
    reference_split_seed: int = 13
    inference_context: str = 'observed'
    use_multi_head: bool = True
    use_gatv2: bool = True
    use_relation_attention: bool = True
    use_relation_messages: bool = True
    use_inverse_relations: bool = True
    use_semantic_features: bool = True
    drop_unknown_in_predict: bool = False
    save_checkpoint: bool = True
    device: str = 'cpu'
    synthetic_if_missing: bool = False
    synthetic_num_entities: int = 500
    synthetic_num_relations: int = 20
    synthetic_num_triples: int = 5000
    model_name: str = 'proposed'
    experiment_seeds: tuple[int, ...] = (13, 37, 71, 101, 137)
    experiment_noise_ratios: tuple[float, ...] = ()
    experiment_models: tuple[str, ...] = ('proposed',)
    run_ablations: bool = False
    run_completion: bool = False
    completion_dim: int = 128
    completion_epochs: int = 300
    completion_batch_size: int = 256
    completion_negative_samples: int = 64
    completion_margin: float = 9.0
    completion_lr: float = 5e-4
    ranking_batch_size: int = 4096
    forest_estimators: int = 300
    context_cache_entries: int = 128
    context_cache_bytes: int = 67108864
    batch_max_nodes: int = 0
    batch_max_edges: int = 0
    gradient_clip_norm: float = 0.0
    record_telemetry: bool = True
    enable_diagnostics: bool = True
    diagnostic_bins: int = 15
    diagnostic_max_groups: int = 128
    bootstrap_samples: int = 0
    confidence_level: float = 0.95
    experiment_resume: bool = True
    explain_candidates: int = 0
    explanation_top_k: int = 8

    def validate(self):
        allowed_models = MODEL_REGISTRY.keys()
        if self.model_name not in allowed_models or any(name not in allowed_models for name in self.experiment_models):
            raise ValueError('Unknown model name')
        if self.mode not in {'train', 'prepare', 'predict', 'experiment'}:
            raise ValueError('Unknown execution mode')
        if not self.experiment_models:
            raise ValueError('experiment_models cannot be empty')
        positive = ('embedding_dim', 'relation_dim', 'num_layers', 'num_heads',
                    'classifier_hidden_dim', 'batch_size', 'max_epochs', 'patience',
                    'scheduler_patience', 'threshold_steps', 'completion_dim',
                    'completion_epochs', 'completion_batch_size', 'completion_negative_samples',
                    'ranking_batch_size', 'forest_estimators', 'diagnostic_bins', 'diagnostic_max_groups', 'explanation_top_k')
        for name in positive:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        nonnegative = ('context_cache_entries', 'context_cache_bytes', 'batch_max_nodes',
            'batch_max_edges', 'bootstrap_samples', 'explain_candidates')
        for name in nonnegative:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        if not math.isfinite(self.gradient_clip_norm) or self.gradient_clip_norm < 0:
            raise ValueError('gradient_clip_norm must be finite and nonnegative')
        if not 0 < self.confidence_level < 1:
            raise ValueError('confidence_level must lie strictly between zero and one')
        if self.semantic_dim < 0:
            raise ValueError('semantic_dim must be nonnegative')
        for name in ('lr', 'completion_lr', 'completion_margin'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        for name in ('noise_ratio', 'weight_decay', 'spurious_fraction', 'substitution_fraction', 'redundant_fraction'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if not 0 <= self.dropout < 1 or not 0 < self.scheduler_factor < 1:
            raise ValueError('Invalid dropout or scheduler factor')
        if any(not 0 < x < 1 for x in (self.train_ratio, self.val_ratio, self.test_ratio)):
            raise ValueError('Split ratios must lie strictly between zero and one')
        if not math.isclose(self.train_ratio + self.val_ratio + self.test_ratio, 1.0):
            raise ValueError('Split ratios must sum to one')
        if not 0 <= self.threshold_min <= self.threshold_max <= 1 or not 0 <= self.threshold <= 1:
            raise ValueError('Invalid threshold interval')
        heads = self.num_heads if self.use_multi_head else 1
        if self.num_layers > 1 and self.embedding_dim % heads:
            raise ValueError('embedding_dim must be divisible by the effective head count')
        if self.noise_ratio and self.spurious_fraction + self.substitution_fraction + self.redundant_fraction <= 0:
            raise ValueError('At least one corruption family must have positive weight')
        if self.dataset.lower() in {'fb15k-237', 'wn18rr', 'nell-995'}:
            if not self.dataset_dir:
                raise ValueError('Public benchmarks require canonical partitions in dataset_dir')
            if self.redundant_fraction or not math.isclose(self.spurious_fraction, self.substitution_fraction):
                raise ValueError('Public benchmarks require equal entity/relation corruption and no duplicates')
        if self.inference_context not in ('observed', 'training'):
            raise ValueError('inference_context must be observed or training')
        if not self.experiment_seeds:
            raise ValueError('experiment_seeds cannot be empty')
        for value in (self.seed, self.reference_split_seed, *self.experiment_seeds):
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2 ** 63:
                raise ValueError('Seeds must be integers in [0, 2**63)')
        if len(set(self.experiment_seeds)) != len(self.experiment_seeds):
            raise ValueError('Experiment seeds must be unique')
        if len(set(self.experiment_models)) != len(self.experiment_models):
            raise ValueError('Experiment models must be unique')
        if len(set(self.experiment_noise_ratios)) != len(self.experiment_noise_ratios):
            raise ValueError('Experiment noise ratios must be unique')
        if any(not math.isfinite(x) or x < 0 for x in self.experiment_noise_ratios):
            raise ValueError('Invalid experiment noise ratios')
        return self


def apply_overrides(config, overrides):
    allowed = {field.name for field in fields(Config)}
    if unknown := overrides.keys() - allowed:
        raise ValueError(f'Unknown configuration fields: {sorted(unknown)}')
    values = {key: value for key, value in overrides.items() if value is not None}
    for name in ('experiment_seeds', 'experiment_noise_ratios', 'experiment_models'):
        if name in values:
            values[name] = tuple(values[name])
    return replace(config, **values)


def load_config(path):
    return Config() if path is None else apply_overrides(Config(), json.loads(Path(path).read_text(encoding='utf-8-sig')))


def config_to_dict(config):
    return asdict(config)
