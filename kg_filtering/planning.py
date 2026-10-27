import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .ablations import ABLATIONS, ablation_config
from .artifacts import with_output_directory
from .config import Config
from .provenance import canonical_digest
from .registry import model_spec


@dataclass(frozen=True, slots=True)
class RunIdentity:
    dataset: str
    noise_ratio: float
    seed: int
    model: str
    ablation: str

    @property
    def key(self) -> str:
        return canonical_digest(asdict(self))[:24]

    def directory(self, root) -> Path:
        dataset = re.sub(r'[^A-Za-z0-9_.-]+', '_', self.dataset).strip('.') or 'custom'
        return Path(root) / dataset / f'noise_{float(self.noise_ratio)}' / f'seed_{self.seed}' / self.model / self.ablation


@dataclass(frozen=True)
class PlannedRun:
    identity: RunIdentity
    config: Config

    def describe(self) -> dict:
        return {'run_id': self.identity.key, **asdict(self.identity), 'output_dir': self.config.output_dir}


class ExperimentPlan:
    def __init__(self, config):
        self.config = config.validate()
        default_ratios = (0.1, 0.2, 0.3, 0.4) if config.dataset.lower() == 'tmf-kg' else (0.05, 0.1, 0.15)
        self.ratios = config.experiment_noise_ratios or ((0.1,) if config.run_ablations else default_ratios)
        if config.run_ablations and not config.entity_features_path:
            raise ValueError('The full ablation suite requires supplied semantic features')
        self.runs = tuple(self._expand())
        keys = [run.identity.key for run in self.runs]
        if len(keys) != len(set(keys)):
            raise ValueError('Experiment axes contain duplicate run identities')

    def _expand(self):
        for ratio in self.ratios:
            for seed in self.config.experiment_seeds:
                for model in self.config.experiment_models:
                    spec = model_spec(model)
                    variants = ABLATIONS if self.config.run_ablations and model == 'proposed' else ('full',)
                    for variant in variants:
                        identity = RunIdentity(self.config.dataset, ratio, seed, model, variant)
                        current = replace(self.config, noise_ratio=ratio, seed=seed, model_name=model)
                        if model == 'proposed':
                            current = ablation_config(current, variant)
                        current = with_output_directory(current, identity.directory(self.config.output_dir))
                        if current.external_scores_path:
                            current = replace(current, external_scores_path=current.external_scores_path.format(
                                dataset=identity.dataset, seed=seed, ratio=f'{ratio:g}', model=model, ablation=variant))
                        current.validate()
                        spec.validate_inputs(current)
                        yield PlannedRun(identity, current)

    def describe(self):
        return {'format_version': 1, 'dataset': self.config.dataset,
            'noise_ratios': list(self.ratios), 'seeds': list(self.config.experiment_seeds),
            'run_count': len(self.runs), 'runs': [run.describe() for run in self.runs]}
