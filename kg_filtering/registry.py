from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ModelSpec:
    name: str
    backend: str
    relational: bool
    neural_checkpoint: bool
    required_path: str | None = None

    def validate_inputs(self, config) -> None:
        if self.required_path:
            value = getattr(config, self.required_path)
            if not value or not Path(value).is_file():
                raise FileNotFoundError(f'{self.name} requires an existing {self.required_path}')


MODEL_REGISTRY = {
    spec.name: spec for spec in (
        ModelSpec('proposed', 'neural', True, True),
        ModelSpec('transe', 'neural', False, True),
        ModelSpec('gcn', 'neural', False, True),
        ModelSpec('gatv2', 'neural', False, True),
        ModelSpec('rgcn', 'neural', True, True),
        ModelSpec('compgcn', 'neural', True, True),
        ModelSpec('random_forest', 'forest', False, False),
        ModelSpec('rule_based', 'constraints', True, False, 'rules_path'),
        ModelSpec('caged', 'external', True, False, 'external_scores_path'),
    )
}


def model_spec(name: str) -> ModelSpec:
    try:
        return MODEL_REGISTRY[name]
    except KeyError as error:
        raise ValueError(f'Unknown model {name}; available models: {sorted(MODEL_REGISTRY)}') from error
