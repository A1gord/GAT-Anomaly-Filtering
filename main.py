import argparse
import json
from dataclasses import asdict, fields

from kg_filtering.config import Config, apply_overrides, load_config
from kg_filtering.experiments import run_experiments
from kg_filtering.pipeline import run_predict_pipeline, run_prepare_pipeline, run_train_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description='Relation-aware knowledge graph structure restoration')
    parser.add_argument('--config')
    defaults = Config()
    for field in fields(Config):
        value = getattr(defaults, field.name)
        name = '--' + field.name.replace('_', '-')
        options = {'dest': field.name, 'default': None}
        if isinstance(value, bool):
            options.update(type=int, choices=(0, 1))
        elif isinstance(value, tuple):
            options.update(nargs='+', type={'experiment_seeds': int, 'experiment_noise_ratios': float,
                'experiment_models': str}[field.name])
        else:
            options['type'] = type(value) if value is not None else str
        parser.add_argument(name, **options)
    for alias, destination in (('--d', 'embedding_dim'), ('--L', 'num_layers'), ('--K', 'num_heads'), ('--sigma', 'noise_ratio')):
        parser.add_argument(alias, dest=destination, type=float if destination == 'noise_ratio' else int,
            default=argparse.SUPPRESS)
    return parser.parse_args()


def main():
    args = vars(parse_args())
    config_path = args.pop('config')
    defaults = Config()
    overrides = {key: bool(value) if isinstance(getattr(defaults, key), bool) else value
        for key, value in args.items() if value is not None}
    config = apply_overrides(load_config(config_path), overrides).validate()
    handlers = {'train': run_train_pipeline, 'predict': run_predict_pipeline,
        'prepare': run_prepare_pipeline, 'experiment': run_experiments}
    if config.mode not in handlers:
        raise ValueError(f'Unknown mode: {config.mode}')
    print(json.dumps(asdict(config), indent=2))
    handlers[config.mode](config)


if __name__ == '__main__':
    main()
