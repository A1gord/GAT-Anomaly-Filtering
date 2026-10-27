from dataclasses import replace
from pathlib import Path

from .ablations import ABLATIONS, ablation_config
from .artifacts import with_output_directory
from .io_utils import save_json
from .pipeline import run_train_pipeline
from .reporting import summarize_runs


def run_experiments(config):
    config.validate()
    defaults = (0.1, 0.2, 0.3, 0.4) if config.dataset.lower() == 'tmf-kg' else (0.05, 0.1, 0.15)
    ratios = config.experiment_noise_ratios or ((0.1,) if config.run_ablations else defaults)
    if config.run_ablations and not config.entity_features_path:
        raise ValueError('The complete ablation suite requires semantic feature vectors')
    runs, fingerprints = [], {}
    for ratio in ratios:
        for seed in config.experiment_seeds:
            for model in config.experiment_models:
                variants = ABLATIONS if config.run_ablations and model == 'proposed' else {'full': {}}
                for variant in variants:
                    current = replace(config, noise_ratio=ratio, seed=seed, model_name=model)
                    if model == 'proposed':
                        current = ablation_config(current, variant)
                    root = Path(config.output_dir) / config.dataset / f'noise_{ratio:g}' / f'seed_{seed}' / model / variant
                    current = with_output_directory(current, root)
                    if current.external_scores_path:
                        current = replace(current, external_scores_path=current.external_scores_path.format(
                            dataset=config.dataset, seed=seed, ratio=f'{ratio:g}', model=model, ablation=variant))
                    result = run_train_pipeline(current)
                    key, fingerprint = (ratio, seed), result['corruption_fingerprint']
                    if fingerprints.setdefault(key, fingerprint) != fingerprint:
                        raise ValueError('Models received different corrupted candidates')
                    runs.append({'model': model, 'noise_ratio': ratio, 'seed': seed,
                        'ablation': variant, 'test': result['test'],
                        'dataset_fingerprint': result['dataset_fingerprint'],
                        'corruption_fingerprint': fingerprint,
                        'test_by_anomaly_type': result['test_by_anomaly_type'],
                        'completion': result.get('completion'), 'metrics_path': current.output_metrics_path})
                    save_json(str(Path(config.output_dir) / 'experiment_runs.json'), {'runs': runs})
    payload = {'dataset': config.dataset, 'seeds': list(config.experiment_seeds),
        'noise_ratios': list(ratios), 'summary': summarize_runs(runs), 'runs': runs}
    save_json(str(Path(config.output_dir) / 'experiment_summary.json'), payload)
    return payload
