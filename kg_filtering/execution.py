from pathlib import Path
from time import perf_counter

from .io_utils import save_json
from .ledger import RunLedger, study_lock
from .pipeline import run_train_pipeline
from .planning import ExperimentPlan
from .provenance import execution_signature
from .reporting import summarize_runs
from .statistics import paired_seed_comparisons


class ExperimentRunner:
    def __init__(self, config):
        self.config = config
        self.plan = ExperimentPlan(config)
        self.root = Path(config.output_dir)
        self.completed = []
        self.fingerprints = {}

    def _verify_candidates(self, identity, result):
        key = identity.noise_ratio, identity.seed
        fingerprint = result['corruption_fingerprint']
        if self.fingerprints.setdefault(key, fingerprint) != fingerprint:
            raise ValueError('Compared models received different corrupted candidates')
        reference_key = 'reference', identity.seed
        reference = result['dataset_fingerprint']
        if self.fingerprints.setdefault(reference_key, reference) != reference:
            raise ValueError('Experiment runs received different clean reference partitions')

    def _record(self, planned, result, reused, elapsed):
        identity = planned.identity
        self._verify_candidates(identity, result)
        self.completed.append({'run_id': identity.key, 'model': identity.model,
            'noise_ratio': identity.noise_ratio, 'seed': identity.seed, 'ablation': identity.ablation,
            'test': result['test'], 'dataset_fingerprint': result['dataset_fingerprint'],
            'corruption_fingerprint': result['corruption_fingerprint'],
            'test_by_anomaly_type': result['test_by_anomaly_type'], 'completion': result.get('completion'),
            'metrics_path': planned.config.output_metrics_path, 'reused': reused, 'elapsed_seconds': elapsed})
        save_json(self.root / 'experiment_runs.json', {'runs': self.completed})

    def _execute(self, planned, ledger):
        signature, inventory = execution_signature(planned.config)
        started = perf_counter()
        result = ledger.reusable(planned.identity, signature) if self.config.experiment_resume else None
        reused = result is not None
        if not reused:
            ledger.begin(planned.identity, signature, inventory)
            try:
                result = run_train_pipeline(planned.config)
                final_signature, _ = execution_signature(planned.config)
                if final_signature != signature:
                    raise RuntimeError('Experiment inputs or implementation changed during execution')
                self._verify_candidates(planned.identity, result)
                ledger.complete(planned.identity, planned.config, result)
            except BaseException as error:
                ledger.fail(planned.identity, error)
                raise
        self._record(planned, result, reused, perf_counter() - started)

    def run(self):
        with study_lock(self.root):
            save_json(self.root / 'experiment_plan.json', self.plan.describe())
            ledger = RunLedger(self.root)
            for planned in self.plan.runs:
                self._execute(planned, ledger)
            payload = {'dataset': self.config.dataset, 'seeds': list(self.config.experiment_seeds),
                'noise_ratios': list(self.plan.ratios), 'summary': summarize_runs(self.completed),
                'paired_comparisons': paired_seed_comparisons(self.completed, self.config.bootstrap_samples,
                    self.config.confidence_level, self.config.seed), 'runs': self.completed}
            save_json(self.root / 'experiment_summary.json', payload)
            return payload
