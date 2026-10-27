import csv
from pathlib import Path

import torch

from .io_utils import ensure_parent_dir
from .storage import atomic_text
from .metrics import evaluate_binary


def anomaly_breakdown(partition, probabilities, threshold):
    kinds = sorted(set(partition.anomaly_types) - {'valid'})
    result = {}
    for kind in kinds:
        indices = [index for index, value in enumerate(partition.anomaly_types) if value in ('valid', kind)]
        result[kind] = evaluate_binary([partition.labels[index] for index in indices],
            torch.as_tensor(probabilities)[indices], threshold)
    return result


def export_candidates(root, partitions):
    root = Path(root)
    for name, partition in partitions.items():
        path = root / f'{name}.tsv'
        ensure_parent_dir(str(path))
        with atomic_text(path, newline='') as handle:
            writer = csv.writer(handle, delimiter='\t')
            writer.writerow(('row', 'head', 'relation', 'tail', 'label', 'anomaly_type'))
            for index, (triple, label, kind) in enumerate(zip(partition.triples, partition.labels, partition.anomaly_types)):
                writer.writerow((index, *triple, label, kind))


def export_scores(path, partitions, probabilities, threshold):
    ensure_parent_dir(path)
    with atomic_text(path, newline='') as handle:
        writer = csv.writer(handle, delimiter='\t')
        writer.writerow(('partition', 'row', 'head', 'relation', 'tail', 'valid_probability', 'retained_by_score'))
        for name, triples in partitions.items():
            if len(triples) != len(probabilities[name]):
                raise ValueError('Score and triple counts differ')
            for index, (triple, probability) in enumerate(zip(triples, probabilities[name])):
                writer.writerow((name, index, *triple, float(probability), int(probability >= threshold)))


def restoration_counts(triples, probabilities, threshold):
    selected = [triple for triple, probability in zip(triples, probabilities) if probability >= threshold]
    return {'retained_by_score': len(selected), 'unique_retained': len(set(selected)),
        'duplicate_occurrences_removed': len(selected) - len(set(selected))}


def aggregate_values(values):
    if not values:
        return {'mean': None, 'std': None, 'count': 0}
    tensor = torch.tensor(values, dtype=torch.float64)
    return {'mean': float(tensor.mean()),
        'std': float(tensor.std(unbiased=True)) if len(values) > 1 else 0.0,
        'count': len(values)}


def summarize_runs(runs):
    grouped = {}
    for run in runs:
        key = run['model'], run['noise_ratio'], run['ablation']
        grouped.setdefault(key, []).append(run)
    summaries = []
    for (model, ratio, ablation), group in grouped.items():
        summary = {'model': model, 'noise_ratio': ratio, 'ablation': ablation, 'runs': len(group)}
        summary.update({metric: aggregate_values([run['test'][metric] for run in group])
            for metric in ('precision', 'recall', 'f1')})
        families = sorted({kind for run in group for kind in run.get('test_by_anomaly_type', {})})
        summary['test_by_anomaly_type'] = {kind: {metric: aggregate_values([
            run['test_by_anomaly_type'][kind][metric] for run in group
            if kind in run.get('test_by_anomaly_type', {})])
            for metric in ('precision', 'recall', 'f1')} for kind in families}
        completion = [run['completion'] for run in group if run.get('completion')]
        if completion:
            summary['completion'] = {condition: {metric: aggregate_values([
                result[condition]['test'][metric] for result in completion])
                for metric in ('mrr', 'hits_at_1', 'hits_at_3', 'hits_at_10')}
                for condition in ('clean', 'noisy', 'restored')}
            summary['completion']['relative_mrr_gain_percent'] = aggregate_values([
                result['relative_mrr_gain_percent'] for result in completion
                if result['relative_mrr_gain_percent'] is not None])
        summaries.append(summary)
    return summaries
