from collections import defaultdict

import torch

from .decisions import confusion_curve
from .metrics import evaluate_binary
from .score_diagnostics import reliability_report, risk_coverage_report, score_ranking_metrics
from .statistics import bootstrap_metric_intervals


def grouped_metrics(labels, probabilities, keys, threshold, max_groups):
    groups = defaultdict(list)
    for index, key in enumerate(keys):
        groups[str(key)].append(index)
    ordered = sorted(groups, key=lambda key: (-len(groups[key]), key))
    scores = torch.as_tensor(probabilities)
    records = []
    for key in ordered[:max_groups]:
        indices = groups[key]
        selected = [labels[index] for index in indices]
        records.append({'group': key, 'occurrences': len(indices), 'valid': sum(selected),
            'anomalies': len(indices) - sum(selected),
            **evaluate_binary(selected, scores[indices], threshold)})
    return {'groups': records, 'omitted_groups': max(0, len(groups) - max_groups),
        'group_selection': 'descending occurrence support, then identifier'}


def build_diagnostics(partition, probabilities, encoded, context, threshold, config):
    labels = partition.labels
    degrees = [context.index.degree(head) + context.index.degree(tail) for head, _, tail in encoded]
    degree_bands = ['isolated' if value == 0 else f'2^{value.bit_length() - 1}' for value in degrees]
    curves = confusion_curve(labels, probabilities, torch.linspace(config.threshold_min,
        config.threshold_max, config.threshold_steps))
    return {'selection_policy': 'descriptive report only; no test-derived parameter selection',
        'ranking': score_ranking_metrics(labels, probabilities),
        'reliability': reliability_report(labels, probabilities, config.diagnostic_bins),
        'risk_coverage': risk_coverage_report(labels, probabilities, threshold),
        'threshold_curve': curves.rows(),
        'by_relation': grouped_metrics(labels, probabilities, [triple[1] for triple in partition.triples],
            threshold, config.diagnostic_max_groups),
        'by_endpoint_degree_band': grouped_metrics(labels, probabilities, degree_bands, threshold,
            config.diagnostic_max_groups),
        'bootstrap': bootstrap_metric_intervals(labels, probabilities, threshold,
            config.bootstrap_samples, config.confidence_level, config.seed)}
