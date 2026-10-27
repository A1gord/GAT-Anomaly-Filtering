import torch


def bootstrap_metric_intervals(labels, probabilities, threshold, samples, confidence, seed):
    if samples < 1:
        return {'samples': 0, 'intervals': {}}
    truth = torch.as_tensor(labels).long().cpu()
    predicted_anomaly = torch.as_tensor(probabilities).cpu() < threshold
    actual_anomaly = truth == 0
    contributions = torch.stack((actual_anomaly & predicted_anomaly,
        ~actual_anomaly & predicted_anomaly, actual_anomaly & ~predicted_anomaly), 1).double()
    generator = torch.Generator().manual_seed(seed)
    values = []
    for _ in range(samples):
        selection = torch.randint(len(truth), (len(truth),), generator=generator)
        tp, fp, fn = contributions[selection].sum(0)
        values.append(torch.stack((tp / (tp + fp).clamp_min(1), tp / (tp + fn).clamp_min(1),
            2 * tp / (2 * tp + fp + fn).clamp_min(1))))
    distribution = torch.stack(values)
    quantiles = torch.quantile(distribution, torch.tensor([(1 - confidence) / 2, (1 + confidence) / 2],
        dtype=torch.float64), dim=0)
    return {'samples': samples, 'confidence': confidence, 'method': 'occurrence_percentile_bootstrap',
        'threshold_refitted': False, 'intervals': {name: {'lower': float(quantiles[0, index]),
            'upper': float(quantiles[1, index])} for index, name in enumerate(('precision', 'recall', 'f1'))}}


def paired_seed_comparisons(runs, samples=0, confidence=0.95, seed=13):
    grouped = {}
    for run in runs:
        grouped.setdefault((run['noise_ratio'], run['model'], run['ablation']), {})[run['seed']] = run
    results = []
    for (ratio, model, ablation), candidates in grouped.items():
        reference = grouped.get((ratio, 'proposed', 'full'))
        if reference is None or (model, ablation) == ('proposed', 'full'):
            continue
        common = sorted(reference.keys() & candidates.keys())
        if not common:
            continue
        if any(reference[value]['corruption_fingerprint'] != candidates[value]['corruption_fingerprint'] for value in common):
            raise ValueError('Paired comparisons require matching corrupted candidates')
        delta = torch.tensor([reference[value]['test']['f1'] - candidates[value]['test']['f1']
            for value in common], dtype=torch.float64)
        result = {'noise_ratio': ratio, 'reference': 'proposed/full', 'compared_model': model,
            'compared_ablation': ablation, 'seeds': common, 'mean_f1_difference': float(delta.mean()),
            'positive_differences': int((delta > 0).sum()), 'ties': int((delta == 0).sum()),
            'negative_differences': int((delta < 0).sum())}
        if samples and len(common) > 1:
            generator = torch.Generator().manual_seed(seed)
            distribution = torch.stack([delta[torch.randint(len(delta), (len(delta),), generator=generator)].mean()
                for _ in range(samples)])
            lower, upper = torch.quantile(distribution, torch.tensor([(1 - confidence) / 2, (1 + confidence) / 2],
                dtype=torch.float64))
            result['paired_seed_bootstrap'] = {'lower': float(lower), 'upper': float(upper),
                'samples': samples, 'confidence': confidence}
        results.append(result)
    return results
