import torch

from .decisions import probability_vectors


def score_ranking_metrics(labels, probabilities):
    truth, validity = probability_vectors(labels, probabilities)
    anomaly, order = (1 - validity).sort(descending=True, stable=True)
    positive = (truth[order] == 0).double()
    total_positive, total_negative = int(positive.sum()), int((1 - positive).sum())
    if total_positive == 0:
        return {'anomaly_auroc': None, 'anomaly_average_precision': None, 'reason': 'no_anomalies'}
    ends = torch.cat((torch.nonzero(anomaly[1:] != anomaly[:-1]).flatten(), torch.tensor([len(anomaly) - 1])))
    tp = positive.cumsum(0)[ends]
    fp = (1 - positive).cumsum(0)[ends]
    recall, precision = tp / total_positive, tp / (tp + fp)
    recall_increment = recall - torch.cat((recall.new_zeros(1), recall[:-1]))
    average_precision = float((recall_increment * precision).sum())
    auroc = None
    if total_negative:
        tpr = torch.cat((recall.new_zeros(1), recall))
        fpr = torch.cat((fp.new_zeros(1), fp / total_negative))
        auroc = float(torch.trapz(tpr, fpr))
    return {'anomaly_auroc': auroc, 'anomaly_average_precision': average_precision,
        'tie_policy': 'equal_score_groups', 'anomaly_support': total_positive, 'valid_support': total_negative}


def reliability_report(labels, probabilities, bins):
    truth, scores = probability_vectors(labels, probabilities)
    values = scores.double()
    targets = truth.double()
    assignments = (values * bins).long().clamp(max=bins - 1)
    records, ece = [], 0.0
    for index in range(bins):
        mask = assignments == index
        count = int(mask.sum())
        if count:
            mean_probability = float(values[mask].mean())
            valid_frequency = float(targets[mask].mean())
            gap = abs(mean_probability - valid_frequency)
            ece += gap * count / len(truth)
        else:
            mean_probability, valid_frequency, gap = None, None, None
        records.append({'bin': index, 'lower': index / bins, 'upper': (index + 1) / bins,
            'count': count, 'mean_validity': mean_probability, 'observed_valid_fraction': valid_frequency,
            'absolute_gap': gap})
    bounded = values.clamp(min=1e-12, max=1 - 1e-12)
    return {'brier_score': float((values - targets).square().mean()),
        'negative_log_likelihood': float(-(targets * bounded.log() + (1 - targets) * (1 - bounded).log()).mean()),
        'expected_calibration_error': ece, 'probability_class': 'valid', 'bins': records,
        'interpretation': 'descriptive; inverse-frequency weighted training need not yield calibrated probabilities'}


def risk_coverage_report(labels, probabilities, threshold, steps=10):
    truth, scores = probability_vectors(labels, probabilities)
    confidence = torch.maximum(scores, 1 - scores)
    _, order = confidence.sort(descending=True, stable=True)
    error = ((scores >= threshold) != truth.bool())[order].double().cumsum(0)
    points = []
    for step in range(1, steps + 1):
        count = max(1, (len(truth) * step + steps - 1) // steps)
        points.append({'coverage': count / len(truth), 'retained_occurrences': count,
            'classification_error': float(error[count - 1] / count),
            'minimum_confidence': float(confidence[order[count - 1]])})
    return {'ordering': 'max(valid_probability, 1 - valid_probability)', 'points': points}


def probability_profile(probabilities, bins=15):
    values = torch.as_tensor(probabilities).cpu().double().reshape(-1)
    if not values.numel() or not torch.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError('A probability profile requires finite values in [0, 1]')
    histogram = torch.bincount((values * bins).long().clamp(max=bins - 1), minlength=bins)
    bounded = values.clamp(min=1e-12, max=1 - 1e-12)
    entropy = -(bounded * bounded.log() + (1 - bounded) * (1 - bounded).log())
    quantiles = torch.tensor([0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0], dtype=torch.float64)
    return {'count': len(values), 'mean_validity': float(values.mean()),
        'mean_binary_entropy_nats': float(entropy.mean()),
        'quantiles': {f'{float(level):g}': float(value) for level, value in
            zip(quantiles, torch.quantile(values, quantiles))},
        'histogram': [{'lower': index / bins, 'upper': (index + 1) / bins,
            'count': int(histogram[index])} for index in range(bins)]}
