import torch


def compute_sample_weights(labels):
    counts = torch.bincount(labels.long(), minlength=2).clamp_min(1)
    return labels.numel() / (2 * counts[labels.long()].to(labels.dtype))


def precision_recall_f1(y_true, y_pred):
    true, pred = y_true.bool(), y_pred.bool()
    tp = int((true & pred).sum())
    fp = int((~true & pred).sum())
    fn = int((true & ~pred).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return precision, recall, f1


def evaluate_binary(labels, probabilities, threshold):
    labels = torch.as_tensor(labels).cpu()
    probabilities = torch.as_tensor(probabilities).cpu()
    if labels.shape != probabilities.shape or not labels.numel():
        raise ValueError('Metrics require equally sized nonempty vectors')
    if not torch.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError('Validity probabilities must lie in [0, 1]')
    predictions = probabilities >= threshold
    precision, recall, f1 = precision_recall_f1(labels == 0, ~predictions)
    valid_p, valid_r, valid_f1 = precision_recall_f1(labels == 1, predictions)
    return {'precision': precision, 'recall': recall, 'f1': f1,
        'valid_precision': valid_p, 'valid_recall': valid_r, 'valid_f1': valid_f1,
        'accuracy': float((predictions == labels.bool()).float().mean()),
        'tp': int(((labels == 0) & ~predictions).sum()),
        'fp': int(((labels == 1) & ~predictions).sum()),
        'tn': int(((labels == 1) & predictions).sum()),
        'fn': int(((labels == 0) & predictions).sum())}


def find_best_threshold(labels, probabilities, threshold_min, threshold_max, threshold_steps):
    thresholds = torch.linspace(threshold_min, threshold_max, threshold_steps).tolist()
    scored = [(threshold, evaluate_binary(labels, probabilities, threshold)) for threshold in thresholds]
    return max(scored, key=lambda pair: (pair[1]['f1'], -abs(pair[0] - 0.5), -pair[0]))


def select_threshold(labels, probabilities, config):
    if config.optimize_threshold:
        return find_best_threshold(labels, probabilities, config.threshold_min,
            config.threshold_max, config.threshold_steps)[0]
    return config.threshold
