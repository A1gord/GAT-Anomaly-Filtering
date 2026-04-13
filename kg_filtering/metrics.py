from typing import Dict, Tuple

import torch
from torch import Tensor


def compute_sample_weights(labels: Tensor) -> Tensor:
    n = labels.numel()
    positives = labels.sum().item()
    negatives = float(n) - positives
    if positives == 0.0 or negatives == 0.0:
        return torch.ones_like(labels)
    w_pos = labels.new_tensor(float(n) / (2.0 * positives))
    w_neg = labels.new_tensor(float(n) / (2.0 * negatives))
    return torch.where(labels > 0.5, w_pos, w_neg)


def precision_recall_f1(y_true: Tensor, y_pred: Tensor) -> Tuple[float, float, float]:
    y_true_int = y_true.long()
    y_pred_int = y_pred.long()
    tp = int(((y_true_int == 1) & (y_pred_int == 1)).sum().item())
    fp = int(((y_true_int == 0) & (y_pred_int == 1)).sum().item())
    fn = int(((y_true_int == 1) & (y_pred_int == 0)).sum().item())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1


def evaluate_binary(
    labels: Tensor, probabilities: Tensor, threshold: float
) -> Dict[str, float]:
    predictions = (probabilities >= threshold).float()
    precision, recall, f1 = precision_recall_f1(labels, predictions)
    return {"precision": precision, "recall": recall, "f1": f1}


def find_best_threshold(
    labels: Tensor,
    probabilities: Tensor,
    threshold_min: float,
    threshold_max: float,
    threshold_steps: int,
) -> Tuple[float, Dict[str, float]]:
    if threshold_steps < 2:
        metrics = evaluate_binary(labels, probabilities, threshold_min)
        return threshold_min, metrics
    thresholds = torch.linspace(
        threshold_min, threshold_max, steps=threshold_steps, device=probabilities.device
    )
    best_threshold = float(thresholds[0].item())
    best_metrics = evaluate_binary(labels, probabilities, best_threshold)
    best_f1 = best_metrics["f1"]
    for threshold in thresholds[1:]:
        value = float(threshold.item())
        metrics = evaluate_binary(labels, probabilities, value)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = value
            best_metrics = metrics
    return best_threshold, best_metrics
