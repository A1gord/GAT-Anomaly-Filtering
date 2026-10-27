from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class ConfusionCurve:
    thresholds: Tensor
    true_positive: Tensor
    false_positive: Tensor
    true_negative: Tensor
    false_negative: Tensor

    @property
    def precision(self) -> Tensor:
        return self.true_positive.double() / (self.true_positive + self.false_positive).clamp_min(1)

    @property
    def recall(self) -> Tensor:
        return self.true_positive.double() / (self.true_positive + self.false_negative).clamp_min(1)

    @property
    def f1(self) -> Tensor:
        return 2 * self.true_positive.double() / (2 * self.true_positive + self.false_positive + self.false_negative).clamp_min(1)

    def rows(self) -> list[dict]:
        return [{'threshold': float(self.thresholds[index]), 'precision': float(self.precision[index]),
            'recall': float(self.recall[index]), 'f1': float(self.f1[index]),
            'tp': int(self.true_positive[index]), 'fp': int(self.false_positive[index]),
            'tn': int(self.true_negative[index]), 'fn': int(self.false_negative[index])}
            for index in range(self.thresholds.numel())]


def probability_vectors(labels, probabilities) -> tuple[Tensor, Tensor]:
    truth = torch.as_tensor(labels, device='cpu').reshape(-1)
    scores = torch.as_tensor(probabilities, device='cpu').reshape(-1)
    if not truth.numel() or truth.shape != scores.shape:
        raise ValueError('Nonempty labels and probabilities must have the same length')
    if not ((truth == 0) | (truth == 1)).all():
        raise ValueError('Labels must contain only zero and one')
    if not torch.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError('Validity probabilities must be finite and lie in [0, 1]')
    return truth.long(), scores


def confusion_curve(labels, probabilities, thresholds) -> ConfusionCurve:
    truth, scores = probability_vectors(labels, probabilities)
    thresholds = torch.as_tensor(thresholds, device='cpu').reshape(-1)
    if not thresholds.numel() or not torch.isfinite(thresholds).all() or ((thresholds < 0) | (thresholds > 1)).any():
        raise ValueError('Decision thresholds must be finite and lie in [0, 1]')
    scores, order = scores.sort(stable=True)
    anomaly = (truth[order] == 0).long()
    cumulative_anomaly = torch.cat((torch.zeros(1, dtype=torch.long), anomaly.cumsum(0)))
    cumulative_valid = torch.cat((torch.zeros(1, dtype=torch.long), (1 - anomaly).cumsum(0)))
    boundary = torch.searchsorted(scores, thresholds.to(scores.dtype), right=False)
    tp, fp = cumulative_anomaly[boundary], cumulative_valid[boundary]
    return ConfusionCurve(thresholds, tp, fp, cumulative_valid[-1] - fp, cumulative_anomaly[-1] - tp)


def best_curve_index(curve: ConfusionCurve) -> int:
    values, thresholds = curve.f1.tolist(), curve.thresholds.tolist()
    return max(range(len(thresholds)), key=lambda index: (
        values[index], -abs(thresholds[index] - 0.5), -thresholds[index]))
