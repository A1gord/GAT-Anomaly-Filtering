from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import Tensor

from .config import Config
from .metrics import compute_sample_weights, evaluate_binary, find_best_threshold
from .model import EdgeValidityModel


@dataclass
class TrainingOutput:
    best_threshold: float
    best_val_f1: float
    history: List[Dict[str, float]]
    val_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    all_probabilities: List[float]
    train_loss: float


def predict_probabilities(
    model: EdgeValidityModel,
    triples: Tensor,
    edge_index: Tensor,
    edge_weight: Tensor | None,
) -> Tensor:
    model.eval()
    with torch.no_grad():
        logits = model(triples, edge_index, edge_weight)
        probs = torch.sigmoid(logits)
    return probs


def evaluate_model(
    model: EdgeValidityModel,
    triples: Tensor,
    labels: Tensor,
    edge_index: Tensor,
    edge_weight: Tensor | None,
    threshold: float,
) -> Dict[str, float]:
    probs = predict_probabilities(model, triples, edge_index, edge_weight)
    return evaluate_binary(labels, probs, threshold)


def train_model(
    model: EdgeValidityModel,
    optimizer: torch.optim.Optimizer,
    train_triples: Tensor,
    train_labels: Tensor,
    val_triples: Tensor,
    val_labels: Tensor,
    test_triples: Tensor,
    test_labels: Tensor,
    all_triples: Tensor,
    edge_index: Tensor,
    edge_weight: Tensor | None,
    config: Config,
) -> TrainingOutput:
    train_weights = compute_sample_weights(train_labels)
    best_val_f1 = -1.0
    best_state: Dict[str, Tensor] | None = None
    stale_epochs = 0
    history: List[Dict[str, float]] = []
    last_train_loss = 0.0
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        train_logits = model(train_triples, edge_index, edge_weight)
        train_loss = F.binary_cross_entropy_with_logits(
            train_logits, train_labels, weight=train_weights
        )
        train_loss.backward()
        optimizer.step()
        last_train_loss = float(train_loss.item())
        val_probs = predict_probabilities(model, val_triples, edge_index, edge_weight)
        val_metrics = evaluate_binary(val_labels, val_probs, config.threshold)
        history.append(
            {
                "epoch": float(epoch),
                "loss": last_train_loss,
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"],
            }
        )
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"epoch={epoch} loss={last_train_loss:.6f} "
                f"val_precision={val_metrics['precision']:.4f} "
                f"val_recall={val_metrics['recall']:.4f} "
                f"val_f1={val_metrics['f1']:.4f}"
            )
        if stale_epochs >= config.patience:
            print(f"early_stopping_epoch={epoch} best_val_f1={best_val_f1:.4f}")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    val_probs = predict_probabilities(model, val_triples, edge_index, edge_weight)
    if config.optimize_threshold:
        best_threshold, _ = find_best_threshold(
            labels=val_labels,
            probabilities=val_probs,
            threshold_min=config.threshold_min,
            threshold_max=config.threshold_max,
            threshold_steps=config.threshold_steps,
        )
    else:
        best_threshold = config.threshold
    val_metrics = evaluate_binary(val_labels, val_probs, best_threshold)
    test_probs = predict_probabilities(model, test_triples, edge_index, edge_weight)
    test_metrics = evaluate_binary(test_labels, test_probs, best_threshold)
    all_probs = predict_probabilities(model, all_triples, edge_index, edge_weight)
    all_probabilities = all_probs.detach().cpu().tolist()
    return TrainingOutput(
        best_threshold=best_threshold,
        best_val_f1=best_val_f1,
        history=history,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        all_probabilities=all_probabilities,
        train_loss=last_train_loss,
    )
