from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .metrics import compute_sample_weights, evaluate_binary, select_threshold
from .optimization import EarlyStopping, build_optimizer


@dataclass
class TrainingOutput:
    best_threshold: float
    best_val_f1: float
    best_epoch: int
    history: list
    val_metrics: dict
    test_metrics: dict
    probabilities: dict


@torch.no_grad()
def predict_probabilities(model, triples, batcher, device):
    model.eval()
    result = torch.empty(len(triples))
    for indices, batch in batcher.batches(triples):
        result[indices] = model(batch.to(device)).sigmoid().cpu()
    return result


def train_model(model, encoded, labels, batcher, config):
    optimizer, scheduler = build_optimizer(model, config)
    stopper = EarlyStopping(config.patience)
    targets = torch.tensor(labels['train'], dtype=torch.float32, device=config.device)
    weights = compute_sample_weights(targets)
    history = []
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        total_loss = 0.0
        for indices, batch in batcher.batches(encoded['train'], shuffle=True, seed=config.seed + epoch):
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch.to(config.device))
            loss = F.binary_cross_entropy_with_logits(logits, targets[indices], weight=weights[indices])
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss at epoch {epoch}')
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
        probabilities = predict_probabilities(model, encoded['valid'], batcher, config.device)
        threshold = select_threshold(labels['valid'], probabilities, config)
        metrics = evaluate_binary(labels['valid'], probabilities, threshold)
        history.append({'epoch': epoch, 'loss': total_loss / len(encoded['train']),
            'val_f1': metrics['f1'], 'val_precision': metrics['precision'],
            'val_recall': metrics['recall'], 'threshold': threshold,
            'lr': optimizer.param_groups[0]['lr']})
        scheduler.step(metrics['f1'])
        stop = stopper.update(metrics['f1'], model, epoch)
        if epoch == 1 or epoch % 10 == 0 or stop:
            print(f"epoch={epoch} loss={history[-1]['loss']:.6f} val_anomaly_f1={metrics['f1']:.4f}")
        if stop:
            break
    stopper.restore(model)
    probabilities = {name: predict_probabilities(model, triples, batcher, config.device)
        for name, triples in encoded.items()}
    threshold = select_threshold(labels['valid'], probabilities['valid'], config)
    return TrainingOutput(threshold, stopper.score, stopper.epoch, history,
        evaluate_binary(labels['valid'], probabilities['valid'], threshold),
        evaluate_binary(labels['test'], probabilities['test'], threshold), probabilities)
