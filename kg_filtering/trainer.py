from dataclasses import dataclass
from time import perf_counter

import torch
from torch.nn import functional as F

from .gradients import inspect_gradients
from .metrics import compute_sample_weights, evaluate_binary, select_threshold
from .optimization import EarlyStopping, build_optimizer
from .telemetry import EventJournal


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
    journal = EventJournal(config.output_dir, config.record_telemetry)
    journal.emit('training_started', candidates=len(encoded['train']), device=config.device,
        trainable_parameters=sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
    try:
        for epoch in range(1, config.max_epochs + 1):
            started = perf_counter()
            model.train()
            total_loss, total_gradient, steps, clipped = 0.0, 0.0, 0, 0
            for indices, batch in batcher.batches(encoded['train'], shuffle=True, seed=config.seed + epoch):
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch.to(config.device))
                loss = F.binary_cross_entropy_with_logits(logits, targets[indices], weight=weights[indices])
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite loss at epoch {epoch}')
                loss.backward()
                gradient = inspect_gradients(model, config.gradient_clip_norm)
                optimizer.step()
                total_loss += float(loss.detach()) * len(indices)
                total_gradient += gradient.norm
                clipped += int(gradient.clipped)
                steps += 1
            train_seconds = perf_counter() - started
            validation_started = perf_counter()
            probabilities = predict_probabilities(model, encoded['valid'], batcher, config.device)
            threshold = select_threshold(labels['valid'], probabilities, config)
            metrics = evaluate_binary(labels['valid'], probabilities, threshold)
            history.append({'epoch': epoch, 'loss': total_loss / len(encoded['train']),
                'val_f1': metrics['f1'], 'val_precision': metrics['precision'],
                'val_recall': metrics['recall'], 'threshold': threshold,
                'lr': optimizer.param_groups[0]['lr'], 'optimizer_steps': steps,
                'mean_gradient_norm': total_gradient / max(1, steps), 'clipped_steps': clipped,
                'training_seconds': train_seconds, 'validation_seconds': perf_counter() - validation_started})
            scheduler.step(metrics['f1'])
            stop = stopper.update(metrics['f1'], model, epoch)
            journal.emit('epoch_completed', **history[-1], best_epoch=stopper.epoch,
                stale_epochs=stopper.stale, batching=batcher.describe(), cache=batcher.context.cache.describe())
            if epoch == 1 or epoch % 10 == 0 or stop:
                print(f"epoch={epoch} loss={history[-1]['loss']:.6f} val_anomaly_f1={metrics['f1']:.4f}")
            if stop:
                journal.emit('early_stopping', epoch=epoch, best_epoch=stopper.epoch)
                break
        stopper.restore(model)
        probabilities = {name: predict_probabilities(model, triples, batcher, config.device)
            for name, triples in encoded.items()}
        threshold = select_threshold(labels['valid'], probabilities['valid'], config)
        journal.emit('training_completed', best_epoch=stopper.epoch, threshold=threshold,
            best_validation_f1=stopper.score)
        return TrainingOutput(threshold, stopper.score, stopper.epoch, history,
            evaluate_binary(labels['valid'], probabilities['valid'], threshold),
            evaluate_binary(labels['test'], probabilities['test'], threshold), probabilities)
    except BaseException as error:
        journal.emit('training_failed', error_type=type(error).__name__, message=str(error))
        raise
