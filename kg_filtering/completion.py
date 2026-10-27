import math
import random

import torch
from torch import nn
from torch.nn import functional as F

from .data import encode_triples
from .optimization import EarlyStopping
from .ranking import FilteredRanking
from .seeding import set_seed


class RotatE(nn.Module):
    def __init__(self, num_entities, num_relations, dimension, margin):
        super().__init__()
        self.entities = nn.Embedding(num_entities, 2 * dimension)
        self.relations = nn.Embedding(num_relations, dimension)
        self.margin = margin
        self.phase_scale = (margin + 2.0) / dimension
        nn.init.uniform_(self.entities.weight, -self.phase_scale, self.phase_scale)
        nn.init.uniform_(self.relations.weight, -self.phase_scale, self.phase_scale)

    def score(self, triples):
        h, r, t = triples.unbind(-1)
        head_real, head_imag = self.entities(h).chunk(2, dim=-1)
        tail_real, tail_imag = self.entities(t).chunk(2, dim=-1)
        phase = self.relations(r) * (math.pi / self.phase_scale)
        real = head_real * phase.cos() - head_imag * phase.sin() - tail_real
        imag = head_real * phase.sin() + head_imag * phase.cos() - tail_imag
        distance = torch.stack((real, imag), -1).norm(dim=-1).sum(-1)
        return self.margin - distance


def sample_negatives(positives, num_entities, count, known, rng):
    negatives = []
    for triple in positives:
        row = []
        for _ in range(max(1000, count * 100)):
            if len(row) == count:
                break
            candidate = list(triple)
            candidate[0 if rng.getrandbits(1) else 2] = rng.randrange(num_entities)
            candidate = tuple(candidate)
            if candidate not in known:
                row.append(candidate)
        if len(row) != count:
            raise ValueError('Cannot sample enough filtered completion negatives')
        negatives.append(row)
    return negatives


def fit_completion(train_triples, valid_triples, test_triples, known, num_entities, num_relations, config):
    if not train_triples:
        raise ValueError('Restoration removed all completion training triples')
    set_seed(config.seed)
    rng = random.Random(config.seed)
    model = RotatE(num_entities, num_relations, config.completion_dim, config.completion_margin).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.completion_lr)
    stopper = EarlyStopping(config.patience)
    ranking = FilteredRanking(known, num_entities, config.ranking_batch_size)
    negative_filter = set(known) | set(train_triples)
    history = []
    for epoch in range(1, config.completion_epochs + 1):
        model.train()
        order = list(range(len(train_triples)))
        rng.shuffle(order)
        total_loss = 0.0
        for start in range(0, len(order), config.completion_batch_size):
            positives = [train_triples[index] for index in order[start:start + config.completion_batch_size]]
            negatives = sample_negatives(positives, num_entities, config.completion_negative_samples, negative_filter, rng)
            positive_scores = model.score(torch.tensor(positives, device=config.device))
            negative_scores = model.score(torch.tensor(negatives, device=config.device))
            adversarial = F.softmax(negative_scores, dim=-1).detach()
            loss = -(F.logsigmoid(positive_scores).mean() +
                (adversarial * F.logsigmoid(-negative_scores)).sum(-1).mean()) / 2
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite RotatE loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(positives)
        metrics = ranking.evaluate(model, valid_triples, config.device)
        history.append({'epoch': epoch, 'loss': total_loss / len(train_triples), 'valid_mrr': metrics['mrr']})
        if stopper.update(metrics['mrr'], model, epoch):
            break
    stopper.restore(model)
    return {'best_epoch': stopper.epoch, 'valid': ranking.evaluate(model, valid_triples, config.device),
        'test': ranking.evaluate(model, test_triples, config.device), 'history': history,
        'training_triples': len(train_triples)}


def compare_completion(reference, noisy_train, restored_train, ent2id, rel2id, config):
    known = encode_triples(sorted(reference.positives), ent2id, rel2id)
    valid = encode_triples(reference.valid, ent2id, rel2id)
    test = encode_triples(reference.test, ent2id, rel2id)
    results = {}
    for name, triples in (('clean', reference.train), ('noisy', noisy_train), ('restored', restored_train)):
        results[name] = fit_completion(encode_triples(sorted(set(triples)), ent2id, rel2id),
            valid, test, known, len(ent2id), len(rel2id), config)
    noisy_mrr = results['noisy']['test']['mrr']
    results['relative_mrr_gain_percent'] = 100 * (results['restored']['test']['mrr'] / noisy_mrr - 1) if noisy_mrr else None
    return results
