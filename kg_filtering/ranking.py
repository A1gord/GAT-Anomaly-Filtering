from collections import defaultdict

import torch


class FilteredRanking:
    def __init__(self, known_positives, num_entities, chunk_size):
        self.num_entities, self.chunk_size = num_entities, chunk_size
        self.heads, self.tails = defaultdict(set), defaultdict(set)
        for h, r, t in known_positives:
            self.heads[r, t].add(h)
            self.tails[h, r].add(t)

    @torch.no_grad()
    def evaluate(self, model, triples, device):
        if not triples:
            raise ValueError('Ranking requires nonempty evaluation triples')
        model.eval()
        ranks = []
        for h, r, t in triples:
            original = torch.tensor([[h, r, t]], device=device)
            target_score = model.score(original)[0]
            for column, truth, blocked in ((0, h, self.heads[r, t]), (2, t, self.tails[h, r])):
                better, tied = 0, 0
                for start in range(0, self.num_entities, self.chunk_size):
                    ids = torch.arange(start, min(start + self.chunk_size, self.num_entities), device=device)
                    candidates = original.expand(ids.numel(), -1).clone()
                    candidates[:, column] = ids
                    scores = model.score(candidates)
                    excluded = torch.tensor([entity in blocked or entity == truth for entity in ids.tolist()],
                        dtype=torch.bool, device=device)
                    better += int(((scores > target_score) & ~excluded).sum())
                    tied += int(((scores == target_score) & ~excluded).sum())
                ranks.append(1.0 + better + tied / 2.0)
        ranks = torch.tensor(ranks, dtype=torch.float64)
        return {'mrr': float(ranks.reciprocal().mean()), 'mean_rank': float(ranks.mean()),
            **{f'hits_at_{k}': float((ranks <= k).double().mean()) for k in (1, 3, 10)},
            'rank_count': ranks.numel(), 'tie_policy': 'average', 'ranking': 'filtered_head_and_tail'}
