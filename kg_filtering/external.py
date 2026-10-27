import csv
import math
from pathlib import Path

import torch


def load_external_probabilities(path, partitions):
    if not path:
        raise ValueError('CAGED requires external_scores_path from its reference implementation')
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle, delimiter='\t'))
    indexed = {}
    for row in rows:
        key = row['partition'], int(row['row'])
        if key in indexed:
            raise ValueError(f'Duplicate external score for {key}')
        indexed[key] = row
    expected = sum(len(partition.triples) for partition in partitions.values())
    if len(indexed) != expected:
        raise ValueError('External scores must cover every exported candidate exactly once')
    result = {}
    for name, partition in partitions.items():
        scores = []
        for index, triple in enumerate(partition.triples):
            row = indexed.get((name, index))
            if row is None or (row['head'], row['relation'], row['tail']) != triple:
                raise ValueError(f'External scores do not match {name} row {index}')
            score = float(row['valid_probability'])
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError('External validity scores must lie in [0, 1]')
            scores.append(score)
        result[name] = torch.tensor(scores, dtype=torch.float32)
    return result
