import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import Tensor

from .types import TripleId, TripleStr


def parse_triple_line(line: str) -> TripleStr:
    stripped = line.strip()
    parts = stripped.split("\t")
    if len(parts) < 3:
        parts = stripped.split()
    if len(parts) < 3:
        raise ValueError(f"Invalid triple line: {line}")
    return parts[0], parts[1], parts[2]


def load_triples(path: str) -> List[TripleStr]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Triple file not found: {file_path}")
    triples: List[TripleStr] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            if raw.strip():
                triples.append(parse_triple_line(raw))
    if len(triples) == 0:
        raise ValueError("Triple file is empty")
    return triples


def generate_synthetic_clean_triples(
    num_entities: int, num_relations: int, num_triples: int, seed: int
) -> List[TripleStr]:
    rng = random.Random(seed)
    entities = [f"e{i}" for i in range(num_entities)]
    relations = [f"r{i}" for i in range(num_relations)]
    seen: set[TripleStr] = set()
    triples: List[TripleStr] = []
    max_tries = max(10000, num_triples * 20)
    tries = 0
    while len(triples) < num_triples and tries < max_tries:
        h = rng.choice(entities)
        t = rng.choice(entities)
        r = rng.choice(relations)
        candidate = (h, r, t)
        if candidate not in seen:
            seen.add(candidate)
            triples.append(candidate)
        tries += 1
    if len(triples) < num_triples:
        raise RuntimeError("Unable to generate enough synthetic clean triples")
    return triples


def build_entity_relation_vocab(
    clean_triples: List[TripleStr],
) -> Tuple[List[str], List[str]]:
    entities = sorted(
        {h for h, _, _ in clean_triples} | {t for _, _, t in clean_triples}
    )
    relations = sorted({r for _, r, _ in clean_triples})
    return entities, relations


def normalize_noise_fractions(
    spurious: float, substitution: float, redundant: float
) -> Tuple[float, float, float]:
    total = spurious + substitution + redundant
    if total <= 0.0:
        return 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0
    return spurious / total, substitution / total, redundant / total


def split_by_weights(
    total: int, w1: float, w2: float, w3: float
) -> Tuple[int, int, int]:
    w1, w2, w3 = normalize_noise_fractions(w1, w2, w3)
    c1 = int(round(total * w1))
    c2 = int(round(total * w2))
    c3 = total - c1 - c2
    if c3 < 0:
        c3 = 0
        overflow = c1 + c2 - total
        if c2 >= overflow:
            c2 -= overflow
        else:
            overflow -= c2
            c2 = 0
            c1 = max(0, c1 - overflow)
    return c1, c2, c3


def inject_anomalies(
    clean_triples: List[TripleStr],
    noise_ratio: float,
    entities: List[str],
    relations: List[str],
    spurious_fraction: float,
    substitution_fraction: float,
    redundant_fraction: float,
    seed: int,
) -> Tuple[List[TripleStr], List[int], Dict[str, int]]:
    rng = random.Random(seed + 1)
    num_clean = len(clean_triples)
    num_anomalies = int(round(num_clean * noise_ratio))
    if num_anomalies <= 0:
        all_triples = list(clean_triples)
        labels = [1 for _ in all_triples]
        combined = list(zip(all_triples, labels))
        rng.shuffle(combined)
        triples_shuffled = [item[0] for item in combined]
        labels_shuffled = [item[1] for item in combined]
        return (
            triples_shuffled,
            labels_shuffled,
            {"spurious": 0, "substitution": 0, "redundant": 0},
        )

    spurious_target, substitution_target, redundant_target = split_by_weights(
        num_anomalies, spurious_fraction, substitution_fraction, redundant_fraction
    )
    clean_set = set(clean_triples)
    pair_set = {(h, t) for h, _, t in clean_set}
    anomalies_unique: set[TripleStr] = set()
    anomalies: List[TripleStr] = []

    spurious: List[TripleStr] = []
    attempts = 0
    max_attempts = max(5000, spurious_target * 300)
    while len(spurious) < spurious_target and attempts < max_attempts:
        h = rng.choice(entities)
        t = rng.choice(entities)
        r = rng.choice(relations)
        candidate = (h, r, t)
        if (h, t) in pair_set:
            attempts += 1
            continue
        if candidate in clean_set or candidate in anomalies_unique:
            attempts += 1
            continue
        spurious.append(candidate)
        anomalies_unique.add(candidate)
        attempts += 1

    substitutions: List[TripleStr] = []
    if len(relations) > 1:
        attempts = 0
        max_attempts = max(5000, substitution_target * 300)
        while len(substitutions) < substitution_target and attempts < max_attempts:
            h, r, t = rng.choice(clean_triples)
            new_r = rng.choice(relations)
            if new_r == r:
                attempts += 1
                continue
            candidate = (h, new_r, t)
            if candidate in clean_set or candidate in anomalies_unique:
                attempts += 1
                continue
            substitutions.append(candidate)
            anomalies_unique.add(candidate)
            attempts += 1

    redundancies: List[TripleStr] = [
        rng.choice(clean_triples) for _ in range(redundant_target)
    ]

    anomalies.extend(spurious)
    anomalies.extend(substitutions)
    anomalies.extend(redundancies)

    while len(anomalies) < num_anomalies:
        anomalies.append(rng.choice(clean_triples))

    all_triples = list(clean_triples) + anomalies
    labels = [1 for _ in clean_triples] + [0 for _ in anomalies]
    combined = list(zip(all_triples, labels))
    rng.shuffle(combined)
    triples_shuffled = [item[0] for item in combined]
    labels_shuffled = [item[1] for item in combined]
    noise_stats = {
        "spurious": len(spurious),
        "substitution": len(substitutions),
        "redundant": len(redundancies),
    }
    return triples_shuffled, labels_shuffled, noise_stats


def build_mappings(
    entities: List[str], relations: List[str]
) -> Tuple[Dict[str, int], Dict[str, int], Dict[int, str], Dict[int, str]]:
    ent2id = {entity: idx for idx, entity in enumerate(entities)}
    rel2id = {relation: idx for idx, relation in enumerate(relations)}
    id2ent = {idx: entity for entity, idx in ent2id.items()}
    id2rel = {idx: relation for relation, idx in rel2id.items()}
    return ent2id, rel2id, id2ent, id2rel


def encode_triples(
    triples: List[TripleStr], ent2id: Dict[str, int], rel2id: Dict[str, int]
) -> List[TripleId]:
    encoded: List[TripleId] = []
    for h, r, t in triples:
        encoded.append((ent2id[h], rel2id[r], ent2id[t]))
    return encoded


def encode_triples_with_mask(
    triples: List[TripleStr],
    ent2id: Dict[str, int],
    rel2id: Dict[str, int],
    drop_unknown: bool,
) -> Tuple[List[TripleId], List[int]]:
    encoded: List[TripleId] = []
    kept_indices: List[int] = []
    for idx, (h, r, t) in enumerate(triples):
        if h not in ent2id or t not in ent2id or r not in rel2id:
            if drop_unknown:
                continue
            else:
                raise ValueError(f"Unknown token in triple: {(h, r, t)}")
        encoded.append((ent2id[h], rel2id[r], ent2id[t]))
        kept_indices.append(idx)
    return encoded, kept_indices


def split_indices(
    num_items: int, train_ratio: float, val_ratio: float, test_ratio: float, seed: int
) -> Tuple[List[int], List[int], List[int]]:
    if num_items < 3:
        raise ValueError("Dataset must contain at least 3 samples")
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-8:
        raise ValueError("Split ratios must sum to 1.0")
    indices = list(range(num_items))
    rng = random.Random(seed + 2)
    rng.shuffle(indices)
    n_train = max(1, int(num_items * train_ratio))
    n_val = max(1, int(num_items * val_ratio))
    n_test = num_items - n_train - n_val
    if n_test <= 0:
        n_test = 1
        if n_train > n_val:
            n_train -= 1
        else:
            n_val -= 1
    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError("Invalid split sizes")
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]
    return train_idx, val_idx, test_idx


def build_graph_from_triples(
    triples: Tensor,
    num_entities: int,
    use_edge_weight_awareness: bool,
    device: torch.device,
) -> Tuple[Tensor, Tensor | None]:
    pairs = [(int(h), int(t)) for h, _, t in triples.tolist()]
    edge_counter = Counter(pairs)
    for node in range(num_entities):
        edge_counter[(node, node)] += 1
    src: List[int] = []
    dst: List[int] = []
    weights: List[float] = []
    for (s, d), count in edge_counter.items():
        src.append(s)
        dst.append(d)
        weights.append(float(count))
    edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)
    if use_edge_weight_awareness:
        edge_weight = torch.tensor(weights, dtype=torch.float, device=device).unsqueeze(
            -1
        )
        edge_weight = edge_weight / torch.clamp(edge_weight.max(), min=1.0)
        return edge_index, edge_weight
    return edge_index, None


def build_dataset_stats(
    clean_count: int,
    noisy_count: int,
    labels: List[int],
    split_sizes: Dict[str, int],
    noise_stats: Dict[str, int],
) -> Dict[str, int]:
    positives = sum(labels)
    negatives = len(labels) - positives
    return {
        "clean_count": clean_count,
        "noisy_count": noisy_count,
        "positive_count": positives,
        "negative_count": negatives,
        "train_size": split_sizes["train"],
        "val_size": split_sizes["val"],
        "test_size": split_sizes["test"],
        "spurious_anomalies": noise_stats["spurious"],
        "substitution_anomalies": noise_stats["substitution"],
        "redundant_anomalies": noise_stats["redundant"],
    }
