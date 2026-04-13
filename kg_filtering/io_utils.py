import json
from pathlib import Path
from typing import Any, Dict, List

import torch

from .types import TripleStr


def ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def save_json(path: str, payload: Dict[str, Any]) -> None:
    ensure_parent_dir(path)
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: str) -> Dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    return json.loads(file_path.read_text(encoding="utf-8"))


def export_filtered_graph(
    triples: List[TripleStr],
    probabilities: List[float],
    threshold: float,
    output_path: str,
) -> int:
    ensure_parent_dir(output_path)
    kept: List[TripleStr] = []
    seen: set[TripleStr] = set()
    for triple, prob in zip(triples, probabilities):
        if prob >= threshold and triple not in seen:
            kept.append(triple)
            seen.add(triple)
    with Path(output_path).open("w", encoding="utf-8") as handle:
        for h, r, t in kept:
            handle.write(f"{h}\t{r}\t{t}\n")
    return len(kept)


def save_checkpoint(path: str, payload: Dict[str, Any]) -> None:
    ensure_parent_dir(path)
    torch.save(payload, path)


def load_checkpoint(path: str) -> Dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {file_path}")
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint payload is not a dictionary")
    return checkpoint


def save_mappings(path: str, ent2id: Dict[str, int], rel2id: Dict[str, int]) -> None:
    payload = {"ent2id": ent2id, "rel2id": rel2id}
    save_json(path, payload)


def load_mappings(path: str) -> tuple[Dict[str, int], Dict[str, int]]:
    payload = load_json(path)
    ent2id = payload.get("ent2id", {})
    rel2id = payload.get("rel2id", {})
    if not isinstance(ent2id, dict) or not isinstance(rel2id, dict):
        raise ValueError("Invalid mappings payload")
    return ent2id, rel2id
