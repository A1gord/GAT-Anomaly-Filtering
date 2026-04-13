from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import torch

from .config import Config, apply_overrides
from .data import (
    build_dataset_stats,
    build_entity_relation_vocab,
    build_graph_from_triples,
    build_mappings,
    encode_triples,
    encode_triples_with_mask,
    generate_synthetic_clean_triples,
    inject_anomalies,
    load_triples,
    split_indices,
)
from .io_utils import (
    export_filtered_graph,
    load_checkpoint,
    load_mappings,
    save_checkpoint,
    save_json,
    save_mappings,
)
from .model import EdgeValidityModel
from .seeding import set_seed
from .trainer import predict_probabilities, train_model


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def load_or_generate_clean_triples(config: Config) -> list[tuple[str, str, str]]:
    clean_path = Path(config.clean_graph_path)
    if clean_path.exists():
        return load_triples(config.clean_graph_path)
    if config.synthetic_if_missing:
        return generate_synthetic_clean_triples(
            num_entities=config.synthetic_num_entities,
            num_relations=config.synthetic_num_relations,
            num_triples=config.synthetic_num_triples,
            seed=config.seed,
        )
    raise FileNotFoundError(
        f"Clean graph not found and synthetic generation is disabled: {clean_path}"
    )


def build_checkpoint_payload(
    model: EdgeValidityModel,
    config: Config,
    best_threshold: float,
    num_entities: int,
    num_relations: int,
) -> Dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "config": asdict(config),
        "best_threshold": best_threshold,
        "num_entities": num_entities,
        "num_relations": num_relations,
    }


def run_train_pipeline(config: Config) -> Dict[str, Any]:
    set_seed(config.seed)
    device = resolve_device(config.device)
    clean_triples = load_or_generate_clean_triples(config)
    entities, relations = build_entity_relation_vocab(clean_triples)
    all_triples_str, all_labels, noise_stats = inject_anomalies(
        clean_triples=clean_triples,
        noise_ratio=config.noise_ratio,
        entities=entities,
        relations=relations,
        spurious_fraction=config.spurious_fraction,
        substitution_fraction=config.substitution_fraction,
        redundant_fraction=config.redundant_fraction,
        seed=config.seed,
    )
    ent2id, rel2id, _, _ = build_mappings(entities, relations)
    all_triples_id = encode_triples(all_triples_str, ent2id, rel2id)
    train_idx, val_idx, test_idx = split_indices(
        num_items=len(all_triples_id),
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        seed=config.seed,
    )
    triples_tensor = torch.tensor(all_triples_id, dtype=torch.long, device=device)
    labels_tensor = torch.tensor(all_labels, dtype=torch.float, device=device)
    train_idx_tensor = torch.tensor(train_idx, dtype=torch.long, device=device)
    val_idx_tensor = torch.tensor(val_idx, dtype=torch.long, device=device)
    test_idx_tensor = torch.tensor(test_idx, dtype=torch.long, device=device)
    train_triples = triples_tensor[train_idx_tensor]
    val_triples = triples_tensor[val_idx_tensor]
    test_triples = triples_tensor[test_idx_tensor]
    train_labels = labels_tensor[train_idx_tensor]
    val_labels = labels_tensor[val_idx_tensor]
    test_labels = labels_tensor[test_idx_tensor]
    edge_index, edge_weight = build_graph_from_triples(
        triples=train_triples,
        num_entities=len(entities),
        use_edge_weight_awareness=config.use_edge_weight_awareness,
        device=device,
    )
    model = EdgeValidityModel(
        num_entities=len(entities),
        num_relations=len(relations),
        config=config,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    train_output = train_model(
        model=model,
        optimizer=optimizer,
        train_triples=train_triples,
        train_labels=train_labels,
        val_triples=val_triples,
        val_labels=val_labels,
        test_triples=test_triples,
        test_labels=test_labels,
        all_triples=triples_tensor,
        edge_index=edge_index,
        edge_weight=edge_weight,
        config=config,
    )
    kept_count = export_filtered_graph(
        triples=all_triples_str,
        probabilities=train_output.all_probabilities,
        threshold=train_output.best_threshold,
        output_path=config.output_filtered_path,
    )
    split_sizes = {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)}
    dataset_stats = build_dataset_stats(
        clean_count=len(clean_triples),
        noisy_count=len(all_triples_str),
        labels=all_labels,
        split_sizes=split_sizes,
        noise_stats=noise_stats,
    )
    metrics_payload: Dict[str, Any] = {
        "config": asdict(config),
        "selected_threshold": train_output.best_threshold,
        "val_precision": train_output.val_metrics["precision"],
        "val_recall": train_output.val_metrics["recall"],
        "val_f1": train_output.val_metrics["f1"],
        "test_precision": train_output.test_metrics["precision"],
        "test_recall": train_output.test_metrics["recall"],
        "test_f1": train_output.test_metrics["f1"],
        "num_clean_triples": len(clean_triples),
        "num_noisy_triples": len(all_triples_str),
        "num_filtered_kept": kept_count,
    }
    save_json(config.output_metrics_path, metrics_payload)
    save_json(config.output_history_path, {"history": train_output.history})
    save_json(config.output_dataset_stats_path, dataset_stats)
    save_mappings(config.mappings_path, ent2id, rel2id)
    if config.save_checkpoint:
        checkpoint_payload = build_checkpoint_payload(
            model=model,
            config=config,
            best_threshold=train_output.best_threshold,
            num_entities=len(entities),
            num_relations=len(relations),
        )
        save_checkpoint(config.checkpoint_path, checkpoint_payload)
    print(f"selected_threshold={train_output.best_threshold:.4f}")
    print(f"test_precision={train_output.test_metrics['precision']:.4f}")
    print(f"test_recall={train_output.test_metrics['recall']:.4f}")
    print(f"test_f1={train_output.test_metrics['f1']:.4f}")
    print(f"filtered_graph_saved={config.output_filtered_path}")
    print(f"metrics_saved={config.output_metrics_path}")
    return metrics_payload


def run_predict_pipeline(config: Config) -> Dict[str, Any]:
    set_seed(config.seed)
    device = resolve_device(config.device)
    inference_path = Path(config.inference_graph_path)
    if not inference_path.exists():
        if Path(config.clean_graph_path).exists():
            triples_str = load_triples(config.clean_graph_path)
        else:
            raise FileNotFoundError(
                f"Inference graph not found: {inference_path}. clean_graph_path fallback also missing."
            )
    else:
        triples_str = load_triples(config.inference_graph_path)
    ent2id, rel2id = load_mappings(config.mappings_path)
    checkpoint = load_checkpoint(config.checkpoint_path)
    checkpoint_config_payload = checkpoint.get("config", {})
    checkpoint_config = Config()
    if isinstance(checkpoint_config_payload, dict):
        checkpoint_config = apply_overrides(
            checkpoint_config, checkpoint_config_payload
        )
    checkpoint_config.device = config.device
    num_entities = int(checkpoint.get("num_entities", len(ent2id)))
    num_relations = int(checkpoint.get("num_relations", len(rel2id)))
    model = EdgeValidityModel(
        num_entities=num_entities, num_relations=num_relations, config=checkpoint_config
    ).to(device)
    if "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint does not contain model_state_dict")
    model.load_state_dict(checkpoint["model_state_dict"])
    best_threshold = float(checkpoint.get("best_threshold", config.threshold))
    encoded, kept_indices = encode_triples_with_mask(
        triples=triples_str,
        ent2id=ent2id,
        rel2id=rel2id,
        drop_unknown=config.drop_unknown_in_predict,
    )
    if len(encoded) == 0:
        raise ValueError("No triples available for prediction after mapping filter")
    triples_tensor = torch.tensor(encoded, dtype=torch.long, device=device)
    edge_index, edge_weight = build_graph_from_triples(
        triples=triples_tensor,
        num_entities=num_entities,
        use_edge_weight_awareness=checkpoint_config.use_edge_weight_awareness,
        device=device,
    )
    probs_tensor = predict_probabilities(model, triples_tensor, edge_index, edge_weight)
    probs = probs_tensor.detach().cpu().tolist()
    full_probs = [0.0 for _ in triples_str]
    for local_idx, original_idx in enumerate(kept_indices):
        full_probs[original_idx] = probs[local_idx]
    kept_count = export_filtered_graph(
        triples=triples_str,
        probabilities=full_probs,
        threshold=best_threshold,
        output_path=config.output_filtered_path,
    )
    inference_metrics: Dict[str, Any] = {
        "mode": "predict",
        "threshold": best_threshold,
        "total_input_triples": len(triples_str),
        "encoded_triples": len(encoded),
        "dropped_unknown_triples": len(triples_str) - len(encoded),
        "num_filtered_kept": kept_count,
        "checkpoint_path": config.checkpoint_path,
        "mappings_path": config.mappings_path,
    }
    save_json(config.output_metrics_path, inference_metrics)
    print(f"selected_threshold={best_threshold:.4f}")
    print(f"filtered_graph_saved={config.output_filtered_path}")
    print(f"metrics_saved={config.output_metrics_path}")
    return inference_metrics
