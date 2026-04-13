import argparse
import json
from dataclasses import asdict
from typing import Any, Dict

from kg_filtering import (
    Config,
    apply_overrides,
    load_config,
    run_predict_pipeline,
    run_train_pipeline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GAT-based Knowledge Graph Anomaly Edge Filtering"
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--mode", type=str, choices=["train", "predict"], default=None)
    parser.add_argument("--clean-graph-path", type=str, default=None)
    parser.add_argument("--inference-graph-path", type=str, default=None)
    parser.add_argument("--output-filtered-path", type=str, default=None)
    parser.add_argument("--output-metrics-path", type=str, default=None)
    parser.add_argument("--output-history-path", type=str, default=None)
    parser.add_argument("--output-dataset-stats-path", type=str, default=None)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--mappings-path", type=str, default=None)
    parser.add_argument("--d", type=int, default=None)
    parser.add_argument("--L", type=int, default=None)
    parser.add_argument("--K", type=int, default=None)
    parser.add_argument("--classifier-hidden-dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--test-ratio", type=float, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--threshold-min", type=float, default=None)
    parser.add_argument("--threshold-max", type=float, default=None)
    parser.add_argument("--threshold-steps", type=int, default=None)
    parser.add_argument("--spurious-fraction", type=float, default=None)
    parser.add_argument("--substitution-fraction", type=float, default=None)
    parser.add_argument("--redundant-fraction", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--use-multi-head", type=int, choices=[0, 1], default=None)
    parser.add_argument("--use-gatv2", type=int, choices=[0, 1], default=None)
    parser.add_argument(
        "--use-edge-weight-awareness", type=int, choices=[0, 1], default=None
    )
    parser.add_argument("--optimize-threshold", type=int, choices=[0, 1], default=None)
    parser.add_argument(
        "--drop-unknown-in-predict", type=int, choices=[0, 1], default=None
    )
    parser.add_argument("--save-checkpoint", type=int, choices=[0, 1], default=None)
    parser.add_argument(
        "--synthetic-if-missing", type=int, choices=[0, 1], default=None
    )
    parser.add_argument("--synthetic-num-entities", type=int, default=None)
    parser.add_argument("--synthetic-num-relations", type=int, default=None)
    parser.add_argument("--synthetic-num-triples", type=int, default=None)
    return parser.parse_args()


def namespace_to_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {
        "mode": args.mode,
        "clean_graph_path": args.clean_graph_path,
        "inference_graph_path": args.inference_graph_path,
        "output_filtered_path": args.output_filtered_path,
        "output_metrics_path": args.output_metrics_path,
        "output_history_path": args.output_history_path,
        "output_dataset_stats_path": args.output_dataset_stats_path,
        "checkpoint_path": args.checkpoint_path,
        "mappings_path": args.mappings_path,
        "embedding_dim": args.d,
        "num_layers": args.L,
        "num_heads": args.K,
        "classifier_hidden_dim": args.classifier_hidden_dim,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "noise_ratio": args.sigma,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "threshold": args.threshold,
        "threshold_min": args.threshold_min,
        "threshold_max": args.threshold_max,
        "threshold_steps": args.threshold_steps,
        "spurious_fraction": args.spurious_fraction,
        "substitution_fraction": args.substitution_fraction,
        "redundant_fraction": args.redundant_fraction,
        "seed": args.seed,
        "device": args.device,
        "synthetic_num_entities": args.synthetic_num_entities,
        "synthetic_num_relations": args.synthetic_num_relations,
        "synthetic_num_triples": args.synthetic_num_triples,
    }
    if args.use_multi_head is not None:
        overrides["use_multi_head"] = bool(args.use_multi_head)
    if args.use_gatv2 is not None:
        overrides["use_gatv2"] = bool(args.use_gatv2)
    if args.use_edge_weight_awareness is not None:
        overrides["use_edge_weight_awareness"] = bool(args.use_edge_weight_awareness)
    if args.optimize_threshold is not None:
        overrides["optimize_threshold"] = bool(args.optimize_threshold)
    if args.drop_unknown_in_predict is not None:
        overrides["drop_unknown_in_predict"] = bool(args.drop_unknown_in_predict)
    if args.save_checkpoint is not None:
        overrides["save_checkpoint"] = bool(args.save_checkpoint)
    if args.synthetic_if_missing is not None:
        overrides["synthetic_if_missing"] = bool(args.synthetic_if_missing)
    return overrides


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    overrides = namespace_to_overrides(args)
    config = apply_overrides(config, overrides)
    print(json.dumps(asdict(config), indent=2))
    if config.mode == "train":
        run_train_pipeline(config)
    elif config.mode == "predict":
        run_predict_pipeline(config)
    else:
        raise ValueError(f"Unsupported mode: {config.mode}")


if __name__ == "__main__":
    main()
