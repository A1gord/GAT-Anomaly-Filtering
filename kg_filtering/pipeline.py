from dataclasses import asdict, replace
from pathlib import Path

import torch

from .artifacts import resolve_artifact_paths
from .baselines import build_model, fit_random_forest
from .checkpoints import dataset_fingerprint, read_model_checkpoint, write_model_checkpoint
from .completion import compare_completion
from .config import Config, apply_overrides
from .constraints import DomainConstraints
from .corruption import build_corrupted_splits
from .data import build_entity_relation_vocab, build_mappings, encode_triples, encode_triples_with_mask, load_triples
from .datasets import load_reference_splits
from .external import load_external_probabilities
from .features import load_semantic_features
from .graph import RelationalContext
from .integrity import audit_partitions, corruption_fingerprint
from .io_utils import export_filtered_graph, save_json, save_mappings
from .metrics import evaluate_binary, select_threshold
from .reporting import anomaly_breakdown, export_candidates, export_scores, restoration_counts
from .sampling import CandidateBatcher
from .seeding import set_seed
from .trainer import predict_probabilities, train_model


def resolve_device(device_name):
    device = torch.device(device_name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')
    return str(device)


def prepare_data(config):
    config.validate()
    reference = load_reference_splits(config)
    values, stats = build_corrupted_splits(reference, config)
    partitions = dict(zip(('train', 'valid', 'test'), values))
    audit = audit_partitions(reference, partitions)
    entities, relations = build_entity_relation_vocab(sorted(reference.positives))
    ent2id, rel2id, id2ent, id2rel = build_mappings(entities, relations)
    encoded = {name: encode_triples(partition.triples, ent2id, rel2id) for name, partition in partitions.items()}
    context = RelationalContext(encoded['train'], len(entities), len(relations), config.use_inverse_relations)
    export_candidates(Path(config.output_dir) / 'candidates', partitions)
    for name, clean in zip(('train', 'valid', 'test'), (reference.train, reference.valid, reference.test)):
        export_filtered_graph(clean, [1.0] * len(clean), 0.5,
            str(Path(config.output_dir) / 'reference' / f'{name}.tsv'))
    save_mappings(config.mappings_path, ent2id, rel2id)
    save_json(config.output_dataset_stats_path, {'dataset': config.dataset, 'seed': config.seed,
        'reference_split_seed': config.reference_split_seed, 'partitions': stats, 'integrity': audit,
        'entities': len(entities), 'relations': len(relations),
        'fingerprint': dataset_fingerprint(reference), 'context': 'corrupted_training_partition',
        'noise_ratio_definition': 'injected_count / clean_partition_count'})
    return reference, partitions, encoded, context, (ent2id, rel2id, id2ent, id2rel)


def run_prepare_pipeline(config):
    config = resolve_artifact_paths(config)
    reference, partitions, _, _, _ = prepare_data(config)
    payload = {'mode': 'prepare', 'dataset_fingerprint': dataset_fingerprint(reference),
        'candidate_directory': str(Path(config.output_dir) / 'candidates'),
        'partition_sizes': {name: len(partition.triples) for name, partition in partitions.items()}}
    save_json(config.output_metrics_path, payload)
    return payload


def run_train_pipeline(config):
    config = resolve_artifact_paths(config)
    config = replace(config, device=resolve_device(config.device))
    set_seed(config.seed)
    reference, partitions, encoded, context, mappings = prepare_data(config)
    ent2id, rel2id, id2ent, id2rel = mappings
    labels = {name: partition.labels for name, partition in partitions.items()}
    model = None
    history, best_epoch = [], None
    if config.model_name == 'random_forest':
        probabilities = fit_random_forest(encoded, labels, context, config)
    elif config.model_name == 'rule_based':
        constraints = DomainConstraints(config.rules_path, context, id2ent, id2rel)
        probabilities = {name: constraints.probabilities(triples) for name, triples in encoded.items()}
    elif config.model_name == 'caged':
        probabilities = load_external_probabilities(config.external_scores_path, partitions)
    else:
        semantic = load_semantic_features(config.entity_features_path, ent2id, config.semantic_dim)
        config = replace(config, semantic_dim=semantic.size(1))
        model = build_model(len(ent2id), len(rel2id), config).to(config.device)
        if hasattr(model, 'encoder'):
            model.encoder.features.semantic.copy_(semantic.to(config.device))
        depth = config.num_layers + int(config.model_name == 'gcn')
        batcher = CandidateBatcher(context, depth, config.batch_size)
        output = train_model(model, encoded, labels, batcher, config)
        probabilities, history, best_epoch = output.probabilities, output.history, output.best_epoch
    threshold = select_threshold(labels['valid'], probabilities['valid'], config)
    metrics = {name: evaluate_binary(labels[name], probabilities[name], threshold) for name in partitions}
    all_triples = [triple for partition in partitions.values() for triple in partition.triples]
    all_probabilities = torch.cat([probabilities[name] for name in partitions]).tolist()
    kept = export_filtered_graph(all_triples, all_probabilities, threshold, config.output_filtered_path)
    export_scores(config.output_scores_path, {name: part.triples for name, part in partitions.items()}, probabilities, threshold)
    for name, partition in partitions.items():
        export_filtered_graph(partition.triples, probabilities[name].tolist(), threshold,
            str(Path(config.output_dir) / 'restored' / f'{name}.tsv'))
    payload = {'config': asdict(config), 'selected_threshold': threshold, 'best_epoch': best_epoch,
        'positive_metric_class': 'anomaly', **metrics,
        'test_by_anomaly_type': anomaly_breakdown(partitions['test'], probabilities['test'], threshold),
        'restoration': restoration_counts(all_triples, all_probabilities, threshold),
        'num_filtered_kept': kept, 'dataset_fingerprint': dataset_fingerprint(reference),
        'corruption_fingerprint': corruption_fingerprint(partitions),
        'checkpoint_available': model is not None and config.save_checkpoint,
        'semantic_features_used': bool(model is not None and hasattr(model, 'encoder') and
            config.semantic_dim and config.use_semantic_features),
        'duplicate_identifiability': 'Identical triples have identical model inputs; deduplication is a separate export operation.'}
    if config.run_completion:
        restored_train = [triple for triple, score in zip(partitions['train'].triples, probabilities['train']) if score >= threshold]
        payload['completion'] = compare_completion(reference, partitions['train'].triples,
            restored_train, ent2id, rel2id, config)
    save_json(config.output_metrics_path, payload)
    save_json(config.output_history_path, {'history': history})
    if model is not None and config.save_checkpoint:
        write_model_checkpoint(config.checkpoint_path, model, config, threshold, mappings, context,
            dataset_fingerprint(reference), best_epoch)
    print(f"test_anomaly_f1={metrics['test']['f1']:.4f} threshold={threshold:.4f}")
    return payload


def run_predict_pipeline(config):
    config = resolve_artifact_paths(config)
    config.validate()
    device = resolve_device(config.device)
    checkpoint = read_model_checkpoint(config.checkpoint_path)
    trained = apply_overrides(Config(), checkpoint['config'])
    trained = replace(trained, device=device)
    ent2id, rel2id = checkpoint['ent2id'], checkpoint['rel2id']
    triples = load_triples(config.inference_graph_path)
    encoded, indices = encode_triples_with_mask(triples, ent2id, rel2id, config.drop_unknown_in_predict)
    if not encoded:
        raise ValueError('No known candidates remain for inference')
    model = build_model(len(ent2id), len(rel2id), trained).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    context_triples = encoded if config.inference_context == 'observed' else checkpoint['context_triples']
    context = RelationalContext(context_triples, len(ent2id), len(rel2id), trained.use_inverse_relations)
    batcher = CandidateBatcher(context, trained.num_layers + int(trained.model_name == 'gcn'), config.batch_size)
    probabilities = predict_probabilities(model, encoded, batcher, device)
    threshold = checkpoint['best_threshold']
    known_triples = [triples[index] for index in indices]
    kept = export_filtered_graph(known_triples, probabilities.tolist(), threshold, config.output_filtered_path)
    score_vector = [None] * len(triples)
    for index, value in zip(indices, probabilities.tolist()):
        score_vector[index] = value
    export_scores(config.output_scores_path, {'predict': known_triples}, {'predict': probabilities}, threshold)
    save_json(str(Path(config.output_dir) / 'prediction_alignment.json'), {
        'original_row_indices': indices, 'valid_probabilities': score_vector})
    payload = {'mode': 'predict', 'threshold': threshold, 'total_input_triples': len(triples),
        'encoded_triples': len(encoded), 'dropped_unknown_triples': len(triples) - len(encoded),
        'num_filtered_kept': kept, 'context': config.inference_context,
        'restoration': restoration_counts(known_triples, probabilities, threshold)}
    save_json(config.output_metrics_path, payload)
    return payload
