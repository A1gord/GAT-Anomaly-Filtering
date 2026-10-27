from dataclasses import replace

ABLATIONS = {
    'full': {},
    'without_relation_attention': {'use_relation_attention': False},
    'without_relation_messages': {'use_relation_messages': False},
    'without_inverse_encoding': {'use_inverse_relations': False},
    'relation_agnostic_gatv2': {'use_relation_attention': False, 'use_relation_messages': False},
    'without_multi_head': {'use_multi_head': False},
    'without_semantic_features': {'use_semantic_features': False},
}


def ablation_config(config, name):
    if name not in ABLATIONS:
        raise ValueError(f'Unknown ablation: {name}')
    if name == 'without_semantic_features' and not config.entity_features_path:
        raise ValueError('Semantic ablation requires supplied semantic features in the full model')
    return replace(config, **ABLATIONS[name])
