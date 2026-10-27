import hashlib
import json


def corruption_fingerprint(partitions):
    digest = hashlib.sha256()
    for name, partition in partitions.items():
        digest.update(name.encode())
        for row in zip(partition.triples, partition.labels, partition.anomaly_types):
            digest.update(json.dumps(row, ensure_ascii=False, separators=(',', ':')).encode())
            digest.update(b'\n')
    return digest.hexdigest()


def audit_partitions(reference, partitions):
    known = reference.positives
    observed_negatives = set()
    for (name, partition), clean in zip(partitions.items(), (reference.train, reference.valid, reference.test)):
        expected = set(clean)
        positives = [triple for triple, label in zip(partition.triples, partition.labels) if label]
        if len(positives) != len(clean) or set(positives) != expected:
            raise ValueError(f'{name}: reference positives were altered')
        for triple, label, family in zip(partition.triples, partition.labels, partition.anomaly_types):
            if label == 1:
                if family != 'valid':
                    raise ValueError(f'{name}: positive row has an anomalous family')
            elif family == 'redundant_assertion':
                if triple not in expected:
                    raise ValueError(f'{name}: duplicate assertion came from another partition')
            elif family in ('entity_corruption', 'relation_substitution'):
                if triple in known or triple in observed_negatives:
                    raise ValueError(f'{name}: false negative or repeated nonduplicate anomaly')
                observed_negatives.add(triple)
            else:
                raise ValueError(f'{name}: unknown anomaly family {family}')
    return {'reference_positives_preserved': True,
        'known_positive_corruptions_rejected': True,
        'nonduplicate_negative_partitions_disjoint': True,
        'corruption_fingerprint': corruption_fingerprint(partitions)}
