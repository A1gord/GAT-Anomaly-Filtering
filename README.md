# GAT-Anomaly-Filtering

This repository implements a GAT-based anomaly filtering method for knowledge graphs. The code is structured into several modules for data processing, model definition, training, and evaluation.  

## Structure

- `main.py` CLI entrypoint
- `kg_filtering/config.py` configuration schema and overrides
- `kg_filtering/data.py` triple loading, anomaly injection, splits, encoding, graph build
- `kg_filtering/model.py` custom GAT/GATv2 encoder and edge classifier
- `kg_filtering/metrics.py` weighted metrics and threshold search
- `kg_filtering/trainer.py` training loop, early stopping, evaluation
- `kg_filtering/pipeline.py` train and predict pipelines
- `kg_filtering/io_utils.py` exports, checkpoint and mappings I/O
