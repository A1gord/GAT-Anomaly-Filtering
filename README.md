# Relation-Aware Knowledge Graph Structure Restoration

Implementation of the method described in **Relation-Aware Graph Attention Model for Knowledge Graph Structure Restoration under Structural Noise**, by Aleksei Golovin, Nataly Zhukova, and Tianxing Man.

The task is supervised classification of observed triples: `1` denotes a valid fact and `0` denotes an injected anomaly. The restored graph retains triples whose predicted validity is at least a threshold selected on validation data, then removes repeated assertions.

## Implementation scope

The repository includes the relation-aware GATv2 architecture, candidate-specific edge exclusion, controlled corruption, canonical benchmark partitions, weighted optimization, validation threshold selection, the architectural ablations, multi-seed experiments, anomaly-type reports, baseline adapters, and downstream RotatE evaluation.

CAGED is integrated through scores exported by its reference implementation. Its contrastive training algorithm is not reimplemented here: the supplied manuscript names the baseline but does not specify its objective or implementation. TMF constraints and text-derived entity vectors must also be supplied separately. No published result is embedded as an output of this implementation, and no claim is made that the manuscript's numerical results have been reproduced.

## Installation

Use Python 3.10 or newer with PyTorch, PyTorch Geometric, and scikit-learn:

```bash
python -m pip install -r requirements.txt
```

The default device is CPU. Select a compatible CUDA installation explicitly with `--device cuda`. The program fails if CUDA is requested but unavailable. Dependencies are lower bounds rather than a pinned, validated environment.

## Relation-aware architecture

For every relation `r`, the encoder learns a forward representation and a separate inverse representation. A further learned relation represents self-loops. An observed `(h, r, t)` creates incoming messages along `(h, r, t)` and `(t, inverse(r), h)`.

For layer `l` and head `k`, the implemented operations correspond to Equations (2)-(8):

```text
r_projected = W_R r
attention_logit = a^T LeakyReLU(W_Q h_target + W_K h_source + W_AR r_projected)
alpha = softmax_over_typed_incoming_edges(attention_logit)
gate = sigmoid(W_G r_projected)
message = gate * (W_N h_source) + W_M r_projected
u_target = W_S h_target + sum(alpha * message)
```

The three attention projections are the block decomposition of the manuscript's linear transformation of concatenated target, source, and projected relation representations. Relation transformations, gates, residual transformations, and attention vectors are head-specific. Intermediate heads are concatenated after ELU; final heads are averaged. Attention dropout and intermediate feature dropout are applied during training.

The edge classifier follows Equations (9)-(11):

```text
edge = contextual_head || candidate_relation || contextual_tail
logit = W_2 ReLU(W_1 edge + b_1) + b_2
valid_probability = sigmoid(logit)
```

Entity and relation dimensions are independent: the manuscript defaults are 128 and 64. The default encoder has two layers and eight heads. The classifier's hidden width is configurable and defaults to 256; this width is an implementation choice because the manuscript does not fix it.

### Candidate exclusion

Each candidate receives its own context with every occurrence of its exact `(head, relation, tail)` removed before inverse augmentation is used. The generated inverse counterpart is therefore excluded as well. Other predicates connecting the same entities remain available. No clean/anomalous labels determine which context edges are retained.

The sampler collects the incoming receptive field required by the encoder, without random neighbor truncation. Candidate graphs are joined as a disjoint union: they share learnable parameters and global entity IDs but have separate local adjacency. Excluding several candidates in one shared graph would remove extra evidence; this implementation instead excludes exactly the target fact in each candidate's context.

The relation-aware encoder uses `L` incoming expansion steps. GCN uses one additional expansion step to provide boundary degrees for normalization. Exact receptive fields can be large on dense graphs, and candidate-specific batching trades computation and memory for explicit exclusion. Lower `--batch-size` when necessary; lowering it does not introduce neighbor sampling.

## Input data and partitions

All triple files contain one assertion per line, without a header, in this order:

```text
head<TAB>relation<TAB>tail
```

Whitespace-separated triples are also accepted when each token has no whitespace. Entity and relation identifiers are strings. Numeric-header or alternative-order source formats must be normalized before use.

Public datasets use their canonical partitions:

```text
data/FB15K-237/train.txt
data/FB15K-237/valid.txt
data/FB15K-237/test.txt
```

The loader also accepts `.tsv` files and `validation` or `dev` as the validation filename. It rejects overlapping or repeated canonical clean facts. All known positive facts are used to construct the vocabulary and reject accidentally valid corruptions. Their validation and test edges are not inserted into the training message-passing context.

For a single custom or TMF clean file, unique reference facts are divided into 70%/15%/15% partitions **before** noise injection. `reference_split_seed` fixes this split independently of the corruption seed. A canonical TMF split can instead be supplied with `dataset_dir`. Input files and dataset downloads are not bundled. Missing data raises an error; random synthetic data is available only through explicit `--synthetic-if-missing 1` and is not a benchmark substitute.

The paper's reference statistics are:

| Dataset | Entities | Relations | Train | Valid | Test |
| --- | ---: | ---: | ---: | ---: | ---: |
| FB15K-237 | 14,541 | 237 | 272,115 | 17,535 | 20,466 |
| WN18RR | 40,943 | 11 | 86,835 | 3,034 | 3,134 |
| NELL-995 | 75,492 | 200 | 123,370 | 15,000 | 15,838 |
| TMF-KG | 10,284 | 62 | 24,788 | 5,312 | 5,312 |

Statistics in exported manifests are calculated from the supplied files. The loader does not manufacture missing entities or triples to match this table.

## Corruption protocol

Entity corruption replaces either the head or the tail of a selected positive triple, retaining its relation and other endpoint. Relation substitution replaces only the predicate with a different one. A nonduplicate anomaly is rejected if it is a known positive in **any** reference partition. Generated nonduplicate anomalies are also kept distinct across partitions.

Each partition uses its own deterministic random stream. The seeds for the five experimental realizations are `13, 37, 71, 101, 137`. Rejection against already generated anomalies prevents identical negative facts from leaking across partitions; this also means collisions can affect later streams.

`noise_ratio` means `injected_count / clean_partition_count`. For public benchmarks the injected count is rounded down to an even number after rounding the requested total, allowing exactly equal entity-corruption and relation-substitution counts. The actual ratio is recorded. Public profiles evaluate 5%, 10%, and 15% noise and exclude redundant assertions.

The TMF profile evaluates 10%, 20%, 30%, and 40% noise, including repeated assertions copied from the same clean partition. Its three corruption weights default to equal values. The manuscript does not specify the TMF mixture, so these weights are an explicit configurable choice. Generation fails when a requested corruption family cannot supply enough distinct negative facts; another family is never silently substituted.

### Duplicate identifiability

An original triple and its exact duplicate have identical inputs to the architecture in the manuscript. A classifier using only those inputs cannot consistently assign different labels to the occurrences. The implementation preserves the manuscript's occurrence labels for classification reports and separately deduplicates threshold-retained facts during graph export. It reports duplicate occurrences removed by this export operation. Anomaly classification F1 is never replaced with a deduplication score. Reproducing the paper's separate redundant-assertion detection numbers would require a specified occurrence feature, provenance signal, or evaluation convention beyond the described architecture.

## Training and inference

Train the proposed model on one corrupted benchmark realization:

```bash
python main.py --config configs/fb15k237.json --mode train --seed 13
```

Run another benchmark or the TMF case study:

```bash
python main.py --config configs/wn18rr.json --mode train
python main.py --config configs/nell995.json --mode train
python main.py --config configs/tmf.json --mode train
```

Defaults follow the stated proposed-model training configuration: batch size 256, Adam at `5e-4`, weight decay `1e-5`, dropout 0.20, at most 300 epochs, and early stopping after 20 epochs without validation anomaly-F1 improvement. Binary cross-entropy uses training-class weights `N / (2 * N_class)`. A ReduceLROnPlateau scheduler monitors validation anomaly F1; factor 0.5 and patience 5 are explicit implementation choices.

The threshold is selected on validation data at each epoch for checkpoint selection, then selected again after restoring the best epoch. Test scores never select a threshold, epoch, or learning rate. The default search has 37 values from 0.05 to 0.95. Equal-F1 thresholds are resolved by proximity to 0.5, then the smaller threshold. Disable search with `--optimize-threshold 0`.

`precision`, `recall`, and `f1` treat **anomalies** as the positive metric class, while the neural output remains a **validity** probability. `valid_precision`, `valid_recall`, and `valid_f1` report the other class explicitly. Confusion-matrix fields follow anomaly-positive semantics. The retained decision is `valid_probability >= threshold`.

Run inference with a trained neural checkpoint:

```bash
python main.py --config configs/predict.json
```

Inference uses the saved architecture, vocabulary, semantic-feature buffer, and validation threshold. By default its context is the observed inference graph with candidate-specific exclusion. Set `--inference-context training` to reuse the saved corrupted training context, matching the context policy used for held-out evaluation. Unknown identifiers raise an error by default. `--drop-unknown-in-predict 1` excludes unknown candidates and records their original positions with null probabilities in `prediction_alignment.json`.

Version-2 checkpoints embed the training context and mappings. Checkpoints from the previous relation-agnostic implementation must be retrained. Standalone prediction currently supports neural model checkpoints; the random forest, rule baseline, and external-score adapter are evaluated through the training/experiment pipeline and do not produce neural checkpoints.

Use `--output-dir` to move default artifacts together; explicit individual output paths take precedence. Every `Config` field has a corresponding CLI option using hyphens. Boolean CLI values are `0` and `1`; booleans in JSON are `false` and `true`. The aliases `--d`, `--L`, `--K`, and `--sigma` are retained. Unknown JSON fields raise an error. The old homogeneous edge-frequency weighting option is superseded by typed relational propagation.

## Semantic features

Supply an entity-to-vector JSON object:

```json
{
  "entity_a": [0.2, -0.1, 0.5],
  "entity_b": [0.4, 0.0, -0.3]
}
```

Use `--entity-features-path data/entity_features.json`. Vector width is inferred unless `semantic_dim` is provided explicitly. Missing entity vectors become zero; unknown entity keys and nonfinite values are rejected. A learned linear projection adds these vectors to learned entity embeddings.

The manuscript permits text-derived semantic features but does not identify a text encoder, corpus, or feature fusion formula. This repository accepts precomputed features and makes additive projection explicit. The semantic ablation requires real supplied features and fails early when none are available. A structural-only run remains available without them, and its metrics record that semantic features were not used.

## Experiments and ablations

Run five corruption realizations at every configured noise level for the selected models:

```bash
python main.py --config configs/fb15k237.json --mode experiment
python main.py --config configs/tmf.json --mode experiment
```

Each model receives the same deterministic reference split and corrupted candidate sets for a given dataset, seed, and noise ratio. Runs are written separately under:

```text
outputs/<dataset>/noise_<ratio>/seed_<seed>/<model>/<ablation>/
```

The aggregate report includes mean and sample standard deviation over seeds, with optional paired-seed comparisons. Individual reports retain thresholds, histories, per-family anomaly metrics, and fingerprints of the actual reference data.

The architectural ablations are:

| Name | Change |
| --- | --- |
| `full` | Complete relation-aware architecture |
| `without_relation_attention` | Remove relation terms from attention logits |
| `without_relation_messages` | Replace gated relation-conditioned messages with node messages |
| `without_inverse_encoding` | Share forward relation embeddings across both message directions |
| `relation_agnostic_gatv2` | Remove relation dependence from both attention and messages |
| `without_multi_head` | Use one attention head |
| `without_semantic_features` | Disable semantic projection |

Run the FB15K-237 ablation suite at 10% noise:

```bash
python main.py --config configs/ablations.json
```

Repeat for WN18RR by changing `dataset`, `dataset_dir`, and `entity_features_path`. The supplied ablation config expects externally prepared semantic vectors.

## Baselines

| CLI model name | Implementation |
| --- | --- |
| `proposed` | Relation-conditioned dynamic attention, feature gates, relation messages, and residual aggregation |
| `random_forest` | Balanced random forest over endpoint degree, clustering, common-neighbor, Jaccard, and overlap features |
| `transe` | Translational L1 plausibility with normalized entity vectors and a learned monotonic validity calibration |
| `gcn` | Graph convolutional layers and the shared edge classifier |
| `gatv2` | Relation-agnostic GATv2 propagation and the shared edge classifier |
| `rgcn` | Basis-decomposed relation-dependent graph convolution and the shared edge classifier |
| `compgcn` | Multiplicative entity-relation composition with direction-specific transformations and the shared classifier |
| `rule_based` | Supplied entity-type domain/range compatibility and hierarchy cycle constraints |
| `caged` | Strict adapter for reference CAGED validity probabilities |

Neural baselines use the same candidate exclusion, partitions, weighting, validation selection, and edge-classification task. TransE is optimized through supervised weighted BCE on calibrated translational scores. The compositional encoder is an edge-classification adaptation with a shared learned relation table. The manuscript does not fix baseline training details, R-GCN basis count, composition operator, random-forest size, or rule definitions; these implementations make those choices explicit and are not certified reproductions of external authors' code. R-GCN uses at most 30 bases; the forest defaults to 300 trees.

For TMF rules, use the schema in `configs/rules.example.json`, replace all example identifiers with actual graph identifiers, and supply `--rules-path`. Hierarchical relations are checked for a cycle after removing the candidate. The baseline raises an error when no supplied rule applies to a candidate; it does not invent TM Forum semantics. Add `rule_based` to `experiment_models` only after supplying a suitable rule file.

### Reference CAGED integration

Export deterministic candidate rows without training:

```bash
python main.py --config configs/fb15k237.json --mode prepare --seed 13
```

Use those exact rows and partitions in the reference CAGED implementation. Convert its output into a headered TSV with:

```text
partition<TAB>row<TAB>head<TAB>relation<TAB>tail<TAB>valid_probability
```

Partition names are `train`, `valid`, and `test`; `row` is zero-based. Supply exactly one finite validity probability in `[0, 1]` for every candidate row. Anomaly scores must be converted to validity probabilities using training/validation information only. The adapter checks row coverage, uniqueness, triple identity, and probability range.

```bash
python main.py --config configs/fb15k237.json --model-name caged --external-scores-path data/caged_scores.tsv
```

For an experiment suite, `external_scores_path` may contain `{dataset}`, `{seed}`, `{ratio}`, `{model}`, and `{ablation}` placeholders. Generate the score file for each exact realization before adding `caged` to `experiment_models`. The predefined benchmark configs include only locally implemented baselines.

## Downstream knowledge graph completion

Enable RotatE comparison on one restoration run:

```bash
python main.py --config configs/fb15k237.json --noise-ratio 0.1 --run-completion 1
python main.py --config configs/tmf.json --noise-ratio 0.4 --run-completion 1
```

For each run, three new RotatE models are trained on the clean, corrupted, and restored **training** facts. All three use the same clean validation and test facts, vocabulary, seed, and completion settings. Restored validation/test triples are never recycled as completion training data. Training assertions are deduplicated for completion.

The implementation uses complex entity embeddings, relation rotations, margin-distance scores, filtered negative sampling, and self-adversarial negative weights. Checkpoints are selected by clean validation MRR. Evaluation ranks both head and tail substitutions against the complete entity vocabulary, filters other known positive facts, and uses average ranks for ties. Reports include MRR, mean rank, Hits@1/3/10, and relative MRR gain from noisy to restored training data.

The manuscript identifies RotatE but does not give its hyperparameters. Defaults here are 128 complex dimensions, margin 9, 64 negative samples per positive, learning rate `5e-4`, batch size 256, and up to 300 epochs. These are configurable implementation choices. Ranking candidates are chunked with `ranking_batch_size`; the entire candidate entity set is still evaluated. Full validation ranking at every epoch and exact restoration contexts can make large experiments expensive.

With `--mode experiment --run-completion 1`, completion runs for every configured restoration realization. To reproduce the scope of Table 6, restrict public experiment ratios to `0.1`, or TMF ratios to `0.4`, using `--experiment-noise-ratios`.

## Output artifacts

| Artifact | Contents |
| --- | --- |
| `metrics.json` | Configuration, selected threshold, train/valid/test metrics, per-family test metrics, restoration counts, optional completion results |
| `history.json` | Training loss, validation anomaly metrics, thresholds, and learning rates |
| `dataset_stats.json` | Realized corruption counts, split sizes, vocabulary sizes, and reference fingerprint |
| `scores.tsv` | Partition, row, original triple, validity probability, and threshold decision |
| `candidates/*.tsv` | Labeled corrupted candidates, anomaly family, and stable row IDs |
| `reference/*.tsv` | Clean canonical or generated reference partitions |
| `restored/*.tsv` | Thresholded and deduplicated partitions |
| `filtered_triples.tsv` | Union of threshold-retained train/valid/test assertions, deduplicated |
| `best_model.pt` | Best neural state, architecture, mappings, context, threshold, and fingerprint |
| `mappings.json` | Readable entity/relation ID mappings |
| `experiment_runs.json` | Incrementally saved completed experiment records |
| `experiment_summary.json` | Aggregate classification, anomaly-family and completion results, with individual run references |

The combined `filtered_triples.tsv` includes held-out partitions for graph inspection and export. Use `restored/train.tsv` for a downstream training graph; the built-in completion workflow already respects this separation.

## Modules

| Module | Responsibility |
| --- | --- |
| `main.py` | Configuration-driven CLI dispatch |
| `config.py` | Typed settings, strict overrides, and validation |
| `types.py` | Immutable partition and candidate-batch contracts |
| `data.py` | Triple parsing, vocabulary, encoding, and explicit synthetic data |
| `datasets.py` | Canonical partition loading and clean-reference splitting |
| `corruption.py` | Family-specific anomaly generation and realized counts |
| `graph.py` | Indexed, cached relational context and candidate-specific exclusion |
| `sampling.py` | Exact receptive fields, resource-aware packing, and batch accounting |
| `features.py` | Semantic-vector validation and entity feature initialization |
| `attention.py` | Head-specific relation attention, gates, messages, and residuals |
| `model.py` | Stacked encoder and triple validity classifier |
| `optimization.py` | Adam, plateau scheduling, and best-state selection |
| `trainer.py` | Mini-batch optimization and batched scoring |
| `metrics.py` | Anomaly-positive metrics and validation threshold selection |
| `baselines.py` | Neural and structural comparison methods |
| `constraints.py` | TMF-compatible supplied domain/hierarchy constraints |
| `external.py` | Reference-model score alignment and validation |
| `ablations.py` | Explicit architectural variants |
| `completion.py` | RotatE optimization and clean/noisy/restored comparison |
| `ranking.py` | Filtered full-vocabulary completion ranking |
| `reporting.py` | Candidate/score exports, anomaly analysis, and aggregation |
| `integrity.py` | Runtime partition invariants and corrupted-data fingerprints |
| `checkpoints.py` | Versioned neural artifacts and reference fingerprints |
| `experiments.py` | Public facade for experiment execution |
| `pipeline.py` | Data preparation, training, inference, and completion coordination |
| `artifacts.py` | Consistent output directories and artifact naming |
| `io_utils.py` | JSON, TSV, mappings, and tensor checkpoint persistence |
| `seeding.py` | Python and PyTorch RNG initialization |
| `indices.py` | Packed incoming adjacency, origin identities, and relation statistics |
| `neighborhood.py` | Immutable candidate receptive fields and exact incoming expansion |
| `cache.py` | Thread-safe LRU eviction with entry and accounted-byte limits |
| `assembly.py` | Disjoint graph assembly and resource-budget contracts |
| `decisions.py` | Sorted cumulative confusion curves and deterministic threshold decisions |
| `score_diagnostics.py` | Tied-score ranking metrics, reliability, entropy, and risk/coverage summaries |
| `diagnostics.py` | Relation/degree slices and validation/test diagnostic reports |
| `statistics.py` | Occurrence bootstrap intervals and paired-seed comparisons |
| `registry.py` | Model capabilities and required external input contracts |
| `planning.py` | Immutable run identities and expansion of experiment axes |
| `execution.py` | Serial execution, verified reuse, and experiment-level consistency |
| `ledger.py` | Exclusive study ownership and durable run state transitions |
| `provenance.py` | Input, implementation, and configuration fingerprints |
| `environment.py` | Python, platform, and installed distribution metadata |
| `manifests.py` | Complete per-run artifact inventories and checksums |
| `telemetry.py` | Session-scoped structured training events |
| `gradients.py` | Finite-gradient inspection and optional norm clipping |
| `explanations.py` | Scoped capture of relation-aware attention and gate summaries |
| `storage.py` | Atomic artifact replacement with temporary sibling files |

## Reproduction status

The manuscript supplies the core architectural equations and proposed-model training settings. It leaves semantic feature extraction, the exact TMF mixture and constraints, scheduler details, threshold grid, several baseline settings, completion settings, and duplicate-occurrence discrimination unspecified. The choices and external requirements above are part of the executable protocol rather than claims about hidden experimental settings.

The source changes were prepared without executing project code or adding automated tests. Runtime behavior, dependency compatibility, performance, and manuscript scores have not been validated in this workspace.

## Extended execution and analysis

The indexed execution, statistical diagnostics, provenance ledger, and attention summaries extend the implementation around the paper's restoration method. They are engineering and analysis facilities, not additional architectural claims from the manuscript. The default relation-aware scoring equations, corruption definitions, class weighting, and validation-only selection remain the methodological baseline. Optional gradient clipping and resource-limited packing are explicit configuration choices.

### Indexed candidate contexts

`RelationalIndex` stores incoming arcs in packed arrays with prefix offsets rather than maintaining a separate Python adjacency list for every node. Each arc carries the identity of its originating fact. Candidate exclusion removes both directions by that shared identity, including every occurrence collapsed into the same fact, while preserving other predicates on the same endpoints.

`IncomingExpansion` returns immutable receptive fields containing global nodes, typed edges, frontier sizes, and the excluded origin identity. A per-context LRU cache stores those objects by candidate and depth. Defaults are 128 entries and 64 MiB of accounted neighborhood object storage. Shared integer objects can be counted repeatedly, making accounting conservative for those objects; the configured byte bound is not a process-RSS bound and does not include adjacency arrays, cache container overhead, tensors, or model memory. Cache counters report hits, misses, evictions, and rejected oversized entries. Set either cache limit to zero to disable storage.

`CandidateAssembler` maps each cached context to a disjoint local graph. `CandidateBatcher` preserves candidate order and emits a batch when any configured candidate, node, or edge limit would be exceeded. `batch_max_nodes=0` and `batch_max_edges=0` disable those two limits. A candidate whose complete context alone exceeds a configured budget raises `ContextBudgetExceeded`; the sampler does not truncate neighbors or weaken exclusion. Nonzero packing budgets can reduce the number of candidates per optimizer update and should be recorded when comparing runs.

The metrics artifact records packed index sizes, isolated entity counts, relation support, cache counters, cumulative batch counts, and peak batch sizes. These measurements describe the actual supplied graph and processed batches.

### Decision curves and probability diagnostics

Threshold evaluation uses sorted validity probabilities and cumulative class counts. For a threshold `tau`, anomalies are exactly the prefix with `valid_probability < tau`; tied scores equal to `tau` remain valid. This evaluates a threshold grid without constructing a candidate-by-threshold prediction matrix. Equal-F1 tie breaking follows the original proximity-to-0.5 policy.

By default, `diagnostics.json` contains separate descriptive validation and test reports:

- Anomaly AUROC and average precision, with equal-score groups handled together.
- Brier score, negative log likelihood, and equal-width reliability bins for validity probabilities.
- Expected calibration error and binary confidence/risk coverage summaries.
- The complete configured precision/recall/F1 decision curve.
- Relation-specific and logarithmic endpoint-degree slices, with class support for each group.
- Optional percentile bootstrap intervals for anomaly precision, recall, and F1.

Undefined ranking metrics are represented by null rather than a fabricated score. Group reports retain at most `diagnostic_max_groups` groups, ordered by occurrence support and identifier. Degree slices use the sum of endpoint typed incoming degrees in the corrupted training context; they are descriptive structural covariates, not additional classifier inputs.

Inverse-frequency weighted BCE changes the interpretation of neural sigmoid outputs. The reliability report measures their behavior but does not claim they are calibrated posterior probabilities or change the decision threshold. Risk/coverage orders occurrences by `max(p, 1-p)` and reports errors at the already selected threshold; this ordering is a descriptive confidence heuristic.

`bootstrap_samples=0` disables resampling. A positive value enables deterministic occurrence-level bootstrap intervals at `confidence_level`, with the validation-selected threshold held fixed. These intervals are conditional on the trained model and do not account for graph dependence, uncertainty from optimization, or threshold selection. The experiment summary additionally compares `proposed/full` against each available model or ablation on matching seeds and corruption fingerprints. Paired seed intervals resample seed-level F1 differences and are emitted only when at least two matched seeds are available. They are distinct from occurrence-level intervals, and neither interval is a significance claim.

### Durable experiment execution

`ExperimentPlan` expands the dataset, noise, seed, model, and ablation axes into immutable run identities and validates external model requirements before training begins. Duplicate run identities are rejected. The serialized plan is saved as `experiment_plan.json`.

A study owns an exclusive `.experiment.lock` in its output root. The lock contains a process identifier, creation time, and ownership token; it is removed on normal completion or handled failure. A lock left by a terminated process must be inspected and removed manually before restarting that study. Concurrent studies should use separate output roots. This mechanism does not lock independent single-run CLI invocations.

`experiment_ledger.json` records `running`, `complete`, and `failed` states. With `experiment_resume=true`, a completed run is reused only when its execution signature and all inventoried artifact checksums still match. The signature includes:

- Effective requested configuration, except the resume toggle.
- SHA-256 and sizes of relevant input files.
- SHA-256 of the Python source modules in `kg_filtering`.
- Python, platform, and installed dependency version metadata.

Inputs and implementation signatures are checked again after an executed run. Changes during execution fail the run instead of certifying a mixed-input result. Artifact inventories include metrics, scores, histories, mappings, reference/corrupted/restored partitions, filtered output, and enabled diagnostic, explanation, telemetry, and checkpoint files. A missing or modified inventoried file invalidates reuse. This resumes at the level of completed runs; interrupted epochs and optimizer states are not resumed. A failed or interrupted run is restarted from its configured seed.

The ledger is updated atomically after each transition. JSON, filtered graphs, candidate/score TSVs, and neural checkpoints are written through temporary sibling files and replaced after successful serialization and flushing. Individual artifact replacement is atomic; the complete set is not a single filesystem transaction, so the ledger certifies completion only after all required files exist. Training event logs are append-only streams with distinct session identifiers.

### Optimization telemetry

`training_events.jsonl` records training start, epoch completion, early stopping, completion, and failure. Epoch entries include loss, validation metrics, threshold, learning rate, optimizer-step count, training/validation elapsed time, gradient norm summaries, clipped-step counts, resource counters, and cache statistics. Counters are cumulative unless their field explicitly denotes an epoch summary.

Every optimization step checks the global parameter gradient norm for finiteness. `gradient_clip_norm=0.0` leaves finite gradients unchanged. A positive norm enables clipping and records how often it occurred. `record_telemetry=0` disables the event journal while retaining history and metrics artifacts. Timing is wall-clock observation rather than a synchronized GPU performance benchmark.

### Attention summaries

Set `explain_candidates` to a positive count to write `attention_explanations.json` for the first encoded test occurrences after training, or the first encoded input occurrences during prediction. `explanation_top_k` bounds the incoming edges reported for each endpoint, layer, and attention head.

The capture scope attaches a temporary observer to relation-aware attention layers, records normalized attention and mean relation-gate activations, and restores the previous observers and training mode afterward. The report names original entity and relation identifiers, distinguishes forward, inverse, and self relations, and includes the retained attention mass of the reported top edges. This observes the model without changing its state dictionary or inference equations. It captures attention into the candidate endpoints at each layer, rather than every intermediate node in the receptive field. Attention weights are descriptive, not causal explanations of the final classification. Models without these relation-aware layers return an explicit unavailable result.

During inference, explanation records also include original input row indices so that optional unknown-token removal does not destroy alignment. Inference metrics include an unlabeled probability profile with quantiles, entropy, and histogram counts.

### Extended configuration

| Field | Default | Purpose |
| --- | ---: | --- |
| `context_cache_entries` | 128 | Maximum cached receptive fields |
| `context_cache_bytes` | 67108864 | Accounted neighborhood storage limit |
| `batch_max_nodes` | 0 | Maximum total nodes per candidate batch; zero disables |
| `batch_max_edges` | 0 | Maximum total edges per candidate batch; zero disables |
| `gradient_clip_norm` | 0.0 | Optional global gradient norm clipping |
| `record_telemetry` | true | Append structured training events |
| `enable_diagnostics` | true | Export descriptive validation/test diagnostics |
| `diagnostic_bins` | 15 | Reliability and probability histogram bins |
| `diagnostic_max_groups` | 128 | Maximum relation or structural groups per report |
| `bootstrap_samples` | 0 | Optional occurrence and paired-seed resamples |
| `confidence_level` | 0.95 | Percentile interval coverage |
| `experiment_resume` | true | Reuse checksum-verified completed study runs |
| `explain_candidates` | 0 | Number of candidates receiving attention summaries |
| `explanation_top_k` | 8 | Incoming edges retained per endpoint/head summary |

`configs/advanced.json` demonstrates a resumable multi-model study with explicit packing budgets, larger cache limits, bootstrap analysis, and attention summaries. The budget values are configuration examples rather than validated memory recommendations; some exact contexts may exceed them. These facilities add reporting and hashing overhead, and bootstrap resampling adds work proportional to the evaluation sample count and requested resamples. No project execution or runtime validation was performed while preparing these extensions.
