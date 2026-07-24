# Multimodal Expansion Plan and Implementation Structure

## 1. Objective

Evolve `agent_system` from a tabular-only research system into a
modality-neutral system that can search, evaluate, fine-tune, promote, prune,
and ensemble solutions for:

- tabular data;
- images;
- audio;
- video; and
- combinations of tabular, text, image, audio, and video inputs.

The expansion should preserve the parts of the current architecture that are
already modality-independent:

- manager-owned execution and experiment budgets;
- tree-shaped execution lineage;
- the separate provenance DAG;
- statistically grounded pruning and promotion;
- diversity-aware candidate selection;
- information-gain-aware exploration;
- tuning-history reuse;
- supervised subprocess execution; and
- memory-pool retrieval and empirical validation.

The main change is to replace tabular-specific data, evaluation, artifact, and
prompt contracts with registered modality implementations. This is an
incremental migration, not a parallel rewrite.

## 2. Architectural Decisions

### 2.1 Keep one shared search system

The scheduler must not acquire branches such as `if image`, `if audio`, or
`if video`. It should continue to consume the same modality-neutral evidence:

- score and metric direction;
- uncertainty;
- fidelity;
- experiment cost;
- lineage history;
- diversity;
- information gain; and
- resource feasibility.

Modality-specific logic belongs below the scheduler in dataset, fidelity,
evaluation, and ensemble implementations. A scheduler change is accepted only
if held-out benchmarks show a measurable improvement in search efficiency.

### 2.2 Separate modality, problem type, and output type

The current broad `task_type` field mixes assumptions that will not scale.
Future tasks should be described along independent dimensions:

| Dimension | Examples |
|---|---|
| Modality | `tabular`, `image`, `audio`, `video`, `multimodal` |
| Component modalities | `["image", "text"]`, `["audio", "tabular"]` |
| Problem type | `classification`, `regression`, `multilabel_classification` |
| Structured problem type | `detection`, `segmentation`, `retrieval`, `captioning`, `temporal_localization` |
| Output type | `class_probabilities`, `continuous`, `boxes`, `masks`, `embeddings`, `ranked_items`, `text` |
| Sample unit | row, image, recording, clip, entity with several aligned inputs |

Initial multimodal support should cover classification and regression. Structured
vision, audio, video, and generative tasks should be added only after the shared
contracts are stable.

### 2.3 Preserve tabular behavior as the first adapter

The existing tabular flow becomes the first implementation of the new
interfaces. Compatibility shims keep existing tasks and tests working while
callers migrate. No phase may require all existing task configurations to be
rewritten at once.

### 2.4 Keep data access and evaluation harness-owned

Generated code should define the model and training procedure, but it should
not independently rediscover files, invent splits, or write ad hoc evaluation
artifacts.

The harness should own:

- task discovery and manifest validation;
- sample indexing and stable IDs;
- train/validation/test splits;
- modality-appropriate decoding and batching;
- fidelity limits;
- prediction validation;
- metric calculation; and
- normalized result artifacts.

This preserves evaluation fairness and prevents every generated solution from
implementing a subtly different loader.

### 2.5 Keep model combination manager-owned

When two nodes are merged, the ManagerAgent should create an ensemble of their
trained model artifacts. It should not ask an LLM to fuse their source code.

The default merge operator is cross-validated stacking when enough aligned OOF
evidence exists. Constrained blending or an output-specific ensemble is the
fallback. A merge node records:

- its execution parent in the execution tree;
- every contributing node in the provenance DAG;
- component model/checkpoint references;
- the fitted combiner;
- compatibility checks;
- OOF evidence used to fit the combiner; and
- the final inference order.

This produces a deployable `EnsembleBundle`, not only an averaged submission
file.

### 2.6 Add modalities in increasing order of operational cost

The recommended order is:

1. extract modality-neutral contracts and prove tabular parity;
2. image classification;
3. audio classification and regression;
4. video classification;
5. late-fusion multimodal classification and regression;
6. intermediate fusion with reusable encoders;
7. structured and generative tasks.

Video and end-to-end multimodal training are deliberately later because they
introduce the greatest decoding, storage, VRAM, and evaluation costs.

## 3. Current State and Required Changes

| Area | Current state | Required target |
|---|---|---|
| File discovery | Flat allowlist of table/array/text files | Manifest-driven, recursive modality-aware discovery |
| Dataset analysis | `pandas.DataFrame` schema and target inference | Registered analyzers returning a common task profile |
| Task taxonomy | Classification, regression, supervised, clustering | Independent modality, problem, target, and output schemas |
| Loader contract | Dict of DataFrames and NumPy arrays | Lazy `DatasetBundle` with sample/entity IDs and split views |
| Evaluation | Row subsets, tabular folds, scalar metrics | Registered split, fidelity, metric, and artifact handlers |
| Predictions | `oof_predictions.csv` and `submission.csv` | Typed prediction bundle plus compatibility CSV export |
| Initial generation | Tabular-specific loader and baseline prompts | Shared runner contract plus modality prompt context |
| Implementation | Tabular pipeline generation and repair | Model/training code against a typed data/evaluation API |
| Validation | Tabular leakage and path-write checks | Shared safety checks plus modality-specific leakage checks |
| Aggregation | Numeric CSV averaging/rank averaging | Output-specific ensemble registry and deployable bundle |
| Memory pool | Mainly tabular L1 categories | Modality/problem/output compatibility and empirical history |
| Artifact verification | Contract runtime already recognizes several modalities | Make explicit contracts mandatory for non-legacy artifacts |
| Resources | CPU/GPU and RAM feasibility | Add VRAM, decode load, disk/cache, workers, and media duration |
| Task inputs | Per-node flat file links | Harness-owned lazy access and one shared indexed cache |

The existing memory-pool verification runtime is a useful head start: it already
recognizes image, audio, video, text, and multimodal input contracts. The runtime
task path and agent prompts are the larger remaining sources of tabular coupling.

## 4. Target Runtime Flow

```text
task_config.json + task-owned inputs
                  |
                  v
        TaskSpec parser/validator
                  |
                  v
          ModalityRegistry
                  |
       +----------+-----------+
       |          |           |
       v          v           v
  discovery   dataset     evaluation
  + profile   adapter      protocol
       |          |           |
       +----------+-----------+
                  |
                  v
       harness-owned DatasetBundle
       + SplitPlan + FidelityProfile
                  |
                  v
       Initial/Technique/Implementation
       agents generate model procedure
                  |
                  v
       supervised model execution
                  |
                  v
       PredictionBundle + ResultRecord
                  |
          +-------+--------+
          |                |
          v                v
  shared statistical   Manager-owned
  search policies      EnsembleBundle
          |                |
          +-------+--------+
                  |
                  v
       execution tree + provenance DAG
```

The manager, search policies, and provenance store consume normalized result
records and do not need to understand image pixels, waveforms, or video frames.

## 5. Proposed Project Structure

The structure below is intentionally small. New files are introduced only where
they replace repeated modality conditionals.

```text
agent_system/
├── core/
│   ├── contracts.py              # TaskSpec, DatasetBundle, ResultRecord
│   └── modality_registry.py      # Adapter/evaluator/ensemble registration
│
├── modalities/
│   ├── base.py                   # Protocols and shared validation
│   ├── tabular.py                # Current behavior behind the new protocols
│   ├── image.py                  # Image indexing, decode, transforms, profile
│   ├── audio.py                  # Waveform indexing, decode, resampling, profile
│   ├── video.py                  # Video metadata, deterministic clip sampling
│   └── multimodal.py             # Entity joins and aligned component adapters
│
├── evaluation/
│   ├── runner.py                 # Shared evaluation lifecycle
│   ├── splitters.py              # Stratified/group/entity/time-aware splits
│   ├── fidelity.py               # Modality-aware fidelity profiles
│   ├── metrics.py                # Metric registry and direction
│   └── prediction_io.py          # PredictionBundle validation and storage
│
├── ensemble/
│   ├── registry.py               # Output-type to ensemble strategy mapping
│   ├── stacking.py               # Cross-fitted meta-model and convex fallback
│   └── structured.py             # Masks, boxes, embeddings, ranked outputs
│
├── resources/
│   └── estimator.py              # RAM/VRAM/decode/cache feasibility estimates
│
├── agents/
│   ├── task_analyzer.py          # New registry-driven analyzer
│   ├── data_analyzer.py          # Temporary tabular compatibility wrapper
│   ├── prompt_context.py         # Modality/problem-specific constraints
│   ├── manager_agent.py
│   ├── initial_agent.py
│   ├── technique_agent.py
│   ├── implementation_agent.py
│   ├── aggregator_agent.py
│   └── validation_guard.py
│
├── evaluation_contract.py        # Compatibility facade during migration
├── tasks/<task_name>/
│   ├── task_config.json
│   ├── task_description.md
│   └── input/
└── tests/
    ├── fixtures/modalities/
    ├── test_modality_contracts.py
    ├── test_prediction_bundles.py
    ├── test_multimodal_leakage.py
    └── test_multimodal_smoke.py
```

`evaluation_contract.py` should remain as a facade until existing generated
algorithms and tests have migrated. Its tabular public functions can delegate to
the new evaluation runner.

## 6. Core Contracts

### 6.1 TaskSpec

`TaskSpec` becomes the validated, immutable task description used by all agents.

```python
@dataclass(frozen=True)
class TaskSpec:
    schema_version: int
    task_id: str
    modality: str
    component_modalities: tuple[str, ...]
    problem_type: str
    inputs: Mapping[str, "InputSpec"]
    target: "TargetSpec | None"
    sample_id_field: str
    entity_id_field: str | None
    group_id_field: str | None
    time_field: str | None
    output: "OutputSpec"
    metrics: tuple["MetricSpec", ...]
    primary_metric: str
    resource_limits: "ResourceLimits"
```

Important rules:

- `modality="multimodal"` requires two or more named inputs.
- All components of one entity share one stable `entity_id`.
- `problem_type` does not imply a data container.
- `primary_metric` must be present in `metrics`.
- The metric direction is explicit; it is never inferred from a score name
  inside the scheduler.
- Legacy tabular configurations are translated into this contract by a
  compatibility parser.

### 6.2 InputSpec

```python
@dataclass(frozen=True)
class InputSpec:
    name: str
    modality: str
    role: str
    source: str
    format: str
    id_field: str | None = None
    required: bool = True
    options: Mapping[str, object] = field(default_factory=dict)
```

Examples of `options` include image color mode, expected audio sample rate,
video clip policy, text field names, and tabular categorical columns. These are
validated by the owning adapter, not by the scheduler.

### 6.3 DatasetBundle

`DatasetBundle` must be lazy. It contains indices and factories, not a fully
decoded image/audio/video population.

```python
@dataclass
class DatasetBundle:
    task: TaskSpec
    dataset_fingerprint: str
    sample_index: "SampleIndex"
    train_ids: Sequence[str]
    test_ids: Sequence[str]
    targets: "TargetView | None"
    adapter: "ModalityAdapter"

    def make_dataset(self, sample_ids, *, mode, fidelity): ...
    def make_loader(self, sample_ids, *, mode, fidelity, seed): ...
```

The stable unit is `sample_id`; for multimodal tasks it maps to an `entity_id`
and several named component references. The harness controls training versus
evaluation mode so random augmentation cannot leak into validation.

### 6.4 SplitPlan

```python
@dataclass(frozen=True)
class SplitPlan:
    split_fingerprint: str
    assignments: Mapping[str, int]
    strategy: str
    seed: int
    group_field: str | None
    leakage_unit: str
```

Every node compared or ensembled under one evaluation protocol must use the
same dataset fingerprint and split fingerprint.

### 6.5 FidelityProfile

Fidelity is a vector rather than only a row fraction and fold count.

```python
@dataclass(frozen=True)
class FidelityProfile:
    name: str
    sample_fraction: float
    folds: int
    max_epochs: int
    max_trials: int
    spatial_size: tuple[int, int] | None = None
    audio_sample_rate: int | None = None
    max_audio_seconds: float | None = None
    video_frames: int | None = None
    video_fps: float | None = None
    clips_per_video: int | None = None
```

The scheduler sees a fidelity name and comparable cost; the evaluator resolves
the actual modality fields. Promotions are valid only when the promoted run
uses a registered higher fidelity from the same protocol.

### 6.6 PredictionBundle

CSV cannot represent every future output. Store a manifest plus an
output-specific payload.

```python
@dataclass(frozen=True)
class PredictionBundle:
    schema_version: int
    task_fingerprint: str
    split_fingerprint: str
    output_type: str
    sample_ids: Sequence[str]
    payload_path: Path
    payload_format: str
    class_names: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)
```

Recommended payloads:

| Output | Storage |
|---|---|
| Scalar or class probabilities | compressed NumPy or Parquet |
| Multilabel probabilities | compressed NumPy plus label metadata |
| Embeddings | memory-mappable NumPy |
| Segmentation masks/logits | chunked arrays or per-sample file references |
| Detection boxes | JSONL or Parquet with sample IDs |
| Ranked items | JSONL |
| Generated text | JSONL |

Continue producing `oof_predictions.csv` and `submission.csv` for compatible
legacy tabular tasks. They become exports from `PredictionBundle`, not the
internal source of truth.

### 6.7 ResultRecord

Each run should emit a normalized record:

```json
{
  "schema_version": 2,
  "status": "success",
  "task_fingerprint": "...",
  "split_fingerprint": "...",
  "modality": "image",
  "problem_type": "classification",
  "output_type": "class_probabilities",
  "primary_metric": "roc_auc",
  "direction": "maximize",
  "score": 0.8123,
  "cv_mean": 0.8123,
  "cv_std": 0.0041,
  "folds": 3,
  "fidelity": "medium",
  "runtime_seconds": 420.0,
  "peak_ram_gb": 7.2,
  "peak_vram_gb": 5.1,
  "decoded_media_seconds": 18000,
  "prediction_bundle": "predictions/manifest.json",
  "model_bundle": "model/manifest.json"
}
```

This is the only result shape consumed by statistical search policy.

### 6.8 ModelBundle and EnsembleBundle

A successful node should describe its inference artifact:

```json
{
  "bundle_type": "model",
  "model_family": "image_classifier",
  "task_fingerprint": "...",
  "output_type": "class_probabilities",
  "checkpoint_paths": ["fold_0.pt", "fold_1.pt"],
  "preprocessing": "preprocessing.json",
  "entrypoint": "inference.py:predict",
  "dependencies": ["torch", "torchvision"]
}
```

A merge node writes:

```json
{
  "bundle_type": "ensemble",
  "strategy": "cross_validated_stacking",
  "component_nodes": ["node_4", "node_7"],
  "component_bundles": ["../node_4/model/manifest.json", "../node_7/model/manifest.json"],
  "combiner": "combiner.joblib",
  "output_type": "class_probabilities",
  "compatibility_key": "...",
  "inference_order": ["node_4", "node_7", "combiner"]
}
```

This keeps node merging model-based and manager-controlled without raw code
fusion.

## 7. Modality Adapter Protocol

```python
class ModalityAdapter(Protocol):
    name: str

    def validate_task(self, task: TaskSpec) -> None: ...
    def discover(self, task_dir: Path, config: Mapping) -> TaskSpec: ...
    def build_index(self, task: TaskSpec) -> SampleIndex: ...
    def profile(self, bundle: DatasetBundle) -> TaskProfile: ...
    def make_dataset(
        self,
        bundle: DatasetBundle,
        sample_ids: Sequence[str],
        *,
        mode: Literal["train", "evaluation", "inference"],
        fidelity: FidelityProfile,
    ) -> object: ...
    def estimate_resources(
        self, bundle: DatasetBundle, fidelity: FidelityProfile
    ) -> ResourceEstimate: ...
    def fingerprint(self, task: TaskSpec, index: SampleIndex) -> str: ...
```

The registry resolves one top-level adapter:

```python
registry.register("tabular", TabularAdapter())
registry.register("image", ImageAdapter())
registry.register("audio", AudioAdapter())
registry.register("video", VideoAdapter())
registry.register("multimodal", MultimodalAdapter(registry))
```

`MultimodalAdapter` composes component adapters and performs entity alignment.
It should not duplicate image/audio/video decoding logic.

## 8. Task Configuration

Keep `task_config.json` as the authoritative format to avoid introducing
another required parser. Add `schema_version` and retain a legacy translator.

### 8.1 Image classification

```json
{
  "schema_version": 2,
  "modality": "image",
  "problem_type": "classification",
  "inputs": {
    "image": {
      "source": "input/images",
      "format": "directory",
      "manifest": "input/labels.csv",
      "path_field": "image_path"
    }
  },
  "sample_id_field": "image_id",
  "target": {
    "source": "input/labels.csv",
    "field": "label"
  },
  "output": {
    "type": "class_probabilities"
  },
  "metrics": [
    {"name": "roc_auc", "direction": "maximize"}
  ],
  "primary_metric": "roc_auc"
}
```

### 8.2 Audio classification

```json
{
  "schema_version": 2,
  "modality": "audio",
  "problem_type": "classification",
  "inputs": {
    "audio": {
      "source": "input/metadata.csv",
      "format": "file_manifest",
      "path_field": "audio_path",
      "sample_rate": 16000
    }
  },
  "sample_id_field": "recording_id",
  "group_id_field": "speaker_id",
  "target": {
    "source": "input/metadata.csv",
    "field": "label"
  },
  "output": {
    "type": "class_probabilities"
  },
  "metrics": [
    {"name": "macro_f1", "direction": "maximize"}
  ],
  "primary_metric": "macro_f1"
}
```

### 8.3 Video classification

```json
{
  "schema_version": 2,
  "modality": "video",
  "problem_type": "classification",
  "inputs": {
    "video": {
      "source": "input/videos.csv",
      "format": "file_manifest",
      "path_field": "video_path",
      "clip_policy": "uniform"
    }
  },
  "sample_id_field": "video_id",
  "group_id_field": "source_video_id",
  "target": {
    "source": "input/videos.csv",
    "field": "label"
  },
  "output": {
    "type": "class_probabilities"
  },
  "metrics": [
    {"name": "accuracy", "direction": "maximize"}
  ],
  "primary_metric": "accuracy"
}
```

### 8.4 Image and tabular multimodal classification

```json
{
  "schema_version": 2,
  "modality": "multimodal",
  "component_modalities": ["image", "tabular"],
  "problem_type": "classification",
  "inputs": {
    "image": {
      "modality": "image",
      "source": "input/entities.csv",
      "format": "file_manifest",
      "path_field": "image_path"
    },
    "metadata": {
      "modality": "tabular",
      "source": "input/entities.csv",
      "format": "csv",
      "feature_fields": ["age", "location", "device_type"]
    }
  },
  "sample_id_field": "entity_id",
  "entity_id_field": "entity_id",
  "target": {
    "source": "input/entities.csv",
    "field": "label"
  },
  "output": {
    "type": "class_probabilities"
  },
  "metrics": [
    {"name": "roc_auc", "direction": "maximize"}
  ],
  "primary_metric": "roc_auc",
  "missing_modalities": {
    "policy": "mask"
  }
}
```

Automatic discovery may propose a manifest for conventional layouts, but the
resolved `TaskSpec` must always be persisted and validated before the first
experiment.

## 9. Evaluation by Modality

### 9.1 Shared rules

All evaluators must:

- create deterministic sample/entity IDs;
- persist dataset and split fingerprints;
- fit preprocessing and augmentation only on training folds;
- validate exact prediction alignment;
- reject NaN, infinite, malformed, or missing predictions;
- report uncertainty from comparable folds or repeated seeds;
- record resource consumption;
- prevent validation/test targets from reaching model training; and
- write a normalized prediction and result bundle.

Statistical comparisons, promotion, and ensembling are allowed only for runs
sharing the same compatibility key:

```text
task fingerprint
+ split fingerprint
+ target schema
+ output schema
+ metric protocol
+ fidelity-comparison rules
```

### 9.2 Tabular

Move the existing row-subset, fold, leakage, and metric behavior behind
`TabularAdapter` and the shared evaluation runner. Preserve current CSV exports
and generated-code compatibility until migration is complete.

### 9.3 Image

First release:

- classification and regression;
- directory or manifest-based image references;
- RGB/grayscale validation;
- deterministic validation transforms;
- training-only random augmentation;
- stratified or group-aware splits; and
- resolution plus sample fraction as fidelity controls.

Later:

- multilabel classification;
- segmentation with IoU/Dice metrics;
- detection with mAP metrics; and
- image-text retrieval.

Leakage checks must detect duplicate/near-duplicate image groups when group
metadata or hashes are available.

### 9.4 Audio

First release:

- waveform classification and regression;
- duration/channel/sample-rate profiling;
- deterministic resampling;
- padded or cropped batches with masks;
- speaker/session/group-aware splits; and
- sample rate, maximum duration, and sample fraction as fidelity controls.

Augmentation such as noise, time masking, or pitch changes is training-only.
Speaker or recording fragments must not cross folds.

### 9.5 Video

First release:

- clip/video classification;
- metadata indexing without decoding all frames;
- deterministic validation clips;
- group splits at source-video or entity level;
- bounded decode workers and cache size; and
- frame count, resolution, FPS, clips per video, and sample fraction as
  fidelity controls.

Random training clips are allowed, but the seed and sampling policy must be
recorded. Frames or clips from one source video must not cross folds.

### 9.6 Multimodal

First release uses late fusion:

1. train or reuse one model per component modality;
2. emit aligned OOF predictions or embeddings by entity ID;
3. fit a manager-owned combiner on OOF data;
4. handle missing components with explicit masks;
5. report individual-modality ablations; and
6. persist one `EnsembleBundle` containing every component.

Every split is performed at the entity level before modality-specific datasets
are constructed. It is invalid to split each modality independently.

After late fusion is reliable, add:

- frozen-encoder embedding fusion;
- a small learned fusion head;
- modality dropout;
- calibrated gating for missing inputs; and
- end-to-end cross-modal fine-tuning only when it beats late fusion under equal
  compute.

Text support should be implemented as a component adapter needed by multimodal
tasks even if standalone text search is not an initial release target.

## 10. Agent Changes

### 10.1 ManagerAgent

Responsibilities to add:

- load and persist `TaskSpec`;
- resolve registered modality components;
- pass a compact task/profile contract to other agents;
- check operator and artifact compatibility before materializing a node;
- compare candidates only under compatible evaluation protocols;
- create manager-owned ensemble merge nodes;
- record merge contributors in the provenance DAG;
- keep a single execution parent in the execution tree; and
- exclude infeasible candidates before spending an implementation experiment.

Responsibilities that should remain unchanged:

- experiment budget accounting;
- statistical evidence collection;
- promotion and pruning decisions;
- diversity and information-gain policies;
- tuning-history reuse; and
- tree scheduling.

### 10.2 TaskAnalyzer

Replace direct CSV assumptions with:

```text
TaskAnalyzer
  -> parse legacy or v2 task config
  -> select adapter from ModalityRegistry
  -> discover/validate inputs
  -> build stable sample index
  -> produce TaskProfile
  -> persist resolved_task_spec.json
  -> persist dataset_profile.json
```

`data_analyzer.py` remains a temporary wrapper around the tabular adapter.

Profiles should be compact and modality-specific:

- tabular: row count, data types, cardinality, missingness;
- image: count, sizes, channels, corrupt files, label balance;
- audio: count, duration, channels, sample rates, label/group balance;
- video: count, duration, FPS, resolution, codec readability;
- multimodal: entity coverage, join cardinality, missing components, label
  balance.

### 10.3 InitialAgent

Change from generating both an arbitrary loader and algorithm to generating a
model/training module against a harness-owned `DatasetBundle`.

The prompt should include:

- `TaskSpec`;
- compact `TaskProfile`;
- fidelity limits;
- resource limits;
- available dependency profile;
- required model and prediction bundle contracts; and
- modality-specific correctness rules from `prompt_context.py`.

Baseline templates should be deterministic and dependency-light. Use one
reference baseline per supported modality/problem slice before allowing a fully
generated alternative.

### 10.4 TechniqueAgent

Memory retrieval must filter on:

- component modality or modalities;
- problem type;
- output type;
- artifact scope;
- accelerator compatibility;
- resource profile;
- fidelity feasibility; and
- empirical history on similar task signatures.

The agent may propose a modality-specific technique only if its model card
declares compatible input and output contracts. Web research and L2 building
should use modality-aware queries rather than the current tabular-only prompt.

### 10.5 ImplementationAgent

Generated implementations should receive typed runtime objects and implement a
small interface such as:

```python
def build_model(context: RunContext) -> TrainableModel: ...
def train_fold(model, train_loader, valid_loader, context) -> FoldArtifact: ...
def predict(model, loader, context) -> PredictionPayload: ...
```

The harness should call these functions, validate outputs, calculate metrics,
and write result artifacts. This removes file discovery, fold creation, and
score calculation from generated code.

Repair prompts should include the failed contract name and modality-specific
diagnostics, such as a corrupt image, inconsistent audio length, decode failure,
tensor shape mismatch, or missing multimodal entity.

### 10.6 ValidationGuard

Split validation into:

- shared code safety and write-boundary rules;
- task-contract validation;
- model/prediction schema validation; and
- registered modality leakage rules.

Examples:

| Modality | Required leakage checks |
|---|---|
| Image | validation augmentation, duplicate group crossing folds, test-informed normalization |
| Audio | speaker/session crossing folds, full-dataset normalization, augmented validation audio |
| Video | clips from one source crossing folds, nondeterministic validation sampling |
| Multimodal | the same entity crossing folds through different component tables |

AST checks alone are insufficient for entity-level leakage. The evaluation
harness should also validate split assignments dynamically.

### 10.7 AggregatorAgent

Refactor aggregation into an output-type strategy registry:

| Output type | Preferred merge | Fallback |
|---|---|---|
| Class probabilities | cross-validated stacking | constrained weighted average |
| Regression values | regularized stacking | non-negative weighted average |
| Multilabel probabilities | per-label or joint stacking | weighted average |
| Segmentation logits | calibrated pixel/logit blend | mean logits |
| Detection boxes | weighted box fusion | class-aware NMS merge |
| Embeddings | normalized learned fusion | normalized average/concatenation |
| Ranked items | learned reranker | rank aggregation |
| Generated text | validation-trained selection/reranking | keep strongest compatible node |

The manager chooses nodes and invokes the aggregator. The aggregator validates
compatibility, fits the combiner using aligned OOF artifacts, and creates an
`EnsembleBundle`. It must not merge source code.

For small OOF sets, high-dimensional outputs, or unstable meta-learners, use a
regularized convex blend. The more complex stacking strategy must demonstrate
an uncertainty-aware improvement before promotion.

### 10.8 SetupAgent and dependencies

Avoid installing all media libraries for every task. Introduce allowlisted,
pinned dependency profiles:

```text
requirements/
├── core.txt
├── tabular.txt
├── image.txt
├── audio.txt
├── video.txt
└── multimodal.txt
```

Initial candidates:

- image: Pillow and torchvision;
- audio: torchaudio and soundfile;
- video: PyAV plus the shared tensor/image stack;
- multimodal: only the component profiles at first.

Additional model libraries should enter the allowlist only with a verified
memory-pool artifact. Pretrained weights require a controlled model registry,
recorded version/license, checksum, and local cache. Node code must not silently
download weights.

## 11. Memory Pool and Tuning History

### 11.1 Model-card changes

Make the following fields required for all new artifacts:

```json
{
  "interface": {
    "input_contract": {
      "modality": "image",
      "container": "tensor",
      "parameter_roles": {
        "train": "train.image",
        "test": "test.image",
        "target": "target"
      }
    },
    "output_contract": {
      "kind": "probabilities",
      "aligned_to": "test",
      "value_type": "probability"
    }
  },
  "capabilities": {
    "modalities": ["image"],
    "problem_types": ["classification"],
    "output_types": ["class_probabilities"],
    "supported_operators": ["refine", "tune", "diversify"]
  },
  "resource_profile": {
    "accelerator": "cuda",
    "min_ram_gb": 8,
    "min_vram_gb": 4,
    "estimated_runtime_seconds": 600,
    "decode_profile": "image"
  }
}
```

Legacy tabular cards may be normalized on read. The current synthetic verifier
should be extended, not replaced, and should make explicit modality contracts
mandatory for new artifacts.

### 11.2 Indexing

Do not create unrelated isolated memory pools. Add compatibility fields to the
existing index:

```text
modality -> problem type -> output type -> scope -> artifact
```

Categories can still describe technique families, but category alone must not
determine compatibility.

### 11.3 Knowledge-guided fine-tuning

Store trials under a reusable signature:

```text
model family
+ artifact version
+ modality/problem/output
+ dataset profile bucket
+ fidelity
+ accelerator class
```

For a new node:

1. retrieve successful and failed trials from compatible signatures;
2. warm-start the search distribution from robust historical regions;
3. reuse exact trials only when parameter semantics and fidelity match;
4. penalize configurations that repeatedly fail resource limits;
5. keep task-local OOF evidence authoritative; and
6. publish results back to global tuning history after validation.

Image resolution, audio sample rate, video clip policy, and fusion architecture
are fidelity or structural choices, not ordinary scalar hyperparameters. Their
history should be reused only across compatible task profiles.

## 12. Resource and Data-Access Design

### 12.1 Lazy access

Never load an entire image, audio, or video corpus during discovery. Index paths
and metadata, then decode batches on demand.

Generated model code should consume harness-created datasets/loaders rather
than raw task directories. This avoids duplicating millions of file links per
node and limits accidental writes to task-owned data.

### 12.2 Shared cache

Create one content-addressed, run-owned cache:

```text
runs/<task>/_cache/<dataset-fingerprint>/<adapter-version>/
```

Cache entries may include decoded metadata, resized images, resampled audio, or
video clip indices. Requirements:

- cache building is deterministic;
- cache keys include every transformation that affects content;
- entries become read-only after successful construction;
- cache time and bytes are measured separately;
- all compared nodes receive the same cache opportunity; and
- eviction respects a task-configured disk budget.

### 12.3 Resource estimates

Extend resource limits and measurements with:

- peak VRAM;
- media decode workers;
- expected decoded bytes per batch;
- audio/video duration;
- image/video spatial size;
- cache disk budget;
- checkpoint disk budget; and
- dataloader prefetch depth.

Feasibility filtering occurs before an experiment consumes budget. Estimated
runtime remains advisory unless a hard task constraint explicitly limits it.

## 13. Search, Promotion, and Diversity Across Modalities

The existing statistical policies remain shared, with the following rules:

- compare primary scores only under compatible metric protocols;
- use fold-, seed-, or bootstrap-based uncertainty appropriate to the task;
- never promote a low-fidelity run solely because resolution or clip sampling
  made its metric incomparable;
- charge observed decode/training cost to search-efficiency reporting;
- use normalized prediction similarity, not model names, for diversity;
- calculate similarity only for compatible aligned outputs;
- treat multimodal ablations as explicit information-gain experiments; and
- keep modality selection out of the scheduler when a task has only one
  declared modality.

For multimodal tasks, information-gain candidates include:

- each single-modality baseline;
- late fusion of the strongest components;
- missing-modality robustness;
- frozen versus fine-tuned encoders; and
- a small fusion head versus a constrained blend.

These experiments answer architectural questions and should be prioritized
only while their expected information gain justifies the cost.

## 14. Phased Implementation

### Phase 0: Contract extraction and tabular parity

Deliverables:

- `TaskSpec`, `DatasetBundle`, `SplitPlan`, `FidelityProfile`,
  `PredictionBundle`, `ResultRecord`, and `ModelBundle`;
- `ModalityRegistry` and base protocols;
- `TabularAdapter` containing existing discovery/profile behavior;
- registry-driven evaluation facade;
- legacy task-config translator;
- compatibility CSV exports; and
- contract and tabular regression tests.

Exit criteria:

- all existing tabular tasks still run without config changes;
- existing result/submission artifacts remain available;
- the same deterministic task produces equivalent folds and scores within
  numerical tolerance;
- the scheduler contains no modality branches; and
- no measurable tabular search-efficiency regression is introduced.

### Phase 1: Image classification

Deliverables:

- manifest and directory image discovery;
- corrupt-file and shape/channel profiling;
- lazy dataset and deterministic transforms;
- classification/regression evaluation;
- screen/medium/full image fidelity profiles;
- one deterministic baseline template;
- image-compatible memory cards and verifier fixtures;
- image leakage checks; and
- synthetic plus small real-fixture end-to-end tests.

Exit criteria:

- a complete image task can initialize, search, fine-tune, promote, prune, and
  generate a deployable best model;
- aligned image model predictions can be stacked by the manager;
- CPU fallback and GPU execution both report resource usage correctly; and
- no task-owned input is modified.

### Phase 2: Audio classification and regression

Deliverables:

- waveform manifest/index adapter;
- duration/channel/sample-rate profiling;
- deterministic resampling, crop/pad, and masks;
- speaker/session-aware split enforcement;
- audio fidelity profiles;
- one deterministic baseline template;
- audio verifier fixtures and leakage checks; and
- decode/cache resource measurements.

Exit criteria:

- variable-duration audio runs without eager corpus loading;
- speaker/session leakage tests fail unsafe splits;
- stacking and weighted blending work on aligned audio predictions; and
- corrupt or unsupported audio fails with a contract diagnostic.

### Phase 3: Video classification

Deliverables:

- video metadata index;
- deterministic clip sampler;
- bounded parallel decoding;
- video fidelity profiles;
- source-video/entity split enforcement;
- cache and disk-budget controls;
- one deterministic baseline template; and
- video smoke tests using tiny generated clips.

Exit criteria:

- screen fidelity completes without decoding the full corpus;
- repeated evaluation selects identical validation clips;
- no source video crosses folds; and
- resource feasibility can reject an oversized run before training.

### Phase 4: Multimodal late fusion

Deliverables:

- compositional `MultimodalAdapter`;
- entity-level join and coverage report;
- missing-modality masks;
- single-modality ablation nodes;
- aligned OOF prediction/embedding bundles;
- manager-owned stacking/blending merge nodes;
- deployable `EnsembleBundle`; and
- multimodal leakage and missing-input tests.

Exit criteria:

- every component for one entity remains in the same fold;
- the system can train, compare, and combine component models;
- the ensemble can run inference with documented missing-modality behavior;
- the provenance DAG records every contributing node; and
- late fusion beats or matches the strongest component on at least one
  representative benchmark without violating compute limits.

### Phase 5: Intermediate fusion

Deliverables:

- encoder output/embedding contract;
- reusable frozen encoders;
- learned fusion heads;
- modality dropout and calibrated gating;
- tuning-history keys for fusion structure; and
- equal-budget comparison against late fusion.

Exit criteria:

- intermediate fusion has a reproducible advantage on held-out multimodal
  benchmarks;
- the gain survives uncertainty-aware promotion;
- inference packaging includes every encoder and fusion component; and
- complexity is removed if it does not improve search efficiency or final
  quality at equal cost.

### Phase 6: Structured and generative outputs

Add one task slice at a time:

- image segmentation;
- object detection;
- temporal localization;
- retrieval;
- captioning or other generated outputs.

Each slice requires its own evaluator, prediction schema, ensemble strategy,
and acceptance benchmark. Merely decoding the modality does not constitute
support for a new output type.

## 15. File-by-File Migration Map

| Existing file | Migration |
|---|---|
| `runtime_utils.py` | Keep process safety; replace task suffix/type assumptions with TaskSpec and adapter calls |
| `agents/data_analyzer.py` | Move current logic to `TabularAdapter`; retain wrapper during migration |
| `agents/initial_agent.py` | Replace tabular prompts and generated loader ownership with RunContext/DatasetBundle |
| `agents/technique_agent.py` | Filter/query by modality, problem, output, resources, and empirical compatibility |
| `agents/implementation_agent.py` | Generate model/training interface; harness owns data, folds, metrics, and artifacts |
| `agents/validation_guard.py` | Add registry-driven modality checks and dynamic split validation |
| `evaluation_contract.py` | Become a compatibility facade over the new evaluation runner |
| `agents/aggregator_agent.py` | Produce output-specific EnsembleBundles and cross-validated stackers |
| `agents/manager_agent.py` | Load TaskSpec, enforce compatibility, and own merge-node creation |
| `memory_pool/query_tool.py` | Add modality/problem/output compatibility filters |
| `memory_pool/l1_index.json` | Add modality metadata without deleting legacy categories |
| `memory_pool/builder/l2_builder.py` | Require explicit contracts for all new artifacts |
| `memory_pool/builder/verification_runtime.py` | Extend existing modality fixtures and output validators |
| `tree/scheduler.py` | No modality logic; accept only measured normalized evidence |
| `search/*` | Keep modality-neutral; add compatibility key only where comparisons require it |
| `requirements.txt` | Retain compatibility and source pinned modality profiles through SetupAgent |

## 16. Testing Strategy

### 16.1 Contract tests

For every adapter:

- valid and invalid task manifests;
- stable dataset fingerprints;
- stable sample/entity IDs;
- corrupt and unsupported files;
- lazy loading;
- train/evaluation transform separation;
- deterministic validation batches; and
- resource-estimate bounds.

### 16.2 Evaluation oracle tests

Use small known predictions to verify:

- metric values and directions;
- class/label ordering;
- prediction alignment;
- group/entity split isolation;
- fidelity promotion compatibility;
- uncertainty summaries; and
- malformed prediction rejection.

### 16.3 Ensemble tests

Test:

- compatible and incompatible model bundles;
- cross-fitted stacker training on OOF only;
- convex fallback for insufficient data;
- inference reproduction after reload;
- no raw source-code fusion;
- provenance edges for every component; and
- no ensemble promotion without statistically supported improvement.

### 16.4 End-to-end fixtures

Maintain tiny, locally generated fixtures for:

- binary tabular classification;
- image classification;
- variable-length audio classification;
- short video classification;
- image plus tabular multimodal classification; and
- a missing-modality case.

Every fixture should finish at screen fidelity in CI without a GPU. GPU-specific
tests may be separately marked.

### 16.5 Regression and performance tests

Track:

- tabular score/fold parity;
- task initialization time;
- peak memory during discovery;
- media decoding throughput;
- cache hit rate;
- experiments to best result;
- best score at a fixed experiment budget;
- area under the best-score-versus-experiment curve;
- ensemble improvement over the best member; and
- failure rate by modality.

## 17. Rollout Controls and Acceptance Metrics

Add feature flags in task configuration:

```json
{
  "enabled_modalities": ["tabular", "image"],
  "experimental_features": {
    "multimodal_late_fusion": false,
    "intermediate_fusion": false,
    "structured_outputs": false
  }
}
```

Advance a modality from experimental to supported only when:

- its end-to-end smoke suite is stable;
- it has at least one verified baseline and one verified reusable artifact;
- split leakage checks cover its identity/group unit;
- prediction and model bundles reload for inference;
- manager-owned ensemble merging is tested;
- resource measurements are credible; and
- fixed-budget benchmarks show no regression in the shared scheduler.

Search-efficiency changes should be evaluated across several tasks and seeds
using:

- best normalized score at a fixed implementation budget;
- experiments required to reach a quality threshold;
- area under the best-score curve;
- wall-clock and accelerator-hours;
- invalid/failed experiment rate; and
- promotion precision: promoted nodes that improve at higher fidelity.

Any new scheduler mechanism that does not improve these measures should be
removed, even if it improves one isolated run.

## 18. Recommended Implementation Backlog

Execute in this dependency order:

1. Add `TaskSpec` and a legacy tabular config translator.
2. Add `ModalityAdapter` and `ModalityRegistry`.
3. Move current discovery/profile behavior into `TabularAdapter`.
4. Add dataset, split, fidelity, result, prediction, and model bundle schemas.
5. Turn `evaluation_contract.py` into a compatibility facade.
6. Make ImplementationAgent use harness-owned data and evaluation.
7. Add compatibility keys to comparison, promotion, and ensemble inputs.
8. Refactor AggregatorAgent into a strategy registry that writes
   `EnsembleBundle`.
9. Make ManagerAgent own merge-node construction and provenance edges.
10. Add memory-pool compatibility filters and require explicit contracts for
    new artifacts.
11. Implement the image adapter and image-classification baseline.
12. Add image end-to-end, leakage, resource, and ensemble tests.
13. Implement audio using the proven image-era contracts.
14. Implement video with bounded decode and cache policies.
15. Implement text as a composable input primitive.
16. Implement entity-aligned multimodal late fusion.
17. Benchmark late fusion before adding intermediate or end-to-end fusion.
18. Add structured output types individually behind acceptance gates.

## 19. Non-Goals

The initial expansion should not:

- support every problem type merely because its files can be decoded;
- create separate search schedulers for each modality;
- let generated code choose or modify validation splits;
- eagerly load or copy entire media datasets for each node;
- install every possible media/model dependency by default;
- merge node source code;
- fine-tune large end-to-end multimodal models before late fusion is measured;
- treat an averaged submission as a complete deployable ensemble; or
- accept architecture complexity without an equal-budget benchmark gain.

## 20. Definition of Done

The multimodal expansion is complete when one unchanged orchestration path can:

1. parse and validate tabular, image, audio, video, and multimodal tasks;
2. build stable sample/entity indices without eager media loading;
3. generate and execute modality-compatible baseline and candidate models;
4. enforce deterministic, leakage-safe evaluation;
5. emit normalized prediction, result, and model bundles;
6. apply the same statistical pruning, promotion, diversity, and
   information-gain policies;
7. reuse compatible memory-pool artifacts and tuning history;
8. let ManagerAgent create a deployable ensemble from compatible nodes;
9. preserve the execution tree while recording all contributors in the
   provenance DAG;
10. reproduce inference from the selected model or ensemble bundle; and
11. demonstrate equal-or-better search efficiency on a held-out benchmark
    suite without regressing existing tabular tasks.
