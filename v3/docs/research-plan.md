# MIDI Velocity Infer v3 Research Plan

## Goal

Build a more general and scalable MIDI velocity inference model than v2 by moving from a tiny seq2seq RNN with local attention to a note-level encoder architecture with stronger context modeling, better note representation, and optional self-supervised pretraining.

Scope: v3 is intentionally specialized for solo piano velocity inference. Multi-instrument or ensemble scenarios are out of scope.

Target venue: ISMIR. The experimental design should prioritize musical interpretation, rigorous ablation, and reproducibility over simple SOTA claims.

## Motivation

The current v2 codebase is useful as a proof of concept, but it has several structural limits.

- It uses a very short fixed note window.
- It depends on a small handcrafted feature set.
- It treats velocity prediction as a seq2seq problem even though input and output notes are aligned one-to-one.
- It cannot model long-range phrasing or repeated musical structure well.
- Its current implementation does not meaningfully use teacher forcing despite exposing the flag in config.

Recent related work such as `midi-velocity-colorizer` suggests that larger-context architectures and more expressive training objectives are promising directions. More broadly, Transformer-family models are better suited than small RNN seq2seq models for learning long-range dependencies in symbolic music.

## Decisions Locked So Far

This section records the decisions that are already fixed for the first v3 milestone. Each item explicitly records the options that were considered and the selected default.

### First Milestone

- Options considered:
  - strong supervised baseline
  - SSL-first design
  - comparison framework first
- Selected:
  - strong supervised baseline

Reasoning:
The first v3 milestone should establish a clean encoder-only baseline before introducing pretraining or large comparison matrices.

### Base Feature Scope

- Options considered:
  - relative-timing-only base representation
  - relative timing plus optional structural hooks
  - structural features as core inputs from day one
- Selected:
  - relative timing plus optional structural hooks

Reasoning:
The base representation should work without beat/bar metadata, while still allowing later conditioning when structural annotations are available.

### Window Unit

- Options considered:
  - note-count windows
  - time-based windows
  - support both from the start
- Selected:
  - note-count windows

Reasoning:
Note-count windows are simpler for the first symbolic sequence baseline and make padding and batching easier to control.

### Velocity Target Type

- Options considered:
  - pure regression
  - classification plus residual regression
  - multi-head comparison from the start
- Selected:
  - pure regression

Reasoning:
Pure regression is the simplest and cleanest baseline for v3. Hybrid targets remain follow-up work.

### Note Embedding Scope

- Options considered:
  - minimal feature set
  - balanced feature set
  - expanded feature set with inferred higher-level roles
- Selected:
  - balanced feature set

Reasoning:
The first v3 embedding should go beyond pitch and timing only, but should avoid unstable inferred labels such as hand, voice, or chord-tone roles until their definitions are validated.

### Initial Dataset Scope

- Options considered:
  - MAESTRO only
  - MAESTRO plus GiantMIDI-Piano
  - broad symbolic MIDI collection from the start
- Selected:
  - MAESTRO only

Reasoning:
MAESTRO should anchor the first supervised baseline and evaluation protocol before broader dataset expansion.

### Position Encoding

- Options considered:
  - relative positional bias
  - relative time-aware attention bias
  - mixed relative position and relative time bias
- Selected:
  - relative positional bias

Reasoning:
The first model should use a standard and stable Transformer mechanism. Timing information will enter through note features first.

### Window Overlap

- Options considered:
  - fixed overlap
  - non-overlapping windows
  - variable overlap policy
- Selected:
  - fixed overlap

Reasoning:
Some overlap should reduce boundary artifacts without adding too much complexity. The exact stride is still open.

### Initial Evaluation Scope

- Options considered:
  - objective metrics only
  - objective metrics plus rendering comparison
  - subjective listening evaluation from day one
- Selected:
  - objective metrics only

Reasoning:
The first v3 milestone should stabilize the benchmark and model behavior before adding subjective evaluation.

### Baseline Model Scale

- Options considered:
  - small baseline
  - medium baseline
  - large baseline
- Selected:
  - medium baseline

Reasoning:
The first supervised baseline should be large enough to be meaningful, but not so large that it slows down iteration or hides data-pipeline issues.

### Velocity Normalization Scheme

- Options considered:
  - raw 0 to 127 regression
  - global 0 to 1 normalization
  - dataset-level standardization
- Selected:
  - global 0 to 1 normalization

Reasoning:
This keeps optimization stable while preserving straightforward conversion back to raw MIDI velocity for evaluation.

### First Window and Stride

- Options considered:
  - 128-note window with stride 64
  - 256-note window with stride 128
  - 512-note window with stride 256
- Selected:
  - 256-note window with stride 128

Reasoning:
This is the current default tradeoff between context length, overlap, and training cost.

### Derived Feature Policy

- Options considered:
  - timestamp-only derived features
  - local harmony rule-based features
  - expanded inferred features such as hand or voice
- Selected:
  - timestamp-only derived features

Reasoning:
The first baseline should use only features that can be computed reliably from note events without higher-level inference.

### v2 Comparison Level

- Options considered:
  - reuse split and key metrics
  - reproduce v2 post-processing as closely as possible
  - treat v2 as a rough historical reference only
- Selected:
  - reuse split and key metrics

Reasoning:
The comparison should remain fair and useful without forcing v3 to inherit all of v2's post-processing decisions.

### Default Training Loss

- Options considered:
  - MSE
  - MAE
  - Huber loss
  - loss ablation in Phase 1
- Selected:
  - Huber loss

Reasoning:
Huber loss combines the outlier robustness of MAE with the smooth convergence of MSE near the target. It is a better default than pure MSE for velocity prediction, where the target distribution is concentrated in the mid-range and mean regression is a concern. The delta hyperparameter defaults to `1.0`.

### Pitch Encoding Strategy

- Options considered:
  - 128 categorical pitch embedding
  - octave + chroma decomposition
  - continuous pitch representation
- Selected:
  - ablation item for Phase 2

Reasoning:
Octave(0-10) + chroma(0-11) decomposition offers better data efficiency (23 vs 128 embeddings) and natural encoding of harmonic equivalence across octaves. However, 128 categorical embedding is simpler and well-established. Both will be compared in Phase 2. The first baseline may use either as default.

### Canonical Event Format

- Options considered:
  - introduce a new canonical note-event format
  - extend the current CSV format
  - support both as equal first-class formats
- Selected:
  - introduce a new canonical note-event format

Reasoning:
v3 needs a richer internal representation than the current CSV was designed to support.

## Research Questions

1. Can note-level velocity be predicted more accurately with an encoder-only architecture than with the current seq2seq attention model?
2. Can a general-purpose `Note Embedding` outperform simple pitch-only or raw feature-only representations?
3. Does self-supervised pretraining on symbolic MIDI improve downstream velocity prediction?
4. How much musical structure can be captured without explicit beat/bar annotations if relative timing features are used?
5. When beat/bar features are available, do they provide meaningful gains on top of relative timing features?

## Design Principles

- Prefer note-aligned prediction over autoregressive decoding.
- Keep the base representation usable even when beat/bar metadata is unavailable.
- Use relative timing features such as IOI and duration as first-class inputs.
- Treat beat/bar features as optional enhancements, not hard requirements.
- Design the model so that pretraining and fine-tuning share the same backbone.

## Proposed v3 Task Formulation

The v3 model should treat each note as one structured event and predict one velocity value per note.

- Input: a sequence of note events
- Output: one velocity value per input note
- Core task type: sequence labeling / per-note regression

This is a better fit than encoder-decoder seq2seq because the target sequence is aligned with the input sequence.

### Canonical Note-Event Format

The first v3 baseline should use a canonical note-event representation as its primary internal data format.

#### Core Event Fields

Each note event should contain at least the following fields.

- `piece_id`: stable identifier for the source performance or file
- `note_index`: stable zero-based note order within `piece_id`
- `pitch`: raw MIDI pitch in `[0, 127]`
- `onset_sec`: note onset time in seconds
- `offset_sec`: note offset time in seconds
- `velocity`: raw MIDI velocity in `[0, 127]` for supervised data, nullable for inference-only or SSL data

#### Ordering Rule

Events within each `piece_id` should be sorted by:

1. `onset_sec`
2. `pitch`
3. `note_index`

This ordering should be treated as the canonical sequence order for windowing and Transformer input construction.

Note: within simultaneous-onset chords, pitch-ascending order implies bass-first sequencing. This is a deliberate choice for determinism. Since the encoder is bidirectional, ordering bias is less impactful than in unidirectional models. Chord-internal ordering alternatives (e.g., soprano-first) are a candidate for Phase 2 ablation.

#### Derived Fields for the First Supervised Baseline

The following fields do not need to be stored in the raw canonical record, but they must be generated deterministically from the canonical note events during preprocessing.

- `duration_sec = offset_sec - onset_sec`
- `ioi_next_sec = onset_sec(next) - onset_sec(current)`
- `delta_pitch_prev = pitch(current) - pitch(previous)`
- `delta_pitch_next = pitch(next) - pitch(current)`
- `same_onset_chord_size`: number of notes whose onset falls within `+-30 ms` of the current note onset
- `local_note_density`: local onset density computed from the symmetric `+-8 note` neighborhood around the current note
- `register_bucket`: 4 coarse bins using MIDI pitch boundaries `< 48`, `48 <= pitch < 60`, `60 <= pitch < 72`, `>= 72`

The `+-30 ms` chord onset tolerance is an initial default. This threshold should be validated empirically by analyzing the onset-difference distribution in MAESTRO. The `chord_onset_tolerance_ms` value should be exposed as a configurable preprocessing parameter.

#### Optional Structural Fields

These fields are not required for the first baseline, but the canonical format may be extended to carry them later.

- `beat_position`
- `bar_position`
- `metrical_strength`
- `tempo_bucket`

#### Compatibility With Existing CSV

The current `midi2csv` output should be treated as an import/export compatibility format, not the primary training representation.

The following legacy values should remain recoverable from the canonical note-event sequence when needed for comparison or export.

- `time`
- `time_diff`
- `note_num`
- `note_num_diff`
- `low_octave`
- `length`
- `velocity`

#### Window Construction Rule

Windowing should be performed only after canonical note-event ordering is established.

- windows must not cross `piece_id` boundaries
- the first baseline uses `256-note` windows with `stride 128`
- labels stay aligned one-to-one with input note events inside each window

## Note Representation

### Base Note Features

The first version of v3 should avoid hard dependence on meter annotations and instead use a note representation that is broadly applicable across MIDI datasets.

Planned base features for the balanced embedding:

- `pitch`
- `duration`
- `IOI` or note-to-next-note timing gap
- `delta_pitch_prev`
- `delta_pitch_next`
- `local_note_density`
- `same_onset_chord_size`
- `register_bucket`

Definitions for the first supervised baseline:

- `same_onset_chord_size`: number of notes whose onset falls within `+-30 ms` of the current note onset
- `local_note_density`: local onset density computed from the symmetric `+-8 note` neighborhood around the current note
- `register_bucket`: 4 coarse bins using MIDI pitch boundaries `< 48`, `48 <= pitch < 60`, `60 <= pitch < 72`, `>= 72`

Candidate additional feature for Phase 2 ablation:

- `local_note_density_temporal`: notes per second computed from a symmetric `+-0.5 s` time window around the current note onset. This captures temporal density independently of note-count neighborhoods and may provide complementary information to the note-count-based density, especially across varying tempi.

Deferred for later validation:

- `is_chord_tone`
- inferred `voice`
- inferred `hand`

### Optional Structural Features

When beat or bar information is available or can be estimated reliably, it can be added as optional conditioning.

- `beat_position`
- `bar_position`
- `metrical_strength`
- `tempo_bucket`

These are optional additions, not requirements for the first supervised baseline.

### Embedding Strategy

Instead of only using pitch embedding, v3 should use a hybrid `Note Embedding`.

- Categorical features: embedding tables
- Continuous features: linear projection or small MLP
- Final note embedding: sum or concatenation followed by projection

Example (with 128 categorical pitch embedding):

```text
E_note =
  E_pitch
  + E_register
  + P(duration, IOI, delta_pitch_prev, delta_pitch_next, density, chord_size)
  + E_optional_structure
  + E_position
```

Alternative (with octave + chroma decomposition):

```text
E_note =
  E_octave + E_chroma
  + E_register
  + P(duration, IOI, delta_pitch_prev, delta_pitch_next, density, chord_size)
  + E_optional_structure
  + E_position
```

The octave+chroma variant uses two small embedding tables (12 chroma + 11 octave) instead of one 128-entry table, providing natural harmonic equivalence across octaves and better data efficiency for rare pitch values. Both variants will be compared in Phase 2 ablation.

This allows the model to remain general while still learning musically meaningful note-level structure.

## Proposed Backbone

### Primary Direction

Use an encoder-only Transformer as the main v3 baseline.

Baseline direction:

- encoder-only Transformer
- relative positional bias
- note-count windows
- per-note regression head
- 4 encoder layers
- `d_model = 256`
- `n_heads = 8`
- `ffn_dim = 1024`
- `dropout = 0.1`
- `Pre-Norm` blocks
- `T5-style` relative position bucket bias

Later ablation candidates:

- 6 layers with the same width
- 4 layers with a wider hidden size
- alternative relative bias designs
- `Pre-Norm` vs `Post-Norm` block ordering (Post-Norm may yield better representation quality in shallow models with proper warmup)
- onset-time-aware relative attention bias as a supplement to T5-style position bucket bias (to address the mismatch between position distance and temporal distance for simultaneous-onset chords)

### Why Encoder-Only

- The problem is aligned note-to-note.
- Bidirectional context is useful for velocity prediction.
- It supports masked-note pretraining naturally.
- It is simpler than a decoder-based rendering pipeline.

### Alternative Backbones to Compare Later

- BiLSTM encoder baseline
- local-window Transformer variants
- hybrid CNN/Transformer
- U-Net-style piano-roll model inspired by `midi-velocity-colorizer`

The encoder-only Transformer remains the main symbolic sequence baseline for the first milestone.

## Self-Supervised Pretraining

### Status

Self-supervised learning remains part of the v3 research direction, but it is not part of the first milestone.

### Hypothesis

Self-supervised learning is likely practical and useful if the new architecture is large enough to benefit from representation learning and if enough symbolic MIDI data can be collected.

### Recommended Pretraining Objective

The first pretraining objective should be masked note modeling.

- randomly mask selected note attributes
- predict masked attributes from surrounding context
- the first SSL version should experiment with both including and excluding velocity as a masking target
- if velocity is included, use a low masking rate to let the backbone learn phrase-level velocity patterns without dominating the structural objective

Candidate maskable attributes:

- `pitch`
- `duration`
- `IOI`
- `delta_pitch_prev`
- `delta_pitch_next`

### Secondary SSL Objectives

After the first version is stable, compare:

- denoising reconstruction
- span masking
- contrastive sequence embedding

Masked note modeling remains the default first SSL direction.

## Fine-Tuning for Velocity Inference

After pretraining, fine-tune the backbone on labeled velocity datasets.

- input: note sequence with note embedding
- backbone: pretrained encoder
- head: lightweight regression module
- output: one scalar velocity per note

Selected first head:

- `Linear -> GELU -> Linear`

Deferred comparison heads:

- classification into velocity bins plus residual regression

Default training loss:

- Huber loss with `delta = 1.0`

Candidate losses to compare after the baseline is stable:

- MAE
- MSE
- Huber loss plus rank-correlation auxiliary term (motivated by the same directional-dynamics objective that drove v2's cosine similarity loss, but formulated as a proper rank correlation)
- coarse-bin cross-entropy plus residual regression loss

The current `MSE + cosine similarity` loss from v2 should be treated as a legacy baseline, not the default v3 objective. However, its motivation — learning relative dynamics patterns, not just absolute velocity values — remains valid and is addressed by the rank-correlation auxiliary term planned for Phase 1 follow-up.

## Data Strategy

### Initial Labeled Data

- MAESTRO should remain the first main benchmark.

### Possible Additional Data

These are not part of the first milestone and require additional investigation.

- GiantMIDI-Piano
- other aligned symbolic piano datasets if licensing and preprocessing are manageable
- broader symbolic MIDI corpora for SSL pretraining

### Data Augmentation

Piano velocity inference places strong constraints on which augmentations are safe. Pitch transposition changes the register-velocity relationship, and time stretching distorts IOI distributions. The following augmentations are adopted as safe for this task:

- `velocity jittering`: add small uniform noise (e.g., `+-2` in raw velocity scale) to target velocities during training to reduce overfitting to exact values
- `window offset jitter`: vary the window start offset by a small random amount (e.g., `+-16 notes`) each epoch so that the model does not memorize fixed window boundaries

These augmentations should be combined with standard regularization: `dropout = 0.1`, `weight_decay = 0.01`, and early stopping.

### Data Windowing

Avoid the extremely short fixed window used in v2.

Selected direction:

- 256-note windows
- stride 128
- fixed overlap

Time-based segmentation such as 5 to 10 second chunks is deferred until after the first note-window baseline.

### v3 Preprocessing and Data Pipeline Contract

The first v3 baseline should use a deterministic preprocessing pipeline that converts source note data into canonical note events, derived features, fixed windows, and model-ready tensors.

#### Stage 1: Source Ingestion

Accepted source types for the first implementation:

- MAESTRO-derived note data
- existing CSV exports from the current `midi2csv` toolchain

Required ingestion outcome:

- one note-event table per source piece
- one stable `piece_id` per source piece

#### Stage 2: Canonical Note-Event Construction

Each ingested piece must be converted into the canonical note-event format described above.

Required operations:

- map source note data into `piece_id`, `note_index`, `pitch`, `onset_sec`, `offset_sec`, `velocity`
- discard malformed notes where `offset_sec <= onset_sec`
- sort events by canonical ordering
- reassign `note_index` after sorting if needed

Output of this stage:

- one ordered canonical note-event sequence per `piece_id`

#### Stage 3: Derived Feature Generation

Derived features must be generated from canonical note events only.

Required generated values:

- `duration_sec`
- `ioi_next_sec`
- `delta_pitch_prev`
- `delta_pitch_next`
- `same_onset_chord_size`
- `local_note_density`
- `register_bucket`

Boundary behavior for the first and last notes:

- missing previous-note values use a deterministic default of `0`
- missing next-note values use a deterministic default of `0`

#### Stage 4: Split-Aware Statistics and Normalization

Normalization statistics must be fit on the training split only, then reused for validation, test, and inference.

Required rules:

- fit training-only statistics for continuous input features that require scaling
- keep categorical features such as `register_bucket` unnormalized
- normalize velocity targets globally from raw `[0, 127]` to `[0, 1]`
- preserve the raw target values for evaluation and export paths

Continuous features expected to be scaled in the first baseline:

- `pitch` if treated as continuous in the final preprocessing path
- `duration_sec`
- `ioi_next_sec`
- `delta_pitch_prev`
- `delta_pitch_next`
- `local_note_density`
- `same_onset_chord_size` if represented as continuous rather than bucketed

#### Stage 5: Window Construction

Windows must be built after normalization metadata is fixed and after canonical ordering is established.

Required rules:

- build windows independently within each `piece_id`
- do not mix events from different pieces in the same window
- use `window_size = 256`
- use `stride = 128`
- keep one target velocity per input note
- allow overlapped windows to contain duplicated note supervision during training

Partial trailing windows:

- the first baseline should pad trailing windows rather than drop them
- padding must use a deterministic maskable representation so padded notes can be excluded from loss and metrics

#### Stage 6: Tensorization

Each window should be transformed into model-ready tensors with explicit alignment between inputs, targets, and masks.

Required outputs per window:

- input feature tensor of shape `[seq_len, feature_dim]` or an equivalent structured representation before embedding
- target velocity tensor of shape `[seq_len]` or `[seq_len, 1]`
- padding mask of shape `[seq_len]`
- metadata sufficient to map predictions back to `piece_id` and note order
- metadata sufficient to resolve duplicated predictions from overlapping windows during evaluation

Required metadata per window:

- `piece_id`
- starting `note_index`
- true unpadded note count
- each note's local index within the window

#### Stage 7: Evaluation Reconstruction

The pipeline must support reconstruction from window-level predictions back to piece-level note order.

Required rules:

- reconstruct predictions in canonical note order
- ignore padded positions during aggregation
- use piece-level reconstruction as the primary evaluation unit
- when overlap creates multiple predictions for the same note, use the prediction whose note position is closest to the center of its window
- if two predictions are equally close to window center, prefer the earlier window in canonical order
- alternative: soft weighting based on center distance (e.g., triangular or Gaussian weight) to blend predictions from multiple windows. This is a candidate for ablation in Phase 2, as it may reduce boundary discontinuities compared to hard selection

The first baseline does not use window-level averaged metrics as its primary benchmark.

#### Compatibility With Existing v2 Artifacts

The preprocessing contract should remain compatible with v2 comparison assets without forcing v3 to reuse the old internal representation.

Required compatibility points:

- reuse MAESTRO split structure
- retain enough metadata to report raw-scale velocity metrics
- retain enough information to export rounded and clipped velocities through CSV or MIDI tooling when needed

## Evaluation Plan

### Initial Objective Metrics

The first v3 milestone should use objective metrics only.

- note-level MAE in raw MIDI velocity scale
- normalized MAE
- MSE
- Spearman correlation
- Pearson correlation
- predicted velocity standard deviation vs. target standard deviation (primary variance collapse diagnostic)
- calibration plots and distribution matching

#### Metric Reporting Units

Metrics should be reported at two levels:

- **Global metrics** (primary): computed over the entire test set. These are the main comparison numbers.
- **Per-piece metrics** (secondary): computed per piece, then reported as mean ± std. Per-piece correlation and MAE distributions provide insight into model consistency across pieces of varying difficulty and style.

Both levels should be reported in the paper. Per-piece distributions visualized as boxplots are recommended for ISMIR.

#### Variance Collapse Monitoring

The ratio of predicted velocity standard deviation to target velocity standard deviation should be tracked as a mandatory training diagnostic. If this ratio drops significantly below `1.0`, it indicates mean regression. In that case, the following interventions should be considered in order:

1. Switch from Huber to MAE loss
2. Add rank-correlation auxiliary loss
3. Consider distributional modeling (e.g., beta distribution head)

#### Qualitative Analysis Plan

The following qualitative analyses should be included in the paper:

- **Velocity contour visualization**: overlay predicted vs. target velocity profiles for selected pieces, showing phrase-level dynamics
- **Error analysis by musical context**: breakdown of prediction error by register (bass/mid/treble), dynamics range (pp/mp/mf/f/ff), chord vs. single note, and passage density
- **Attention pattern visualization**: selected attention heads to illustrate what musical relationships the model learns
- **Per-piece error distribution**: identify pieces with unusually high or low error and analyze common characteristics

### v2-Compatible Evaluation Contract

The first v3 baseline should remain comparable to the existing v2 pipeline without inheriting all of its implementation constraints.

#### Reused Comparison Basis

The v2 comparison should reuse the same dataset split structure as the current MAESTRO pipeline.

- train split
- validation split
- test split

The comparison should also reuse the same raw target interpretation.

- raw MIDI velocity range: `[0, 127]`
- v2-style metadata values such as `velocity_min`, `velocity_max`, and related normalization metadata remain valid comparison references

#### Reporting Rule

The primary reported scale for comparison must be raw MIDI velocity, even if training uses normalized targets.

Required reporting flow:

1. predict normalized velocity values
2. convert predictions back to raw velocity scale
3. report metrics on raw-scale predictions

#### Raw and Post-Processed Outputs

Two output forms should be distinguished during evaluation.

- raw reconstructed output:
  - normalized prediction converted back to raw velocity scale
- simple post-processed output:
  - raw reconstructed output after rounding to integer velocity
  - optional clipping to `[0, 127]` when required for MIDI export or strict comparison

The first v3 comparison should report raw-scale metrics as primary results and may additionally report rounded or clipped results as secondary outputs.

Primary comparison unit:

- piece-level reconstruction after overlap resolution

Secondary optional view:

- window-level diagnostic metrics only, never as the main benchmark

#### Required Metrics for v2-v3 Comparison

The mandatory metrics for the first baseline comparison are:

- note-level MAE on raw velocity scale
- normalized MAE
- MSE
- predicted velocity standard deviation vs. target standard deviation

Recommended additional metrics:

- Spearman correlation
- Pearson correlation

The custom `f1`-style score used in the existing attention demo may be preserved as a historical secondary metric, but it should not replace the core regression metrics above.

#### What Is Not Required to Match Exactly

The v3 baseline does not need to inherit all v2 evaluation-side behavior.

- random augmentation of predicted velocity for demo-style output is not part of the comparison contract
- exact notebook-specific plotting is not part of the comparison contract
- exact demo export behavior is not required unless generating MIDI outputs for qualitative review

#### Export Compatibility Rule

If v3 outputs are exported back through the existing CSV or MIDI toolchain, the exported velocity values should follow this order:

1. convert to raw velocity scale
2. round to integer
3. clip to `[0, 127]` if needed by the target format

This preserves compatibility with the existing `csv2midi` expectation while keeping evaluation and export concerns separate.

### Deferred Evaluation

These are valuable but not required for the first milestone.

- rendered MIDI comparison
- pairwise preference tests
- MOS-style listening evaluation

### Required Comparisons

- v2 attention baseline
- v3 Transformer supervised baseline
- v3 Transformer with balanced note embedding ablations
- cross-ablation: v3 architecture with v2 features, v2 architecture with v3 features, v2 architecture with v3 window size
- later: v3 pretrained Transformer fine-tuned on velocity
- later: optional U-Net or piano-roll model inspired by `midi-velocity-colorizer` (treated as related work in the first paper, experimental comparison deferred to Phase 4)

## Experimental Roadmap

### Phase 1: Strong Supervised Baseline

Build a clean encoder-only supervised baseline without SSL.

- implement note embedding
- implement Transformer encoder
- train on MAESTRO
- compare against v2

Success criterion (multi-dimensional):

- clear gain over v2 in MAE and correlation metrics
- improved velocity variance matching (predicted std / target std ratio closer to `1.0`)
- both conditions must be met simultaneously — MAE improvement with variance collapse is not considered a success

### Phase 2: Feature and Architecture Ablations

Test which note representation and architecture choices matter most.

Feature ablations:

- pitch only
- pitch plus relative timing
- balanced note embedding
- balanced note embedding plus optional beat/bar features

Architecture and encoding ablations:

- 128 categorical pitch embedding vs. octave + chroma decomposition
- Pre-Norm vs. Post-Norm Transformer blocks
- T5 position bias only vs. T5 position bias + onset-time-aware bias
- center-preference overlap resolution vs. soft-weighted blending
- chord-internal ordering: pitch-ascending (bass-first) vs. alternatives
- `local_note_density` note-count-based vs. time-based vs. both

Cross-ablation (for disentangling contribution sources):

- v3 Transformer with v2 feature set (isolates architecture contribution)
- v2 architecture with v3 feature set (isolates feature contribution)
- v2 architecture with v3 window size (isolates context length contribution)

Success criterion:

- identify a minimal robust feature set
- clearly attribute performance gains to architecture, features, and context length

### Phase 3: Self-Supervised Pretraining

Pretrain the same encoder backbone, then fine-tune for velocity inference.

- pretrain with masked note modeling
- fine-tune on MAESTRO
- compare with training from scratch

Success criterion:

- improved sample efficiency
- improved generalization or stability

### Phase 4: Architecture Expansion and Conditioning

Compare symbolic sequence modeling against structured alternatives, and explore conditioning mechanisms.

Architecture comparisons:

- local/global Transformer variants
- hybrid CNN/Transformer
- U-Net-style piano-roll formulation (comparison with `midi-velocity-colorizer`, treated as related work in Phase 1 paper and as experimental comparison in Phase 4)

Conditioning experiments:

- performer embedding or style token conditioning using MAESTRO performer metadata
- this may help address the "average style" convergence problem inherent in performer-agnostic models

Success criterion:

- determine whether image-like velocity filling is better than note-sequence modeling for this task
- evaluate whether performer conditioning improves prediction quality or stylistic fidelity

## Training Configuration

The following training hyperparameters are fixed for the first supervised baseline.

- Optimizer: `AdamW` with `weight_decay = 0.01`
- Learning rate schedule: linear warmup (`5%` of total steps) followed by cosine decay
- Peak learning rate: `3e-4`
- Batch size: `32` windows (adjustable based on GPU memory)
- Maximum epochs: `100` with early stopping (`patience = 10`, monitored on validation MAE)
- Gradient clipping: `max_norm = 1.0`
- Loss function: Huber loss with `delta = 1.0`
- Dropout: `0.1`

These settings are standard for medium-scale Transformers and should provide a stable starting point. All hyperparameters should be documented in the paper for reproducibility.

## Open Questions

There are no remaining architecture or feature-definition open questions for the first supervised v3 baseline.

The following baseline choices are now fixed:

- medium-scale encoder-only Transformer
- 4 layers, `d_model = 256`, `n_heads = 8`, `ffn_dim = 1024`, `dropout = 0.1`
- `Pre-Norm` Transformer blocks (with Post-Norm as Phase 2 ablation)
- `T5-style` relative position bucket bias (with time-aware bias as Phase 2 ablation)
- global `0..1` velocity normalization with raw-scale reporting
- `256-note` window with `stride 128`
- Huber loss with `delta = 1.0`
- timestamp-based derived features only
- `same_onset_chord_size` with `+-30 ms` onset tolerance (to be validated empirically)
- `local_note_density` from a symmetric `+-8 note` neighborhood
- 4-bin `register_bucket` with boundaries `48/60/72`
- pitch encoding: both 128 categorical and octave+chroma will be tested in Phase 2
- data augmentation: velocity jittering and window offset jitter
- training: AdamW, warmup + cosine decay, early stopping

Any further uncertainty for v3 now falls under `Required Research Before Expansion`, not baseline-defining open questions.

## Required Research Before Expansion

These items require additional investigation before they should influence the roadmap beyond the first milestone. They are organized by priority so that implementation-blocking research is handled first.

### Research Result Format

Each research item should produce the same four outputs:

- conclusion: `adopt`, `defer`, or `exclude`
- evidence: code, dataset, or paper facts that support the conclusion
- impact: what changes in the v3 baseline or roadmap
- follow-up action: document update, implementation task, or later experiment

### P0: Research That Directly Blocks a Clean v3 Baseline

#### Canonical Note-Event Format vs. Existing CSV Pipeline

Why it matters:

- v3 now assumes a new canonical note-event format, but v2 comparisons still depend on the existing CSV-based toolchain

Questions to confirm:

- which minimal fields must exist in the canonical note-event format
- which legacy CSV fields must remain available for comparison or export
- whether all current planned derived features can be reconstructed from canonical note events without relying on the old CSV layout

Expected output:

- validation that the documented canonical note-event schema is sufficient
- a list of legacy compatibility fields
- a conclusion on whether CSV should be treated as an import/export format only

#### v2-Compatible Evaluation Contract

Why it matters:

- the comparison level is fixed as "reuse split and key metrics", but the exact contract still needs to be documented

Questions to confirm:

- which split files and metadata artifacts from v2 are reused directly
- which metrics are mandatory for v2-v3 comparison
- whether raw predictions, clipped predictions, or both should be reported

Expected output:

- validation that the documented v2-compatible evaluation contract is sufficient
- a metric list with raw-scale reporting rules
- a conclusion on which v2 post-processing steps remain relevant

### P1: Research Needed to Lock the First Feature Set

#### Timestamp-Based Derived Feature Definitions

Why it matters:

- the first baseline now limits itself to timestamp-based derived features, so those definitions must be made reproducible

Questions to confirm:

- how `local_note_density` is defined
- how `same_onset_chord_size` is defined
- how `register_bucket` is discretized

Expected output:

- exact mathematical or algorithmic definitions for each feature
- allowed value ranges and edge-case handling
- a conclusion on whether all selected features remain in the first baseline

#### Velocity Normalization and Reporting Stability

Why it matters:

- the selected default is global 0 to 1 normalization, but this should be checked against MAESTRO's actual target distribution and reporting needs

Questions to confirm:

- whether global normalization is sufficient for stable training
- whether raw-scale reporting remains faithful after conversion
- whether there is any practical reason to keep a raw-target or standardized-target fallback

Expected output:

- a final normalization rule for training
- a reporting rule for converting predictions back to raw velocity
- a conclusion on whether alternative target scalings can be dropped from v1

### P2: Research for Post-Baseline Feature and Dataset Expansion

#### Beat/Bar Features as Optional Inputs

Why it matters:

- the document keeps beat/bar optional, but their extraction reliability and actual benefit are still unclear

Questions to confirm:

- whether beat/bar positions can be extracted robustly from the target datasets
- whether extraction is dataset-specific or general enough for the v3 design goals
- whether the expected gain is large enough to justify the added complexity

Expected output:

- a conclusion on whether beat/bar remains optional, deferred, or worth early ablation
- an extraction path if retained
- a short risk summary

#### GiantMIDI-Piano as a Supervised or SSL Resource

Why it matters:

- the roadmap mentions it as a likely expansion target, but its role is not yet clear

Questions to confirm:

- whether it is suitable for direct supervised velocity learning
- whether it is better treated as evaluation data, SSL-only data, or excluded from early work
- whether its target distribution is compatible enough with MAESTRO for meaningful comparison

Expected output:

- a conclusion of `supervised`, `SSL only`, or `defer`
- the main reasons for that conclusion
- any compatibility notes for later experiments

### P3: Research for Comparison Credibility and Longer-Term Expansion

#### Reproducibility Level for `midi-velocity-colorizer`

Why it matters:

- the current plan cites it as a strong related direction, but the level of comparison is still ambiguous

Questions to confirm:

- whether v3 needs only conceptual comparison or also a lightweight implementation baseline
- which dimensions matter most: segmentation, representation, loss, or architecture
- what minimum comparison is needed for a credible claim in a report or paper

Expected output:

- a decision on `conceptual reference`, `lightweight baseline`, or `later faithful reproduction`
- a short comparison checklist
- a statement of what will and will not be claimed

#### Additional Symbolic MIDI for SSL

Why it matters:

- SSL is deferred, but its future usefulness depends on whether realistic pretraining sources are available

Questions to confirm:

- which additional symbolic MIDI sources are legally and practically usable
- how noisy their event timing and note data are
- whether mixing them with MAESTRO-style data is likely to help or hurt downstream velocity inference

Expected output:

- a shortlist of viable SSL data sources
- a recommendation to adopt, defer, or avoid multi-source SSL for v3
- any licensing or preprocessing warnings

## Practical Assessment

### Immediately Practical

- note embedding
- encoder-only Transformer
- larger note-count windows
- supervised fine-tuning without seq2seq decoding
- masked note modeling as a later SSL step

### Deferred

- complex diffusion-based generation
- full style-control conditioning
- multi-task expressive rendering beyond velocity

These may be valuable later, but they are not necessary to establish a strong v3 baseline.

## Implementation Priorities

1. Resolve P0 research items and lock the canonical v3 note-event format plus the v2 comparison contract.
2. Resolve P1 research items and finalize the timestamp-based derived feature definitions.
3. Implement the balanced `Note Embedding` with relative timing features.
4. Build a supervised encoder-only Transformer baseline on MAESTRO.
5. Reproduce v2 comparison metrics under the locked evaluation contract.
6. Run note-embedding ablations.
7. Resolve P2 and P3 research items before expanding to optional structural features, extra datasets, or SSL.

## Expected Outcome

The expected outcome of v3 is not just a marginally better predictor, but a more general research platform for solo piano velocity inference with the following properties.

- works without mandatory beat/bar annotations
- uses a musically richer note representation
- scales to longer context
- supports pretraining and fine-tuning
- provides a stronger comparison point against recent related work
- includes rigorous ablation to attribute improvements to specific design choices (architecture, features, context length)
- supports both quantitative evaluation and qualitative musical analysis suitable for ISMIR publication

## References

- Taein Kim and Yunho Kim, "Piano Velocity Prediction Using a Seq2Seq Model with Attention Mechanism," 2023.
- Zhanhao He et al., "Filling MIDI Velocity using U-Net Image Colorizer," 2025.
- Ashish Vaswani et al., "Attention Is All You Need," 2017.
- Cheng-Zhi Anna Huang et al., "Music Transformer," 2018.
