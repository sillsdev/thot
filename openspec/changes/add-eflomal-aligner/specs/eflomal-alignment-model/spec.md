## ADDED Requirements

### Requirement: Eflomal model is a selectable alignment model
The system SHALL provide an `EflomalAlignmentModel` that implements thot's
`AlignmentModel` interface and is identified by a new `AlignmentModelType::Eflomal`
value. The model SHALL be a clean-room implementation of the eflomal algorithm
written from its published description; no GPL-licensed eflomal/efmaral source may
be incorporated.

#### Scenario: Model type is registered
- **WHEN** `EflomalAlignmentModel::getModelType()` is called
- **THEN** it returns `AlignmentModelType::Eflomal`
- **AND** `getModelTypeStr()` returns `"eflomal"`

#### Scenario: Created through the factory and Python enum
- **WHEN** an Eflomal model is created via the C shared library factory or the
  Python `EFLOMAL` enum value
- **THEN** a usable `EflomalAlignmentModel` instance is returned

### Requirement: Bayesian Gibbs-sampling training cascade
The model SHALL train with collapsed Gibbs sampling over a configurable
IBM1 → HMM(jump) → fertility cascade using Dirichlet priors, driven by the standard
`startTraining()` / `train()` / `endTraining()` sequence. After training it SHALL
hold normalized lexical, jump, and fertility distributions.

#### Scenario: Training the toy corpus produces expected alignments
- **WHEN** the standard test corpus is added and the model is trained for the
  configured number of iterations with a fixed seed in deterministic mode
- **THEN** `getBestAlignment` returns the expected `PositionIndex` vectors for the
  standard probe sentence pairs

#### Scenario: Empty training does not crash
- **WHEN** training is run with no sentence pairs
- **THEN** training completes without error

### Requirement: Deterministic decoding and scoring of arbitrary pairs
Given the trained distributions, the model SHALL deterministically produce the best
alignment (Viterbi over lexical+jump followed by fertility-aware hill-climbing) and
score alignments for arbitrary, including held-out, sentence pairs.

#### Scenario: Best-alignment log-prob matches computeLogProb
- **WHEN** `getBestAlignment` returns an alignment and its log-probability for a pair
- **AND** `computeLogProb` is called on the same pair and alignment matrix
- **THEN** the two log-probabilities are equal within EPSILON

#### Scenario: Lexical, jump, and fertility probabilities are queryable
- **WHEN** `translationProb`, the jump probability, or the fertility probability is
  queried after training
- **THEN** a valid probability in [0,1] is returned, with smoothing for unseen items

### Requirement: Serialization round-trip
The model SHALL persist its trained distributions and configuration via
`print(prefix)` and restore an equivalent model via `load(prefix)`, with training
hyperparameters in YAML config and learned tables in their own files.

#### Scenario: Reload reproduces alignments
- **WHEN** a trained model is printed to a prefix and a new model is loaded from it
- **THEN** the loaded model returns the same best alignments and translation
  probabilities as the original

### Requirement: Reproducibility via seedable RNG
The model SHALL use a seedable RNG and provide a deterministic mode that yields
identical results across runs and platforms for a given seed and input.

#### Scenario: Same seed yields identical results
- **WHEN** two models are trained on the same corpus with the same seed in
  deterministic mode
- **THEN** their best alignments and persisted tables are identical

### Requirement: Batch-only model
The model SHALL NOT implement the incremental training interface
(`IncrAlignmentModel`), consistent with the IBM3/IBM4 models.

#### Scenario: No incremental interface
- **WHEN** the class hierarchy is inspected
- **THEN** `EflomalAlignmentModel` inherits `AlignmentModelBase` only and exposes no
  `incrTrain`/`startIncrTraining`/`endIncrTraining` methods

### Requirement: Test coverage at least equal to the HMM model
The model SHALL ship unit tests that cover at minimum the behaviors covered by the
HMM model's tests (training to expected alignments and `computeLogProb`
consistency), plus added tests for the new jump/fertility tables, serialization
round-trip, and determinism.

#### Scenario: Test suite mirrors and extends HMM coverage
- **WHEN** the thot test suite is run
- **THEN** Eflomal tests assert expected alignments, `computeLogProb` consistency,
  table probabilities, a serialization round-trip, and seed-based determinism, and
  all pass
