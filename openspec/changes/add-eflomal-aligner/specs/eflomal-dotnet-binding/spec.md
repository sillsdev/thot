## ADDED Requirements

### Requirement: Eflomal exposed through machine's word-alignment API
The `machine` library SHALL expose the thot Eflomal model as a
`ThotEflomalWordAlignmentModel` implementing `IWordAlignmentModel` (and
`IHmmWordAlignmentModel`, since it provides a jump probability), selectable via the
`ThotWordAlignmentModelType` enum and the model factory, mirroring the FastAlign
integration.

#### Scenario: Model is selectable and creatable
- **WHEN** an Eflomal model is requested by enum value or CLI model name
- **THEN** the factory returns a `ThotEflomalWordAlignmentModel`
- **AND** it aligns sentence pairs through `Align`/`AlignBatch` and returns
  translation scores

#### Scenario: Trainable with Eflomal parameters
- **WHEN** a trainer is created with Eflomal iteration/seed/null-prob parameters and
  run on a parallel corpus
- **THEN** training completes and produces a model that can be saved and reloaded

### Requirement: Builds against a locally built thot
The binding SHALL build and pass its tests against a locally built thot package
(local NuGet feed), so it can be validated before a public `Thot` release.

#### Scenario: Local thot build wired up
- **WHEN** thot is built locally and packaged, and `machine` is pointed at the local
  package
- **THEN** `SIL.Machine.Translation.Thot` and its test project build successfully
- **AND** the Eflomal binding tests run against the local native library

#### Scenario: Binding tests mirror FastAlign tests
- **WHEN** the machine Thot test suite is run
- **THEN** the Eflomal tests cover Align, AlignBatch, translation probability, vocab
  counts, symmetrized model, and corrupted-model error handling, and pass

### Requirement: Real-data end-to-end alignment with timing
The system SHALL demonstrate end-to-end alignment on a real parallel corpus
available in `machine`, verifying plausible output and recording timing.

#### Scenario: Real corpus aligns within expected time
- **WHEN** forward and reverse Eflomal models are trained on a real parallel corpus
  and symmetrized through the existing symmetrization path
- **THEN** the produced alignments are non-empty and plausible
- **AND** training/alignment time and memory are recorded and are no worse than the
  HMM model on the same corpus
