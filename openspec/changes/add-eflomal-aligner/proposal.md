## Why

Thot offers IBM 1–4, HMM, and FastAlign word aligners, but lacks a Bayesian
MCMC aligner. Eflomal (Östling & Tiedemann, 2016) is a fast, low-memory
IBM1→HMM→fertility cascade trained by collapsed Gibbs sampling; adding word-order
and fertility models does not increase its inference cost (unlike EM), and it is
competitive on alignment error rate while using far less memory than GIZA++.
Making it a first-class thot model also makes it available to `machine`, which
wraps thot natively.

Eflomal's own source is **GPL-3.0**; thot is **LGPL-3.0** and its binary ships as
the `Thot` NuGet that **MIT-licensed `machine`** links. Copying GPL code is
therefore prohibited. This change delivers a **clean-room reimplementation from
the published algorithm** — no eflomal source is read or copied — that fits thot's
existing alignment-model architecture.

## What Changes

- Add a new `EflomalAlignmentModel` (C++) to thot's `sw_models`, implementing the
  full Bayesian IBM1→HMM→fertility Gibbs-sampling cascade with Dirichlet priors.
- The model trains by sampling but persists normalized distributions (lexical,
  jump, fertility) and decodes deterministically (Viterbi + fertility
  hill-climbing) so it satisfies thot's queryable/serializable `AlignmentModel`
  interface, including for held-out sentence pairs.
- Add `Eflomal` to the `AlignmentModelType` enum and expose it through the
  C shared library and the pybind11 Python module (`EFLOMAL`).
- Add a seedable RNG / categorical-sampling utility (thot has none in `sw_models`)
  with a deterministic test mode for reproducible, exact-match unit tests.
- Add unit tests that **meet or exceed the existing HMM model's coverage**, plus
  added coverage for the new tables, serialization round-trip, and determinism.
- Reuse permissively licensed components where available (MIT `SIMD-math-prims`
  for the precision/SIMD optimization phase; C++ `<random>` for sampling). No
  GPL/efmaral/eflomal code is used.
- A dedicated **optimization phase** after the model is correct (big-O → precision
  → SIMD/x86 on the hot loops) and a **style-review phase** against thot's
  conventions.
- Add a `.NET` binding in `machine` (`ThotEflomalWordAlignmentModel`) mirroring
  the FastAlign integration, built and tested against a **locally built thot**,
  with a **real-data end-to-end test and timing** using a corpus available in
  `machine`.

## Capabilities

### New Capabilities
- `eflomal-alignment-model`: thot's C++ Bayesian Gibbs-sampling word aligner —
  training cascade, deterministic decoding/scoring, serialization, reproducibility,
  and exposure through the C library and Python bindings.
- `eflomal-dotnet-binding`: the `machine` .NET wrapper that exposes the thot
  Eflomal model through `IWordAlignmentModel`, including end-to-end alignment of
  real parallel data with verified timing.

### Modified Capabilities
<!-- None: openspec/specs/ is empty (OpenSpec newly adopted); this change introduces the first capabilities. -->

## Impact

- **thot (new):** `src/sw_models/EflomalAlignmentModel.{h,cc}`,
  `EflomalJumpTable.{h,cc}`, `SamplingUtils.h`,
  `tests/sw_models/EflomalAlignmentModelTest.cc`.
- **thot (edited):** `src/sw_models/AlignmentModel.h` (enum),
  `src/sw_models/CMakeLists.txt`, `tests/CMakeLists.txt`,
  `src/shared_library/thot.cc`, `src/python_module/module.cc`.
- **packaging:** a new `Thot` NuGet release containing the model is required
  before `machine` can build against it (a local package is used during development).
- **machine (new/edited):** `ThotEflomalWordAlignmentModel.cs`,
  `ThotWordAlignmentModelType.cs`, `Thot.cs`, `ThotWordAlignmentModel.cs`,
  `ThotWordAlignmentModelTrainer.cs`, `ThotWordAlignmentParameters.cs`,
  `AlignmentModelCommandSpec.cs`, `.csproj` Thot version bump, and
  `tests/SIL.Machine.Translation.Thot.Tests/ThotEflomalWordAlignmentModelTests.cs`.
- **dependencies:** optional MIT `SIMD-math-prims` header (optimization phase only);
  no new runtime dependencies.
