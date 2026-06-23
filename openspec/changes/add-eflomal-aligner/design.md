## Context

Thot's `AlignmentModel` (`src/sw_models/AlignmentModel.h`) is the contract every
aligner implements; `AlignmentModelBase` supplies shared scaffolding (vocab,
sentence handler, word classes, YAML config plumbing). Existing models are
EM-trained and **queryable/serializable**: they expose `translationProb(s,t)`,
`getBestAlignment` for arbitrary pairs, `computeLogProb`, `computeSumLogProb`, and
`load`/`print`. Training is driven externally as `startTraining()` →
`train()` (per iteration) → `endTraining()` (see `tests/sw_models/TestUtils.cc:78`).

Eflomal is structurally different: a **transductive batch Gibbs sampler** over the
training corpus with Dirichlet priors on lexical, jump, and fertility
distributions. Its native artifact is alignments + priors, not a queryable model.
The closest existing precedent is `FastAlignModel` (a third-party algorithm fit to
the interface, persisting a lexical table + a scalar tension parameter). The
fertility machinery and move/swap decoding in `Ibm3AlignmentModel` are directly
relevant since Eflomal also models fertility.

Licensing is a hard constraint: eflomal = GPL-3.0, thot = LGPL-3.0, machine = MIT.
No eflomal/efmaral source may be read or copied.

## Goals / Non-Goals

**Goals:**
- A clean-room `EflomalAlignmentModel` that fits thot's architecture and passes the
  same kinds of exact-match tests the other models do.
- Test coverage at least equal to the HMM model, plus coverage for the new tables,
  serialization, and determinism.
- Competitive alignment quality and memory/time no worse than HMM on the toy corpus.
- A measurable optimization phase and a style-conformance phase.
- A `machine` .NET binding, buildable against a local thot, validated on real data
  with timing.

**Non-Goals:**
- Incremental/online training (Eflomal is batch-only, like IBM3/4).
- Word-class–conditioned distributions (eflomal proper does not use them; classes
  are accepted for API symmetry but unused by the sampler).
- Symmetrization inside the model — handled by the existing higher-level
  symmetrization path (two directional models), as for every other model.
- Bit-identical parity with upstream eflomal numbers (different code, different RNG).

## Decisions

### D1. Train by sampling, persist distributions, decode by marginal-mode sampling
The trained model persists three normalized tables — lexical p(t|s)
(`MemoryLexTable`), jump p(Δ) (new `EflomalJumpTable`), fertility p(φ|s)
(`FertilityTable`) — plus scalars (null prob, sentence-length stats). Queries read
only these tables. To **match eflomal's alignment extraction**, `getBestAlignment`
samples the pair against the trained distributions over a few short sampler chains,
accumulates the normalized per-position conditional into a per-target marginal, and
takes the **argmax per target token** (eflomal's final argmax pass) — *not* a Viterbi
MAP path. This is made deterministic with a fixed-seeded, thread-local
`std::mt19937_64`. `computeLogProb` scores a given alignment from the tables;
`computeSumLogProb` returns the decoded-alignment score. *Alternative rejected:*
Viterbi MAP decode — it was the initial implementation (F1 0.902 vs eflomal) but the
marginal-mode decode matches eflomal's per-word posterior far better (F1 0.92+).

The sampler also mirrors eflomal's exact conditional: jumps use **nearest-non-NULL**
neighbours (a NULL token contributes one "skip" jump between its neighbours, a
non-NULL token an incoming + outgoing jump), and fertility is a **Dirichlet draw**
per source word per sweep (via `std::gamma_distribution`), capped at 7. Jump and
fertility distributions are held fixed per sweep (blocked Gibbs) while the lexical
table is sampled collapsed.

### D2. Map start/train/end onto the Gibbs cascade
`startTraining()` builds the in-memory corpus + latent alignment state + sparse
count tables and seeds the RNG. Each `train()` runs one Gibbs sweep, with the
active stage (IBM1 / +HMM jump / +fertility) selected from an `iter` counter vs a
configured schedule (mirrors `FastAlignModel`'s `iter` use). `endTraining()`
normalizes accumulated (variance-reduced) counts into the persisted tables and
drops sampler state. Stage boundaries clamp so extra `train()` calls extend the
last stage. *Alternative rejected:* one model = one full cascade hidden inside a
single `train()` call — breaks the per-iteration driver the test harness uses.

### D3. New jump table parametrized by signed offset
Eflomal's jump distribution is over the signed offset Δ = i − prev_i, bucketed into
a window [−J..+J] plus a null bucket, **independent of source length** — unlike
`HmmAlignmentTable` which keys on `(prev_i, slen)`. A small new `EflomalJumpTable`
(structural twin of `FertilityTable`: log-numerator vector + log-denominator) is
cleaner and lower-memory than bending the HMM table. *Alternative rejected:* reuse
`HmmAlignmentTable` — wastes memory and mismatches the model.

### D4. Reuse vs new
- **Reuse:** `MemoryLexTable`, `FertilityTable`, `LexCounts`/`OrderedVector`
  (sparse counts), `AlignmentInfo` + IBM3 move/swap decode, a sentence-length
  model, all of `AlignmentModelBase`, `Md`/`MathFuncs` log-space helpers.
- **New:** the Gibbs kernel (decrement → conditional → sample → increment), the
  cascade scheduler + count averaging, `EflomalJumpTable`, and a header-only
  `SamplingUtils.h` (seedable `std::mt19937_64` + categorical/Dirichlet draws —
  thot has no RNG in `sw_models`).

### D5. Determinism strategy
`std::mt19937_64` with a fixed default seed (portable across platforms/compilers
for a literal seed). A `deterministic` flag gates the `#pragma omp parallel for`
on `train()` and `endTraining()` via an `if(!deterministic)` OpenMP clause, so
the training loop runs single-threaded and in visitation order when determinism
is required — this gives bit-reproducible training results within a platform so
tests can assert exact `std::vector<PositionIndex>`. Decode (`sampleDecode` /
`decodeMarginalFromTables`) is inherently deterministic: each decode call uses
a fixed per-chain seed and reads only from immutable trained tables, so no
parallelism guard is needed there. In production mode (deterministic=false),
parallelism is across **independent chains** (each with its own RNG seeded
`seed + s*2654435761u`) — order-independent, so normalized tables are
reproducible-per-seed and decode is always deterministic.

### D6. Permissively licensed reuse for the statistical/optimization side
RNG and sampling use the C++ standard library (no third-party). For the
precision/SIMD optimization phase, the MIT-licensed `SIMD-math-prims`
(github.com/jhjourdan/SIMD-math-prims) provides vectorizable `exp`/`log`
approximations and may be vendored with its license header. No GPL code is used.

### D7. machine integration mirrors FastAlign and builds against local thot
`machine` is enum-driven P/Invoke over the `Thot` NuGet. Adding
`ThotEflomalWordAlignmentModel` (implementing `IHmmWordAlignmentModel`, since it has
a jump probability) plus enum/factory/trainer/parameter entries mirrors FastAlign
exactly. During development, machine builds against a **locally produced Thot
package** (local NuGet feed / `nuget.config`) so the binding and end-to-end test can
be validated before a public Thot release.

## Risks / Trade-offs

- **Gibbs count bookkeeping errors** (off-by-one in fertility/null deltas silently
  degrade quality) → add an assert-mode that re-derives counts from the current
  alignment each sweep; cross-check `computeLogProb` vs `getBestAlignment`.
- **Cross-platform determinism** (float summation order, threading) → single-thread
  fixed-order deterministic mode; avoid order-dependent `double` sums in that mode;
  rely on `std::mt19937_64`'s standard-defined output.
- **Decode/score inconsistency** (Viterbi+hill-climb vs `computeLogProb`) → a test
  asserts they agree (as in `FastAlignModelTest.cc`).
- **Quality regression vs upstream eflomal** → validate AER on real data against
  HMM/FastAlign baselines; tune priors/iteration schedule, not parity with upstream.
- **machine blocked on Thot release** → use a local Thot package during development;
  public release is a separate packaging step.
- **Optimization destabilizing correctness** → optimization phase starts only after
  tests pass and re-runs the full suite after each step (big-O, precision, SIMD).

## Migration Plan

Additive only — no existing behavior changes. New enum value is **appended**
(`Eflomal = 9`); existing values are not renumbered (they are serialized and
exposed through the Python enum). Rollback = remove the new files and the additive
enum/registration lines.

## Open Questions

- Default iteration schedule and Dirichlet hyperparameters (lexical/jump/fertility,
  null prob) — pick eflomal-documented defaults, then tune on the real-data test.
- Jump window size J and large-offset clamping behavior — confirm during tuning.
- Which real corpus in `machine` to use for the end-to-end timing test (e.g. the
  existing toy/parallel corpora under the test fixtures) — confirm in the testing phase.
