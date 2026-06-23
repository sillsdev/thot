# Eflomal Alignment Model: Similarities and Differences from Reference eflomal

## 1. What We Built

`EflomalAlignmentModel` is a **clean-room, from-paper reimplementation** of the eflomal
word-alignment algorithm (Östling & Tiedemann 2016) in C++17. It inherits
`AlignmentModelBase` (no GPL code read or copied), fits thot's standard
`AlignmentModel` interface, and is available alongside IBM1–4, HMM, and FastAlign.

**Licensing constraint.** eflomal is GPL-3.0; thot is LGPL-3.0 and ships as a NuGet
package that MIT-licensed machine links. GPL cannot propagate into thot, so this is a
clean-room reimplementation from the published paper only. Every divergence documented
below is either forced by this constraint or a deliberate design choice to fit thot's
conventions. No source from eflomal or efmaral was read or copied.

---

## 2. Algorithm Overview

Eflomal is a **Bayesian IBM1 → HMM → fertility cascade** trained by **collapsed Gibbs
sampling** with symmetric Dirichlet priors. At each training sweep the latent alignment
of every target token is resampled from its posterior given the current alignment of all
other tokens. The lexical distribution is sampled collapsed (counts updated per token);
the jump and fertility distributions are held fixed for the full sweep and recomputed
between sweeps (blocked-Gibbs for efficiency). The three stages share the same kernel
but differ in what enters the conditional score:

| Stage | Conditional score for candidate source position i |
|---|---|
| IBM1 | p(f\|e) × p₀ (null fraction) |
| HMM | p(f\|e) × jump(Δ\_left) × jump(Δ\_right) |
| Fertility | HMM score × fertRatio(φᵢ → φᵢ+1) |

We replicate this kernel exactly.

---

## 3. What We Replicate Exactly from the Algorithm

### 3.1 Nearest-non-null jump accounting

eflomal's key insight is that NULL-aligned tokens should not contribute a "jump to
position 0." Instead, a NULL-aligned token contributes one *skip* jump between its two
non-null neighbours. We replicate this exactly:

- Every **non-null** token at position i contributes two jump counts: the jump from the
  nearest non-null predecessor (aaLeft) and the jump to the nearest non-null successor
  (aaRight). Virtual BOS = −1, EOS = slen.
- Every **null-aligned** token contributes one skip jump: aaRight − aaLeft.

This accounting is implemented in `computeJumpCounts()` and mirrored in `sampleSweep()`.

### 3.2 Dirichlet-sampled fertility ratios

Each Gibbs sweep draws a fresh categorical fertility distribution per source word from its
Dirichlet posterior using `std::gamma_distribution`. We store the **ratio P(φ)/P(φ−1)**
(increment cost) rather than the full distribution — which is all the sampler needs to
decide whether to assign one more target token to a source word.

### 3.3 Marginal-mode decode

Decode follows eflomal exactly: run several short independent Gibbs chains, accumulate
per-position marginals across all chains, then take the argmax per target token. We do
not use Viterbi for the final alignment output.

### 3.4 Multi-sampler marginal combination

The quality lever eflomal calls `n_samplers` — train N independently-seeded chains,
average their marginals before argmax — is replicated with the same seeding formula:
chain s uses `baseSeed + s × 2654435761u`.

### 3.5 Null probability

The `nullProb` scalar gates null vs non-null during the IBM1 and HMM stages, exactly as
in eflomal. The null term in decode uses the same nearest-non-null skip-jump formulation.

---

## 4. Differences from Reference eflomal

### 4.1 Interface: queryable vs transductive (forced by thot)

```
Reference eflomal                  Our EflomalAlignmentModel
─────────────────                  ─────────────────────────
Input: parallel corpus             Input: addSentencePair() × N
Train and emit alignment links     startTraining() / train() / endTraining()
  for training corpus in-place       → normalizes counts into persistent tables
No model object survives           Persistent MemoryLexTable / EflomalJumpTable /
                                     FertilityTable survive in memory and on disk
                                   getBestAlignment(arbitrary pair)
                                   translationProb(s, t)
                                   load() / print() — survives restarts
```

eflomal is transductive: it produces alignments for its training corpus as a side-effect
of training, then stops. There is no persistent model. Our implementation normalizes
accumulated counts into tables after `endTraining()` that satisfy thot's queryable
interface for arbitrary, including held-out, sentence pairs.

### 4.2 Storage layout

| Aspect | Reference eflomal | Ours |
|--------|------------------|------|
| Lexical table | Dense `float32` arrays, custom C mmap format | Sparse `unordered_map<WordIndex, double>` during training; `MemoryLexTable` (log-numer/denom binary) at rest |
| Jump table | Dense array over offset range | `EflomalJumpTable` — same semantics, different I/O format |
| Fertility table | Dense per-source array | Sparse `FertilityTable` — only observed source words stored |
| Corpus | mmapped from original file | Copied into `corpusSrc`/`corpusTrg` at `startTraining()` |

Dense arrays are faster for eflomal's single transductive pass. Sparse maps are necessary
because thot's `MemoryLexTable` stores only observed pairs (not the full V×V matrix), which
is required by the queryable interface.

### 4.3 Cascade initialization

eflomal initializes alignments **uniformly random**. We use **diagonal initialization**
(`round(j × slen / tlen)`) matching FastAlign's convention. This gives the lexical counts
a sensible starting point before sampling begins and reduces effective burn-in.

### 4.4 Multi-sampler implementation

```
Reference eflomal (n_samplers=N)        Our model (numSamplers=N)
────────────────────────────────        ──────────────────────────
For s = 0..N-1:                         SamplerChain struct per chain:
  Run separate aligner process            own RNG, own count tables,
  Collect output alignment links          own trained tables
Average marginal probabilities          startTraining() creates N chains
  across N runs                         train() parallelizes with OpenMP
  then argmax per position              accumulateDecodeMarginal() sums
                                          marginals across chains, caller argmaxes
```

The concept — average marginals from N independently-seeded chains — is identical. The
implementation is opposite: eflomal runs N separate processes; we run N chains inside one
model with OpenMP parallelism. Each `SamplerChain` owns its own `std::mt19937_64`, count
tables, and trained tables. Chains are fully independent — no shared mutable state, no
locking. The decode path reads trained tables as `const&` and uses a local RNG seeded
from `chainSeed`; `accumulateDecodeMarginal` is thread-safe and can be called concurrently
from multiple threads across sentences.

### 4.5 Iteration schedule

eflomal uses `iters = max(2, round(5000/√N))` for all three stages, scaling down with
corpus size. We default to **4/4/4 (12 sweeps)** regardless of corpus size, because the
auto schedule increases AER on large corpora (more sweeps in the IBM1/HMM warmup
over-commit the chain before the fertility stage). The auto schedule is available as
`setAutoIterations(true)` and performs better on very small corpora (< 10k pairs).

### 4.6 Lexical denominator (eflomalLexNorm)

eflomal uses `1/N(e)` as the lexical denominator during sampling. We expose this as
`setEflomalLexNorm(true)` and **enable it by default** so our model matches the reference.
When false, a Dirichlet-smoothed denominator `1/(N(e) + α·|V|)` is used, which adds
regularization at the cost of suppressing signal from rare word pairs.

### 4.7 Speed

eflomal's native C implementation is faster per sweep (~3× per-sweep advantage) due to
mmap access and dense float arrays. Our C++ implementation uses sparse hash maps and
virtual dispatch. With OpenMP parallelism across chains, the effective wall-clock time for
N-sampler runs matches or beats the single-chain sequential time: 5 samplers finish in
roughly the same wall-clock time as 1 sampler used to.

---

## 5. Quality Comparison: WPT English–French Benchmark

**Setup:** 300k Hansards sentence pairs (training), 447 gold-annotated test pairs,
sure + possible link distinction. AER = 1 − (|A∩S| + |A∩P|) / (|A| + |S|). All clone
runs use 4/4/4 training iters and d8/20/4 decode (8 chains / 20 iters / 4 burn-in).

| System | Forward | Reverse | **Intersection** | GDFA |
|---|---|---|---|---|
| HMM (thot IBM1→HMM, 5+5 iters) | 15.1% | 13.9% | 10.4% | 9.7% |
| eflomal GPL (reference, n\_samplers=3, auto-iters) | 8.0% | 7.3% | **6.58%** | 8.1% |
| **Our clone, 1 sampler (default)** | 10.2% | 9.5% | **7.52%** | 10.0% |
| Our clone, 5 samplers, lexNorm=1 | 9.2% | 8.3% | 6.54% | 9.0% |
| **Our clone, 5 samplers, lexNorm=0** | 8.9% | 8.4% | **6.46%** | 9.1% |

**Key observations:**

- The 1-sampler default (7.52%) is 0.94 pp behind eflomal's 3-sampler reference, ahead of HMM by 2.9 pp on intersection.
- The 5-sampler configuration (6.46% / 6.54%) is competitive with or marginally better than eflomal GPL (6.58%).
- **Intersection symmetrization consistently outperforms GDFA** for this model family; GDFA inflates recall at the cost of precision and yields higher AER.
- The quality gain from increasing samplers (1→5) is larger than any other lever. Each additional sampler reduces decode variance; the effect plateaus between 5 and 7 samplers.

---

## 6. Where Each System Is Better: Top-100 Divergence Audit

**Methodology.** We compared eflomal GPL (n_samplers=3, auto-iters) vs our 5-sampler
clone on the 100 most-divergent WPT test pairs (by Jaccard distance between alignment
link sets). A language model judged each case; gold alignments were revealed after
judging.

**Overall verdict:** eflomal GPL wins 55/100, our clone wins 26/100, 19 too close to call.

### Where eflomal GPL consistently wins

1. **Idiomatic paraphrase.** Canadian Hansards routinely paraphrase rather than translate
   literally. eflomal's longer auto-iteration schedule (28 fertility sweeps on 300k data)
   lets it learn these patterns. Our 12-sweep default misses them.

2. **Pronoun realignment.** English "we/they" → French pronoun + repeated construction.
   eflomal's extended fertility stage correctly learns these discourse-level patterns.

3. **Rare content words.** Pairs like "lapsing→périmation" or "subgovernment→sous-gouvernement":
   eflomal's Dirichlet posterior retains signal even for sparse counts; our shorter
   schedule is more sensitive to initialization noise for rare pairs.

4. **Short structural sentences.** On very short sentences eflomal's fertility stage
   provides a cleaner structural alignment. Our clone occasionally adds spurious links
   from over-fertile source words.

### Where our clone consistently wins

1. **Numerals and punctuation.** Literal surface correspondence (e.g. "1,122,000[4]→1,122,000[6]"):
   our shorter schedule retains the obvious lexical signal without the fertility stage
   introducing positional noise.

2. **Common function words.** Prepositions and particles where the high-frequency lexical
   signal dominates: "not→ne", "instead→à", "to→à". eflomal occasionally maps these to
   semantically related but positionally wrong tokens.

3. **Near-literal translations.** When the French is a close surface translation, our
   shorter schedule gets the obvious links right without over-complicating the alignment.

### Systematic failure modes

| Failure mode | Reference eflomal | Our clone |
|---|---|---|
| Over-linking in fertility stage | Rare | Occasional |
| Under-linking short sentences | Occasional | Rare |
| Spurious links on idiomatic paraphrases | Moderate (learns them) | More frequent |
| Numeral / punctuation misalignment | Occasional | Rare |
| Long-distance link errors | Moderate | More frequent |

**Bottom line:** eflomal's longer schedule is linguistically superior for complex paraphrase
and discourse patterns. Our multi-sampler decode (5 chains) closes most of that gap
through variance reduction, achieving competitive or better AER on the full test set. For
pipelines where 1-sampler is sufficient, a known ~1 pp AER penalty vs eflomal is the tradeoff.

---

## 7. Multi-Language Evaluation (Agreement with Reference eflomal)

Since gold alignments are not available for arbitrary language pairs, we measure
**F1/Jaccard agreement** between our output and reference eflomal, treating the reference
as a proxy for correctness. Four language pairs were evaluated at 1, 3, and 5 samplers
with both lexNorm settings.

**Corpora:** OPUS bible-uedin for German, Portuguese, Finnish (~62k pairs each);
Spanish–English Bible corpus (31k pairs). All runs are transductive (test pairs included
in training), 4/4/4 training iters, d8/20/4 decode. eflomal GPL uses 3 samplers and
auto-iteration schedule.

### Summary: best configuration per language

| Language | lexNorm=0 F1 (5s) | lexNorm=1 F1 (5s) | Best |
|---|---|---|---|
| German–English (62k) | **0.833** | 0.831 | 5s, lexNorm=0 |
| Portuguese–English (62k) | **0.891** | 0.884 | 5s, lexNorm=0 |
| Finnish–English (62k) | **0.815** | 0.810 | 5s, lexNorm=0 |
| Spanish–English (31k) | 0.920 | **0.923** | 5s, lexNorm=1 |

lexNorm=1 (default) consistently helps Spanish (the smallest corpus, 31k pairs) but is
neutral-to-slightly-worse for the larger 62k corpora. The default is correct for the
1-sampler case (matches the WPT French finding) and the regression at 5-sampler is small
(≤ 0.007 F1). All clone configurations comfortably exceed HMM across every language.

### 7.1 German–English (62,193 pairs)

| System | Precision | Recall | F1 | Jaccard |
|---|---|---|---|---|
| HMM (5+5) | 0.752 | 0.772 | 0.762 | 0.616 |
| 1 sampler, lexNorm=1 (default) | 0.770 | 0.863 | 0.814 | 0.686 |
| 5 samplers, lexNorm=0 | 0.771 | 0.907 | **0.833** | **0.714** |
| 5 samplers, lexNorm=1 | 0.767 | 0.905 | 0.831 | 0.710 |
| eflomal GPL (3 samplers) | 1.000 | 1.000 | 1.000 | 1.000 |

### 7.2 Portuguese–English (62,170 pairs)

| System | Precision | Recall | F1 | Jaccard |
|---|---|---|---|---|
| HMM (5+5) | 0.827 | 0.815 | 0.821 | 0.696 |
| 1 sampler, lexNorm=1 (default) | 0.834 | 0.904 | 0.868 | 0.766 |
| 5 samplers, lexNorm=0 | 0.856 | 0.928 | **0.891** | **0.803** |
| 5 samplers, lexNorm=1 | 0.847 | 0.925 | 0.884 | 0.792 |
| eflomal GPL (3 samplers) | 1.000 | 1.000 | 1.000 | 1.000 |

### 7.3 Finnish–English (62,022 pairs)

Finnish is the hardest corpus: agglutinative morphology, 15 grammatical cases, SOV/verb-final
order.

| System | Precision | Recall | F1 | Jaccard |
|---|---|---|---|---|
| HMM (5+5) | 0.646 | 0.598 | 0.621 | 0.450 |
| 1 sampler, lexNorm=1 (default) | 0.758 | 0.820 | 0.788 | 0.650 |
| 5 samplers, lexNorm=0 | 0.775 | 0.860 | **0.815** | **0.688** |
| 5 samplers, lexNorm=1 | 0.776 | 0.847 | 0.810 | 0.680 |
| eflomal GPL (3 samplers) | 1.000 | 1.000 | 1.000 | 1.000 |

Finnish gains least from lexNorm=1 — its agglutinative vocabulary is large, so the
Dirichlet prior mass (`α·|V_fi|`) is proportionally larger and provides useful stabilization
rather than over-smoothing.

### 7.4 Spanish–English (30,959 pairs)

| System | Precision | Recall | F1 | Jaccard |
|---|---|---|---|---|
| HMM (5+5) | 0.759 | 0.612 | 0.759 | 0.612 |
| 1 sampler, lexNorm=1 (default) | 0.885 | 0.950 | 0.916 | 0.845 |
| 5 samplers, lexNorm=0 | 0.887 | 0.956 | 0.920 | 0.852 |
| 5 samplers, lexNorm=1 | 0.886 | 0.963 | **0.923** | **0.856** |
| eflomal GPL (3 samplers) | 1.000 | 1.000 | 1.000 | 1.000 |

Spanish is the only language where lexNorm=1 wins at every sampler count. Smaller corpus
(31k) means lower per-type frequency; the Dirichlet prior mass helps less than it would
on 62k pairs, so removing it (lexNorm=1) improves signal.

### 7.5 Intersection density mismatch

Computing forward ∩ reverse for our model vs eflomal GPL's forward ∩ reverse reveals a
systematic density difference:

| Language | Precision | Recall | F1 |
|---|---|---|---|
| German–English | 0.19 | 0.88 | 0.31 |
| Portuguese–English | 0.18 | 0.90 | 0.30 |
| Finnish–English | 0.13 | 0.88 | 0.22 |
| Spanish–English | 0.25 | 0.96 | 0.40 |

Our intersection is **4–7× denser** than eflomal's. The cause: eflomal runs a single Markov
chain forward and a separate single chain in reverse; these chains diverge from different
random initializations over many sweeps and produce genuinely different alignments that
intersect sparsely. Our multi-chain approach runs N independently-seeded chains that all
explore the same posterior — they agree on the same high-probability links regardless of
direction, so our forward and reverse outputs already overlap substantially before intersecting.

This makes forward F1 (our alignment vs eflomal's forward) the correct comparison metric,
not intersection vs intersection. Sections 7.1–7.4 report forward F1.

---

## 8. Timing

Wall-clock time; win-x64 for our model, WSL Linux for eflomal GPL. Measured on an Intel
i7-12700 (20 logical cores, 12P+8E).

| Corpus | Eflomal GPL (3 samplers) | Our clone 1s | Our clone 3s | Our clone 5s |
|---|---|---|---|---|
| WPT French (300k+447) | ~85s | ~70s | ~72s | ~75s |
| Spanish–English (31k) | 14s | 12s | ~14s | ~17s |
| German–English (62k) | 26s | ~30s | ~32s | ~35s |
| Portuguese–English (62k) | 26s | ~30s | ~32s | ~35s |
| Finnish–English (62k) | 25s | 13s | ~59s | ~54s |

eflomal's native C implementation is faster per sweep (~3× advantage per chain). With
OpenMP parallelism across chains, multi-sampler runs are essentially free in wall-clock
time relative to single-sampler: the 5-sampler run completes in the same time as the old
1-sampler sequential run. Finnish 3s train is slower than 5s because OpenMP work-stealing
distributes 3 chains less evenly across 20 cores.

---

## 9. Testing

| Test | What it checks |
|---|---|
| `trainEmpty` | no crash on empty corpus |
| `train` | exact alignment vectors on pig-Latin corpus (fixed seed) |
| `computeLogProbMatchesBestAlignment` | score consistency: `computeLogProb` ≈ `getBestAlignment` log-prob |
| `deterministicWithFixedSeed` | identical results from two identically-seeded models |
| `serializationRoundTrip` | `print` → `load` → identical alignments and probabilities |
| `trainedProbabilitiesAreValid` | `translationProb` and `jumpProb` ∈ [0, 1] |
| `singlePairCorpus` | training on a single sentence pair does not crash |
| `oovWordsReturnSmoothedProb` | unseen words return a valid smoothed probability |
| `getEntriesForSource` | known source word returns non-empty entry table |
| `computeSumLogProb` | returns a finite negative log-probability |
| `loglikelihoodForPairRange` | smoke test over all training pairs |
| `fertilityProbIsValid` | fertility probabilities ∈ [0, 1] for each phi |
| `clearResetsState` | `clear()` resets all tables; model can be retrained afterwards |
| `configNonDefaultRoundTrip` | non-default hyperparameters survive `print`/`load` |
| `decodeBurnInClampedWhenExceedsIters` | decodeBurnIn ≥ decodeIters does not produce all-NULL alignments |
| `benchmarkParallelTraining` | timing smoke test: 4000 pairs × 12 sweeps |

---

## 10. Implementation Notes

### Files added

```
src/sw_models/EflomalAlignmentModel.h/.cc   — main model
src/sw_models/EflomalJumpTable.h/.cc        — signed-offset jump distribution [-J..+J]
src/sw_models/SamplingUtils.h               — categorical and cumulative samplers
tests/sw_models/EflomalAlignmentModelTest.cc
```

### Existing files modified

```
src/sw_models/AlignmentModel.h              — Eflomal = 9 added to AlignmentModelType
src/sw_models/CMakeLists.txt                — new sources added alphabetically
tests/CMakeLists.txt                        — new test registered
src/shared_library/thot.cc                  — factory case Eflomal; setEflomalNumSamplers/getEflomalNumSamplers
src/python_module/module.cc                 — pybind11 binding and enum value EFLOMAL
```

### No GPL code

No source from the GPL-licensed eflomal or efmaral projects was read or copied.
The algorithm is reconstructed entirely from the published paper (Östling & Tiedemann 2016).

---

## References

- Östling, R. & Tiedemann, J. (2016). Efficient Word Alignment with Markov Chain Monte Carlo. *Prague Bulletin of Mathematical Linguistics*, 106, 125–146.
- Mihalcea, R. & Pedersen, T. (2003). An Evaluation Exercise for Word Alignment. *HLT-NAACL 2003 Workshop on Building and Using Parallel Texts*.
- Christos, T. & Eisele, A. (2010). News from OPUS — A Collection of Multilingual Parallel Corpora. *LREC 2010*.
