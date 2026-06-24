#pragma once

#include "sw_models/AlignmentModelBase.h"
#include "sw_models/EflomalJumpTable.h"
#include "sw_models/FertilityTable.h"
#include "sw_models/MemoryLexTable.h"

#include <tsl/robin_map.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

// Clean-room implementation of the eflomal word-alignment algorithm
// (Oestling & Tiedemann, 2016, "Efficient Word Alignment with Markov Chain
// Monte Carlo"). It is a Bayesian IBM1 -> HMM(jump) -> fertility cascade
// trained by collapsed Gibbs sampling with Dirichlet priors. No source from
// the GPL-licensed eflomal/efmaral projects was read or copied; the algorithm
// is reconstructed from its published description and fitted to thot's
// architecture.
//
// Training samples alignments over the corpus, but the trained model persists
// normalized lexical, jump and fertility distributions and decodes via
// marginal-mode sampling (sampleDecode / decodeMarginalFromTables), so it
// satisfies thot's queryable/serializable AlignmentModel interface for
// arbitrary, including held-out, sentence pairs. Like IBM3/IBM4 it is
// batch-only and does not implement IncrAlignmentModel.
class EflomalAlignmentModel : public AlignmentModelBase
{
public:
  EflomalAlignmentModel();

  AlignmentModelType getModelType() const override
  {
    return Eflomal;
  }

  // Training controls.
  void setSeed(unsigned int s);
  unsigned int getSeed() const;
  // Number of independent Gibbs chains trained in parallel; their decode
  // marginals are summed at alignment time, matching eflomal's n_samplers
  // combination scheme. Each chain uses seed + s*2654435761 as its RNG seed.
  void setNumSamplers(int value);
  int getNumSamplers() const;
  void setDeterministic(bool value);
  bool getDeterministic() const;
  void setIterations(int ibm1, int hmm, int fertility);
  int getIbm1Iters() const;
  int getHmmIters() const;
  int getFertilityIters() const;
  void setAlphaLex(double value);
  double getAlphaLex() const;
  // Dirichlet prior applied to the NULL source word's lexical distribution.
  // Matches eflomal's NULL_ALPHA (default 0.001 = LEX_ALPHA). Separate from
  // alphaLex so NULL can be given a tighter or looser prior independently.
  void setAlphaNull(double value);
  double getAlphaNull() const;
  void setAlphaJump(double value);
  double getAlphaJump() const;
  void setAlphaFertility(double value);
  double getAlphaFertility() const;
  void setJumpWindow(int value);
  int getJumpWindow() const;
  void setNullProb(double value);
  double getNullProb() const;

  // eflomal-fidelity flags. See the add-eflomal-aligner ablation for benchmarks.
  // - eflomalLexNorm (on by default): use the eflomal lexical denominator 1/N(e)
  //   instead of the Dirichlet-smoothed 1/(N(e) + alpha*|V|). The eflomal-style
  //   denominator gives better AER on large corpora (WPT 300k: 7.52% vs 8.05%).
  // - autoIterations (off by default): derive the IBM1/HMM/fertility schedule from
  //   the corpus size as eflomal does: iters = max(2, round(5000/sqrt(N))), all
  //   three stages use the same count.
  void setEflomalLexNorm(bool value);
  bool getEflomalLexNorm() const;
  void setAutoIterations(bool value);
  bool getAutoIterations() const;
  // Decode-time sampler chains / iterations / burn-in for the marginal-mode decode.
  void setDecodeParams(int samplers, int iters, int burnIn);
  // Total number of sweeps the configured schedule requires (valid after
  // startTraining, which resolves the auto schedule). The driver should call
  // train() this many times.
  int getScheduledIterations() const;

  // Accumulates decode marginals from all numSamplers chains into accOut.
  // acc[j][k]: k in [0,slen-1] -> source position k+1, k==slen -> NULL.
  // Each call adds to accOut so multiple calls from separate models compose.
  // Thread-safe when called concurrently for different sentence pairs on the
  // same model (after endTraining), since decode uses only local state.
  void accumulateDecodeMarginal(const std::vector<WordIndex>& srcSentence, const std::vector<WordIndex>& trgSentence,
                                std::vector<std::vector<double>>& accOut);

  unsigned int startTraining(int verbosity = 0) override;
  void train(int verbosity = 0) override;
  void endTraining() override;

  std::pair<double, double> loglikelihoodForPairRange(std::pair<unsigned int, unsigned int> sentPairRange,
                                                      int verbosity = 0) override;

  // Trained distributions. Always query chain[0]'s tables (consistent with
  // scoreAlignment, computeLogProb, and getEntriesForSource).
  Prob translationProb(WordIndex s, WordIndex t) override;
  LgProb translationLogProb(WordIndex s, WordIndex t) override;
  Prob jumpProb(int offset);
  LgProb jumpLogProb(int offset);
  Prob fertilityProb(WordIndex s, PositionIndex phi);
  LgProb fertilityLogProb(WordIndex s, PositionIndex phi);

  Prob sentenceLengthProb(unsigned int slen, unsigned int tlen) override;
  LgProb sentenceLengthLogProb(unsigned int slen, unsigned int tlen) override;

  bool getEntriesForSource(WordIndex s, NbestTableNode<WordIndex>& trgtn) override;

  using AlignmentModel::getBestAlignment;
  LgProb getBestAlignment(const std::vector<WordIndex>& srcSentence, const std::vector<WordIndex>& trgSentence,
                          std::vector<PositionIndex>& bestAlignment) override;
  using AlignmentModel::computeLogProb;
  LgProb computeLogProb(const std::vector<WordIndex>& srcSentence, const std::vector<WordIndex>& trgSentence,
                        const WordAlignmentMatrix& aligMatrix, int verbose = 0) override;
  using AlignmentModel::computeSumLogProb;
  LgProb computeSumLogProb(const std::vector<WordIndex>& srcSentence, const std::vector<WordIndex>& trgSentence,
                           int verbose = 0) override;

  bool load(const char* prefFileName, int verbose = 0) override;
  bool print(const char* prefFileName, int verbose = 0) override;

  void clearSentenceLengthModel() override;
  void clearTempVars() override;
  void clear() override;

  virtual ~EflomalAlignmentModel()
  {
  }

private:
  // Training stages of the cascade.
  enum Stage
  {
    Ibm1Stage = 0,
    HmmStage = 1,
    FertilityStage = 2
  };

  // Fast integer finalizer for the lexical count maps. robin_map uses a
  // power-of-two bucket count and keys off the low hash bits, so it needs strong
  // avalanche there; std::hash for integers does not reliably provide it
  // (identity on libstdc++, byte-wise FNV on MSVC). This is the "lowbias32"
  // mixer from Chris Wellons' hash-prospector (public domain): two multiplies
  // and three xorshifts, with lower bias.
  struct WordIndexHash
  {
    std::size_t operator()(WordIndex x) const noexcept
    {
      std::uint32_t h = static_cast<std::uint32_t>(x);
      h ^= h >> 16;
      h *= 0x7feb352du;
      h ^= h >> 15;
      h *= 0x846ca68bu;
      h ^= h >> 16;
      return h;
    }
  };
  using LexCountMap = tsl::robin_map<WordIndex, float, WordIndexHash>;

  // Per-chain Gibbs sampler state. N chains train in parallel; each has its
  // own RNG, alignment, count tables and trained parameter tables. Decode
  // sums marginals across all chains (eflomal's n_samplers scheme). Chains
  // are independent so training is embarrassingly parallel.
  struct SamplerChain
  {
    unsigned int chainSeed = 0;
    std::mt19937_64 rng;

    // Current corpus alignment (training state).
    std::vector<std::vector<PositionIndex>> alig;

    // Running lexical counts (collapsed sampler: decremented before
    // sampling, incremented after for each target token in each sentence).
    // Open-addressing map with float values: contiguous keys/values packed into
    // one allocation (no per-node pointer chase) + 4-byte values, matching
    // eflomal's count storage. Counts are integer-valued (±1) and bounded under
    // 2^24 so float holds them exactly; the running sums stay double (can exceed
    // float's exact-int range and are not read in the per-candidate loop).
    std::vector<LexCountMap> lexCounts;
    std::vector<double> lexCountSum;

    // Jump and fertility counts (recomputed from the full alignment before
    // each sweep — not collapsed, held fixed within the sweep).
    std::vector<double> jumpCounts;
    double jumpCountSum = 0;
    std::vector<std::vector<double>> fertCounts; // [s][phi]
    std::vector<double> fertCountSum;
    std::vector<std::vector<double>> fertRatioSampled; // [s][phi] = sampled P(phi)/P(phi-1)

    // Jump and fertility Dirichlet prior masses (alpha * support size). Constant
    // within a sweep; cached to avoid recomputation in the per-candidate inner loop.
    // (lexPriorMass is not stored here because the NULL word uses a different alpha.)
    double jumpPriorMass = 0;
    double fertPriorMass = 0;

    // Lexical counts accumulated over final-stage sweeps (variance reduction);
    // jump and fertility tables are read off the final alignment only.
    std::vector<LexCountMap> accumLexCounts;
    std::vector<double> accumLexCountSum;
    bool accumulated = false;

    // Trained parameter tables, populated by normalizeChain in endTraining
    // and by load; they persist after clearTempVars.
    std::shared_ptr<MemoryLexTable> lexTable;
    std::shared_ptr<EflomalJumpTable> jumpTable;
    std::shared_ptr<FertilityTable> fertilityTable;
  };

  const double SmoothingProb = 1e-9;
  const double SmoothingLogProb = std::log(SmoothingProb);
  const PositionIndex MaxFertility = 7;
  const unsigned int DefaultSeed = 1351155463u;
  const int DefaultJumpWindow = 100;
  const double DefaultAlphaLex = 0.001;
  const double DefaultAlphaNull = 0.001;
  // WPT 300k factorial sweep: aj+af=2.0 gives best 1-sampler AER (7.03% vs
  // 7.62% baseline). These two hyperparameters interact synergistically; their
  // combination outperforms any other pair or triple of the four tested.
  const double DefaultAlphaJump = 2.0;
  const double DefaultAlphaFertility = 2.0;
  const double DefaultNullProb = 0.2;
  const int DefaultIbm1Iters = 4;
  const int DefaultHmmIters = 4;
  const int DefaultFertilityIters = 4;
  // Decoding mirrors eflomal's final argmax pass: a few short sampler chains
  // whose per-position marginals are averaged, then argmaxed per target token.
  // Ablation on WPT French (300k) showed (8, 20, 4) reduces 1-sampler AER by
  // ~0.17% vs the old (4, 12, 4) default at negligible extra alignment cost.
  const int DefaultDecodeSamplers = 8;
  const int DefaultDecodeIters = 20;
  const int DefaultDecodeBurnIn = 4;

  std::string getModelTypeStr() const override
  {
    return "eflomal";
  }

  // Index helpers.
  std::vector<WordIndex> getSrcSent(unsigned int n);
  std::vector<WordIndex> getTrgSent(unsigned int n);

  // Stage scheduling for the externally-driven train() loop.
  Stage stageForIter(int it) const;

  // Corpus builder (fills corpusSrc / corpusTrg; does not touch chains).
  void buildCorpus();

  // Chain-level Gibbs sampler operations. Each takes a chain reference so
  // they can run in parallel (no shared mutable state with other chains).
  void initializeChain(SamplerChain& chain);
  void computeJumpCounts(SamplerChain& chain);
  void computeFertilityCounts(SamplerChain& chain);
  // Draws a categorical fertility distribution per source word from its Dirichlet
  // posterior (matching eflomal), storing the ratio P(phi)/P(phi-1) used by the
  // sampler to score incrementing a word's fertility.
  void sampleFertilityRatios(SamplerChain& chain);
  void sampleSweep(SamplerChain& chain, Stage stage, bool accumulate);
  int jumpBucket(int offset) const;
  // Normalizes this chain's accumulated counts into its own lex/jump/fert tables.
  void normalizeChain(SamplerChain& chain);

  // Decoding / scoring shared by getBestAlignment and computeSumLogProb.
  // eflomal extracts alignments as the per-target marginal mode of the Gibbs
  // posterior; sampleDecode mirrors that by sampling against each chain's
  // trained distributions, summing per-position marginals, then argmaxing.
  void sampleDecode(const std::vector<WordIndex>& nsrc, const std::vector<WordIndex>& trg,
                    std::vector<PositionIndex>& alignment);
  // Runs the decode sampler against explicit tables with a fixed seed. Thread-
  // safe: all state is local (no writes to shared fields). chainSeed seeds the
  // local RNG so each chain's decode is reproducible and independent.
  void decodeMarginalFromTables(const MemoryLexTable& lex, const EflomalJumpTable& jump,
                                 const FertilityTable& fert, unsigned int chainSeed,
                                 const std::vector<WordIndex>& nsrc, const std::vector<WordIndex>& trg,
                                 std::vector<std::vector<double>>& acc);
  double transitionLogProb(PositionIndex prev, PositionIndex i, PositionIndex slen) const;
  double scoreAlignment(const std::vector<WordIndex>& nsrc, const std::vector<WordIndex>& trg,
                        const std::vector<PositionIndex>& alignment);

  void loadConfig(const YAML::Node& config) override;
  void createConfig(YAML::Emitter& out) override;
  bool loadParams(const std::string& filename);
  bool printParams(const std::string& filename) const;

  // Hyperparameters / configuration.
  unsigned int seed = DefaultSeed;
  int numSamplers = 1;
  bool deterministic = true;
  int ibm1Iters = DefaultIbm1Iters;
  int hmmIters = DefaultHmmIters;
  int fertilityIters = DefaultFertilityIters;
  int jumpWindow = DefaultJumpWindow;
  bool eflomalLexNorm = true;
  bool autoIterations = false;
  int decodeSamplers = DefaultDecodeSamplers;
  int decodeIters = DefaultDecodeIters;
  int decodeBurnIn = DefaultDecodeBurnIn;
  double alphaLex = DefaultAlphaLex;
  double alphaNull = DefaultAlphaNull;
  double alphaJump = DefaultAlphaJump;
  double alphaFertility = DefaultAlphaFertility;
  double nullProb = DefaultNullProb;
  int iter = 0;

  // Model-level query tables: always point to chains[0]'s tables so that
  // translationLogProb, jumpLogProb, fertilityLogProb, and scoreAlignment
  // work without knowing about the chain structure.
  std::shared_ptr<MemoryLexTable> lexTable;
  std::shared_ptr<EflomalJumpTable> jumpTable;
  std::shared_ptr<FertilityTable> fertilityTable;
  double totLenRatio = 0;

  // Shared corpus (read-only during training; cleared after endTraining).
  std::vector<std::vector<WordIndex>> corpusSrc; // null-extended (index 0 == NULL_WORD)
  std::vector<std::vector<WordIndex>> corpusTrg;

  // Per-chain training state and trained tables. chains.size() == numSamplers
  // during and after training. chains[0].lexTable == this->lexTable always.
  std::vector<SamplerChain> chains;
};
