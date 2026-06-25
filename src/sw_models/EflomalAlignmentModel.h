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
#include <utility>
#include <vector>

// Implements the eflomal word-alignment algorithm of Oestling & Tiedemann (2016),
// "Efficient Word Alignment with Markov Chain Monte Carlo": a Bayesian
// IBM1 -> HMM(jump) -> fertility cascade trained by collapsed Gibbs sampling with
// Dirichlet priors. Training samples alignments over the corpus, but the trained
// model persists normalized lexical, jump and fertility distributions and decodes
// via marginal-mode sampling, so it satisfies thot's queryable/serializable
// AlignmentModel interface for arbitrary (including held-out) sentence pairs.
// Batch-only: does not implement IncrAlignmentModel.
class EflomalAlignmentModel : public AlignmentModelBase
{
public:
  EflomalAlignmentModel();

  AlignmentModelType getModelType() const override
  {
    return Eflomal;
  }

  void setSeed(unsigned int s);
  unsigned int getSeed() const;
  // Number of independent Gibbs chains trained in parallel; their decode marginals
  // are summed at alignment time. Each chain is seeded with seed + s*2654435761.
  void setNumSamplers(int value);
  int getNumSamplers() const;
  void setDeterministic(bool value);
  bool getDeterministic() const;
  void setIterations(int ibm1, int hmm, int fertility);
  int getIbm1Iters() const;
  int getHmmIters() const;
  int getFertilityIters() const;
  void setP0(double value);
  double getP0() const;
  // Dirichlet prior masses (lexical / jump / fertility) and the jump-distribution
  // half-window; exposed so callers can tune them.
  void setAlphaLex(double value);
  double getAlphaLex() const;
  void setAlphaJump(double value);
  double getAlphaJump() const;
  void setAlphaFertility(double value);
  double getAlphaFertility() const;
  void setJumpWindow(int value);
  int getJumpWindow() const;

  // - eflomalLexNorm (on): use the lexical denominator 1/N(e) instead of the
  //   Dirichlet-smoothed 1/(N(e) + alpha*|V|).
  // - autoIterations (on): derive the schedule from the corpus size as
  //   iters = max(2, round(5000/sqrt(N))); the IBM1 and HMM stages get
  //   max(2, iters/4) and iters/4 sweeps, the fertility stage gets the full iters.
  //   setIterations turns this off in favour of the explicit schedule.
  void setEflomalLexNorm(bool value);
  bool getEflomalLexNorm() const;
  void setAutoIterations(bool value);
  bool getAutoIterations() const;
  // Decode-time sampler chains / iterations / burn-in for the marginal-mode decode.
  void setDecodeParams(int samplers, int iters, int burnIn);
  int getDecodeSamplers() const;
  int getDecodeIters() const;
  int getDecodeBurnIn() const;
  // Total number of sweeps the resolved schedule requires (valid after
  // startTraining); the driver should call train() this many times.
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

  // Trained distributions; all query chain[0]'s tables.
  Prob translationProb(WordIndex s, WordIndex t) override;
  LgProb translationLogProb(WordIndex s, WordIndex t) override;
  Prob jumpProb(int offset);
  LgProb jumpLogProb(int offset);
  Prob fertilityProb(WordIndex s, PositionIndex phi);
  LgProb fertilityLogProb(WordIndex s, PositionIndex phi);
  // HMM-style alignment transition probability: aligning a target token to source
  // position i given the previous non-NULL alignment prevI (0 = none / sentence
  // start), in a source sentence of length slen. i == 0 is the NULL alignment.
  // Same argument order as HmmAlignmentModel::hmmAlignmentProb.
  Prob hmmAlignmentProb(PositionIndex prevI, PositionIndex slen, PositionIndex i);
  LgProb hmmAlignmentLogProb(PositionIndex prevI, PositionIndex slen, PositionIndex i);

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
  enum Stage
  {
    Ibm1Stage = 0,
    HmmStage = 1,
    FertilityStage = 2
  };

  // Finalizer-style integer mixer (two multiplies + three xorshifts) for the
  // lexical count maps: robin_map uses power-of-two buckets and keys off the low
  // hash bits, so it needs strong avalanche there, which std::hash for integers
  // does not reliably provide.
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

  // Per-chain Gibbs sampler state. N chains train in parallel; each has its own
  // RNG, alignment, count tables and trained parameter tables. Decode sums
  // marginals across all chains. Chains are independent, so training parallelizes.
  struct SamplerChain
  {
    unsigned int chainSeed = 0;
    std::mt19937_64 rng;

    // Current corpus alignment (training state).
    std::vector<std::vector<PositionIndex>> alig;

    // Running lexical counts (collapsed sampler: decremented before sampling,
    // incremented after, per target token). Open-addressing map with float values
    // (contiguous, no per-node pointer chase): counts are integer-valued (+/-1) and
    // bounded under 2^24, so float holds them exactly. The running sums stay double
    // (they can exceed float's exact-int range and are not read per candidate).
    std::vector<LexCountMap> lexCounts;
    std::vector<double> lexCountSum;

    // Jump counts: seeded per sweep by computeJumpCounts, then updated per token
    // during the collapsed sweep. Fertility counts: recomputed per sweep.
    std::vector<double> jumpCounts;
    double jumpCountSum = 0;
    std::vector<std::vector<double>> fertCounts; // [s][phi]
    std::vector<double> fertCountSum;
    std::vector<std::vector<double>> fertRatioSampled; // [s][phi] = sampled P(phi)/P(phi-1)

    // Dirichlet prior masses (alpha * support size). Constant within a sweep;
    // cached here to avoid recomputation in the per-candidate inner loop.
    double lexPriorMass = 0;
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
  const double DefaultAlphaJump = 0.5;
  const double DefaultAlphaFertility = 0.5;
  const double DefaultP0 = 0.2;
  const int DefaultIbm1Iters = 4;
  const int DefaultHmmIters = 4;
  const int DefaultFertilityIters = 4;
  // Unless setDecodeParams is called, the decode parameters are auto-derived in
  // startTraining: decodeIters tracks the resolved training-schedule total (decode
  // is the same Gibbs sampler over the same posterior), with no burn-in, and
  // decodeSamplers=1 (the numSamplers training chains already supply the marginal
  // ensemble, so the inner decode-sampler loop would be redundant). The constants
  // below are only the pre-training / cleared / non-auto fallback.
  const int DefaultDecodeSamplers = 1;
  const int DefaultDecodeIters = 20;
  const int DefaultDecodeBurnIn = 4;
  // Floor for the auto-derived decodeIters: the corpus-scaled schedule collapses
  // to single digits on very large corpora, but decode still needs enough sweeps
  // to estimate the marginal.
  const int MinDecodeIters = 40;

  std::string getModelTypeStr() const override
  {
    return "eflomal";
  }

  std::vector<WordIndex> getSrcSent(unsigned int n);
  std::vector<WordIndex> getTrgSent(unsigned int n);

  // Stage scheduling for the externally-driven train() loop.
  Stage stageForIter(int it) const;

  // Fills corpusSrc / corpusTrg; does not touch chains.
  void buildCorpus();

  // Chain-level Gibbs sampler operations. Each takes a chain reference so they can
  // run in parallel (no shared mutable state across chains).
  void initializeChain(SamplerChain& chain);
  void computeJumpCounts(SamplerChain& chain);
  void computeFertilityCounts(SamplerChain& chain);
  // Draws a categorical fertility distribution per source word from its Dirichlet
  // posterior, storing the ratio P(phi)/P(phi-1) used by the sampler to score
  // incrementing a word's fertility.
  void sampleFertilityRatios(SamplerChain& chain);
  void sampleSweep(SamplerChain& chain, Stage stage, bool accumulate);
  int jumpBucket(int offset) const;
  // Normalizes this chain's accumulated counts into its own lex/jump/fert tables.
  void normalizeChain(SamplerChain& chain);

  // Decoding / scoring shared by getBestAlignment and computeSumLogProb. Extracts
  // the alignment as the per-target marginal mode of the Gibbs posterior: sample
  // against each chain's trained distributions, sum per-position marginals, argmax.
  void sampleDecode(const std::vector<WordIndex>& nsrc, const std::vector<WordIndex>& trg,
                    std::vector<PositionIndex>& alignment);
  // Sums the warm decode marginal across chains for training-corpus pair n, seeding
  // each chain from its converged alignment. Called by endTraining (before
  // clearTempVars) while the corpus and chain alignments are still resident.
  void accumulateWarmDecodeMarginal(size_t n, std::vector<std::vector<double>>& accOut);

  // Emits the warm final-argmax alignment for every training-corpus pair (one pass
  // seeded from each chain's converged alignment, marginals summed across chains),
  // rather than recomputing via getBestAlignment.
  void computeTrainingAlignments() override;

  // warmStart (optional): if non-null, seed the decode alignment from it and run a
  // single accumulate pass with no burn-in (one decode chain) instead of the cold
  // diagonal-init multi-iteration re-sample.
  void decodeMarginalFromTables(const MemoryLexTable& lex, const EflomalJumpTable& jump,
                                 const FertilityTable& fert, unsigned int chainSeed,
                                 const std::vector<WordIndex>& nsrc, const std::vector<WordIndex>& trg,
                                 std::vector<std::vector<double>>& acc,
                                 const std::vector<PositionIndex>* warmStart = nullptr);
  double scoreAlignment(const std::vector<WordIndex>& nsrc, const std::vector<WordIndex>& trg,
                        const std::vector<PositionIndex>& alignment);

  void loadConfig(const YAML::Node& config) override;
  void createConfig(YAML::Emitter& out) override;
  bool loadParams(const std::string& filename);
  bool printParams(const std::string& filename) const;

  // Hyperparameters / configuration.
  unsigned int seed = DefaultSeed;
  int numSamplers = 3;
  bool deterministic = false;
  int ibm1Iters = DefaultIbm1Iters;
  int hmmIters = DefaultHmmIters;
  int fertilityIters = DefaultFertilityIters;
  int jumpWindow = DefaultJumpWindow;
  bool eflomalLexNorm = true;
  bool autoIterations = true;
  int decodeSamplers = DefaultDecodeSamplers;
  int decodeIters = DefaultDecodeIters;
  int decodeBurnIn = DefaultDecodeBurnIn;
  // True once decode params are pinned by the user (setDecodeParams) or restored
  // from a serialized config; suppresses the train-time auto-derivation.
  bool decodeParamsExplicit = false;
  double alphaLex = DefaultAlphaLex;
  double alphaJump = DefaultAlphaJump;
  double alphaFertility = DefaultAlphaFertility;
  double p0 = DefaultP0;
  int iter = 0;

  // Model-level query tables: always point to chains[0]'s tables so the public
  // probability queries and scoreAlignment work without knowing the chain layout.
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
