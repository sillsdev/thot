#include "sw_models/EflomalAlignmentModel.h"

#ifdef THOT_BENCH_MIMALLOC
// Diagnostic-only (Phase 1 opt 1, see eflomal-optimizations.md): overrides global
// operator new/delete to route through mimalloc, to test whether the default
// allocator is a scaling bottleneck. Only compiled in when configured with
// -DTHOT_BENCH_MIMALLOC=ON; no effect on the normal build.
#include <mimalloc-new-delete.h>
#endif

#include "TestUtils.h"
#include "nlp_common/MathDefs.h"
#include "sw_models/FastAlignModel.h"
#include "sw_models/HmmAlignmentModel.h"
#include "sw_models/Ibm1AlignmentModel.h"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <future>
#include <gtest/gtest.h>
#include <iomanip>
#include <memory>
#include <omp.h>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

namespace
{
constexpr double kEpsilon = 1e-6;

void trainEflomal(EflomalAlignmentModel& model)
{
  // Pin to a single deterministic chain and an explicit tiny schedule so the golden
  // alignments below are stable regardless of the shipped defaults (setIterations
  // also pins the schedule, overriding the default-on autoIterations).
  model.setNumSamplers(1);
  model.setDeterministic(true);
  model.setIterations(2, 2, 2);
  model.setDecodeParams(4, 12, 4);
  addTrainingData(model);
  train(model, 6);
}
} // namespace

TEST(EflomalAlignmentModelTest, trainEmpty)
{
  EflomalAlignmentModel model;
  model.setIterations(2, 2, 2);
  EXPECT_NO_THROW(train(model, 6));
}

TEST(EflomalAlignmentModelTest, train)
{
  EflomalAlignmentModel model;
  trainEflomal(model);

  std::vector<PositionIndex> alignment;
  model.getBestAlignment("isthay isyay ayay esttay-N .", "this is a test N .", alignment);
  EXPECT_EQ(alignment, (std::vector<PositionIndex>{1, 2, 3, 4, 0, 5}));

  model.getBestAlignment("isthay isyay otnay ayay esttay-N .", "this is not a test N .", alignment);
  EXPECT_EQ(alignment, (std::vector<PositionIndex>{1, 2, 3, 4, 5, 0, 6}));

  model.getBestAlignment("isthay isyay ayay esttay-N ardhay .", "this is a hard test N .", alignment);
  EXPECT_EQ(alignment, (std::vector<PositionIndex>{1, 2, 3, 5, 4, 0, 6}));
}

// initializeChain runs under `#pragma omp parallel for ... if(!deterministic)`
// (EflomalAlignmentModel.cc, in startTraining): each chain only ever reads the
// shared read-only corpus and writes its own SamplerChain slot, seeded purely from
// its chain index (seed + s*2654435761), so concurrent init should be exactly as
// reproducible as the serial version it replaced. Trains the same corpus twice with
// numSamplers>1 and deterministic=false (the mode that actually exercises the
// parallel path) and checks the two runs agree, which a data race or
// order-dependency bug in the parallelized init would be very likely to break.
TEST(EflomalAlignmentModelTest, initializeChainParallelIsReproducible)
{
  auto trainAndAlign = [] {
    EflomalAlignmentModel model;
    model.setNumSamplers(4);
    model.setDeterministic(false);
    model.setIterations(2, 2, 2);
    addTrainingData(model);
    train(model, 6);
    std::vector<PositionIndex> a1, a2, a3;
    model.getBestAlignment("isthay isyay ayay esttay-N .", "this is a test N .", a1);
    model.getBestAlignment("isthay isyay otnay ayay esttay-N .", "this is not a test N .", a2);
    model.getBestAlignment("isthay isyay ayay esttay-N ardhay .", "this is a hard test N .", a3);
    return std::make_tuple(a1, a2, a3);
  };

  auto run1 = trainAndAlign();
  auto run2 = trainAndAlign();
  EXPECT_EQ(run1, run2);
}

// Regression coverage for a code-review finding: an earlier attempt at optimizing
// sampleSweep's per-candidate jumpBucket() clamp into a running saturating counter
// (bucket = min(bucket+1,cap) / max(bucket-1,0)) was wrong whenever the counter started
// already pinned to a boundary at i==1 - it would immediately start drifting away from
// the pin instead of staying there until the true unclamped offset re-entered range.
// The default jumpWindow=100 combined with this file's <10-token test sentences never
// reaches that boundary, so the whole existing suite passed anyway; only a jumpWindow
// smaller than the sentence length forces every candidate past the boundary immediately.
// Pins the alignment from the correct per-candidate jumpBucket() computation; a
// reintroduction of an incorrectly-saturating running counter would change these values
// without crashing or failing any other test.
// Trains the same jumpWindow=2 + long-sentence scenario (see below) and returns the
// long sentence's decoded alignment; factored out so the test can both check it and
// re-run it for a same-platform determinism cross-check.
namespace
{
std::vector<PositionIndex> trainSaturationScenarioAndAlignLongSentence()
{
  EflomalAlignmentModel model;
  model.setNumSamplers(1);
  model.setDeterministic(true);
  model.setIterations(2, 2, 2);
  model.setDecodeParams(4, 12, 4);
  model.setJumpWindow(2); // shorter than the long sentence below: forces saturation
  addTrainingData(model);
  addSentencePair(model, "wanyay owtay eethray ourfay ivefay ixsay evensay eightyay ninenay tenyay elevenyay elvetway irteenthay ourteenfay",
                  "one two three four five six seven eight nine ten eleven twelve thirteen fourteen");
  train(model, 6);

  std::vector<PositionIndex> alignment;
  model.getBestAlignment(
      "wanyay owtay eethray ourfay ivefay ixsay evensay eightyay ninenay tenyay elevenyay elvetway irteenthay ourteenfay",
      "one two three four five six seven eight nine ten eleven twelve thirteen fourteen", alignment);
  return alignment;
}
} // namespace

TEST(EflomalAlignmentModelTest, jumpBucketSaturationBoundary)
{
  EflomalAlignmentModel model;
  model.setNumSamplers(1);
  model.setDeterministic(true);
  model.setIterations(2, 2, 2);
  model.setDecodeParams(4, 12, 4);
  model.setJumpWindow(2); // shorter than every training sentence: forces saturation
  addTrainingData(model);
  // A longer, purpose-built pair on top of addTrainingData's toy corpus: long enough
  // (14 tokens) relative to jumpWindow=2 that mid-sentence candidates sit far past the
  // saturation boundary for many consecutive resamples - the regime a since-reverted
  // running-counter optimization got wrong (see eflomal-optimizations.md, "Opt B",
  // and fable-improvements.md item 10's follow-up attempt).
  addSentencePair(model, "wanyay owtay eethray ourfay ivefay ixsay evensay eightyay ninenay tenyay elevenyay elvetway irteenthay ourteenfay",
                  "one two three four five six seven eight nine ten eleven twelve thirteen fourteen");
  train(model, 6);

  // These three short (<10 token) sentences are stable pinned-value checks: verified
  // identical across Windows/MSVC, Linux/glibc, and macOS/libc++ CI runs, because
  // jumpWindow=2 doesn't push them far enough into the saturation regime to make the
  // few sampling decisions involved sensitive to platform libm/RNG differences.
  std::vector<PositionIndex> alignment;
  model.getBestAlignment("isthay isyay ayay esttay-N .", "this is a test N .", alignment);
  EXPECT_EQ(alignment, (std::vector<PositionIndex>{1, 2, 3, 0, 4, 5}));

  model.getBestAlignment("isthay isyay otnay ayay esttay-N .", "this is not a test N .", alignment);
  EXPECT_EQ(alignment, (std::vector<PositionIndex>{1, 2, 3, 4, 0, 5, 6}));

  model.getBestAlignment("isthay isyay ayay esttay-N ardhay .", "this is a hard test N .", alignment);
  EXPECT_EQ(alignment, (std::vector<PositionIndex>{1, 2, 3, 5, 0, 4, 6}));

  // The long (14-token) sentence is NOT pinned to an exact value: with jumpWindow=2 -
  // deliberately small relative to sentence length, to force many resamples deep into
  // the saturation regime this test targets - the 6-sweep collapsed Gibbs chain
  // accumulates enough sampling decisions that tiny, platform-specific differences in
  // libm (exp/log) and std::uniform_real_distribution's implementation-defined
  // mapping compound into a genuinely different (but equally valid, since both are
  // legitimate draws from the same posterior) final alignment on each platform -
  // confirmed empirically: Windows/MSVC, Linux/glibc, and macOS/libc++ CI runs each
  // produced a different, self-consistent alignment for this specific sentence.
  // SamplingUtils.h documents this exact tradeoff ("cross-platform bit-identical
  // results are not required"). What IS portable and still meaningful: the alignment
  // is well-formed (right length, every link a valid NULL-or-1-based source position)
  // and reproducible on a given platform (same input, same output, twice).
  model.getBestAlignment(
      "wanyay owtay eethray ourfay ivefay ixsay evensay eightyay ninenay tenyay elevenyay elvetway irteenthay ourteenfay",
      "one two three four five six seven eight nine ten eleven twelve thirteen fourteen", alignment);
  ASSERT_EQ(alignment.size(), 14u);
  for (PositionIndex link : alignment)
    EXPECT_LE(link, 14u); // 0 = NULL, else a valid 1-based source position

  std::vector<PositionIndex> repeat = trainSaturationScenarioAndAlignLongSentence();
  EXPECT_EQ(alignment, repeat);
}

TEST(EflomalAlignmentModelTest, trainingAlignmentAccessor)
{
  EflomalAlignmentModel model;
  model.setNumSamplers(1);
  model.setDeterministic(true);
  model.setIterations(2, 2, 2);
  model.setEmitTrainingAlignments(true);
  addTrainingData(model); // 8 sentence pairs
  train(model, 6);

  ASSERT_EQ(model.numSentencePairs(), 8u); // one alignment per training pair

  // The single-pair accessor fills the per-target form (like getBestAlignment: one
  // entry per target token) and returns a real log-probability. Pair 0's target
  // "this is a test N ." has 6 tokens. Tally non-NULL links across all pairs.
  size_t aligned = 0;
  for (unsigned int n = 0; n < model.numSentencePairs(); ++n)
  {
    std::vector<PositionIndex> alig;
    LgProb lp = model.getTrainingAlignment(n, alig);
    if (n == 0)
    {
      EXPECT_EQ(alig.size(), 6u);
    }
    EXPECT_GT((double)lp, (double)SMALL_LG_NUM); // a real score, not the sentinel
    EXPECT_LE((double)lp, 0.0);                  // a log-probability
    for (PositionIndex a : alig)
      if (a > 0)
        ++aligned;
  }
  EXPECT_GT(aligned, 0u); // training produced some non-NULL alignments

  // Out-of-range index clears the alignment and returns the sentinel, not a crash.
  std::vector<PositionIndex> oob{1, 2, 3};
  LgProb oobLp = model.getTrainingAlignment(1000, oob);
  EXPECT_TRUE(oob.empty());
  EXPECT_EQ((double)oobLp, (double)SMALL_LG_NUM);
}

TEST(EflomalAlignmentModelTest, computeLogProbMatchesBestAlignment)
{
  EflomalAlignmentModel model;
  trainEflomal(model);

  std::vector<PositionIndex> alignment;
  LgProb expectedLogProb = model.getBestAlignment("isthay isyay ayay esttay-N .", "this is a test N NULL .", alignment);
  WordAlignmentMatrix waMatrix{5, 7};
  waMatrix.putAligVec(alignment);
  LgProb logProb = model.computeLogProb("isthay isyay ayay esttay-N .", "this is a test N NULL .", waMatrix);
  EXPECT_NEAR(logProb, expectedLogProb, kEpsilon);
}

TEST(EflomalAlignmentModelTest, deterministicWithFixedSeed)
{
  EflomalAlignmentModel a;
  EflomalAlignmentModel b;
  trainEflomal(a);
  trainEflomal(b);

  for (const char* probe : {"isthay isyay ayay esttay-N .", "isthay isyay otnay ayay esttay-N ."})
  {
    std::vector<PositionIndex> alignmentA, alignmentB;
    a.getBestAlignment(probe, "this is a test N .", alignmentA);
    b.getBestAlignment(probe, "this is a test N .", alignmentB);
    EXPECT_EQ(alignmentA, alignmentB);
  }
}

TEST(EflomalAlignmentModelTest, serializationRoundTrip)
{
  EflomalAlignmentModel model;
  trainEflomal(model);

  std::string prefix = "eflomal_round_trip_test";
  ASSERT_EQ(model.print(prefix.c_str()), THOT_OK);

  EflomalAlignmentModel loaded;
  ASSERT_EQ(loaded.load(prefix.c_str()), THOT_OK);

  std::vector<PositionIndex> original, restored;
  LgProb originalLogProb = model.getBestAlignment("isthay isyay ayay esttay-N .", "this is a test N .", original);
  LgProb restoredLogProb = loaded.getBestAlignment("isthay isyay ayay esttay-N .", "this is a test N .", restored);
  EXPECT_EQ(original, restored);
  EXPECT_NEAR(restoredLogProb, originalLogProb, kEpsilon);
  EXPECT_NEAR(loaded.translationProb(model.stringToSrcWordIndex("isthay"), model.stringToTrgWordIndex("this")),
              model.translationProb(model.stringToSrcWordIndex("isthay"), model.stringToTrgWordIndex("this")),
              kEpsilon);
}

TEST(EflomalAlignmentModelTest, trainedProbabilitiesAreValid)
{
  EflomalAlignmentModel model;
  trainEflomal(model);

  Prob p = model.translationProb(model.stringToSrcWordIndex("isthay"), model.stringToTrgWordIndex("this"));
  EXPECT_GT((double)p, 0.0);
  EXPECT_LE((double)p, 1.0);

  // The jump distribution must be a proper probability for the zero offset.
  EXPECT_GT((double)model.jumpProb(0), 0.0);
  EXPECT_LE((double)model.jumpProb(0), 1.0);
}

TEST(EflomalAlignmentModelTest, singlePairCorpus)
{
  EflomalAlignmentModel model;
  model.setIterations(2, 2, 2);
  addSentencePair(model, "isthay isyay ayay esttay-N .", "this is a test N .");
  train(model, 6);

  std::vector<PositionIndex> alignment;
  model.getBestAlignment("isthay isyay ayay esttay-N .", "this is a test N .", alignment);
  EXPECT_EQ(alignment.size(), 6u);
}

TEST(EflomalAlignmentModelTest, oovWordsReturnSmoothedProb)
{
  EflomalAlignmentModel model;
  model.setIterations(2, 2, 2);
  addTrainingData(model);
  train(model, 6);

  // Word indices for words not seen in training should return the smoothing prob.
  WordIndex unkSrc = model.addSrcSymbol("UNSEEN_SRC");
  WordIndex unkTrg = model.addTrgSymbol("UNSEEN_TRG");
  Prob p = model.translationProb(unkSrc, unkTrg);
  EXPECT_GT((double)p, 0.0);
  EXPECT_LE((double)p, 1.0);
}

// Regression coverage: decodeMarginalFromTables' fertRatioTable/decodeFertRatio (see
// buildDecodeCache) is sized to the vocab as of the last training/load, but
// getBestAlignment(string, string, ...) grows the vocab via addSrcSymbol for every
// word in the query (strVectorToSrcIndexVector -> addSrcSymbol), including brand-new
// words never seen during training. Confirms a post-training query containing a new
// word doesn't index the flat decode cache out of bounds - it should fall back to the
// same "unseen word" smoothing the old per-call table lookup gave, not crash.
TEST(EflomalAlignmentModelTest, getBestAlignmentWithPostTrainingNewWord)
{
  EflomalAlignmentModel model;
  trainEflomal(model);

  std::vector<PositionIndex> alignment;
  EXPECT_NO_THROW(model.getBestAlignment("brandnewunseenword isyay ayay esttay-N .",
                                         "brandnewunseentarget is a test N .", alignment));
  EXPECT_EQ(alignment.size(), 6u);
  for (PositionIndex p : alignment)
    EXPECT_LE(p, 6u); // every link is either NULL (0) or a valid 1-based source position
}

TEST(EflomalAlignmentModelTest, getEntriesForSource)
{
  EflomalAlignmentModel model;
  model.setIterations(2, 2, 2);
  addTrainingData(model);
  train(model, 6);

  NbestTableNode<WordIndex> entries;
  WordIndex s = model.stringToSrcWordIndex("isthay");
  bool found = model.getEntriesForSource(s, entries);
  EXPECT_TRUE(found);
  EXPECT_GT(entries.size(), 0u);

  WordIndex unkSrc = model.addSrcSymbol("UNSEEN_FOR_ENTRIES");
  found = model.getEntriesForSource(unkSrc, entries);
  EXPECT_FALSE(found);
}

TEST(EflomalAlignmentModelTest, computeSumLogProb)
{
  EflomalAlignmentModel model;
  model.setIterations(2, 2, 2);
  addTrainingData(model);
  train(model, 6);

  std::vector<std::string> src = {"isthay", "isyay", "ayay", "esttay-N", "."};
  std::vector<std::string> trg = {"this", "is", "a", "test", "N", "."};
  LgProb logProb = model.computeSumLogProb(src, trg);
  EXPECT_LT((double)logProb, 0.0);
  EXPECT_GT((double)logProb, -1e6);
}

TEST(EflomalAlignmentModelTest, loglikelihoodForPairRange)
{
  EflomalAlignmentModel model;
  model.setIterations(2, 2, 2);
  addTrainingData(model);
  train(model, 6);

  auto result = model.loglikelihoodForPairRange({0, model.numSentencePairs() - 1});
  EXPECT_LT(result.first, 0.0);
  EXPECT_LT(result.second, 0.0);
  EXPECT_GT(result.second, -1e6);
}

// Regression coverage for loglikelihoodForPairRange's parallel rewrite (fable-
// improvements.md item 11): scoring is now split into a serial getSrcSent/getTrgSent
// pass (which can intern new vocab words) followed by a parallel
// #pragma omp reduction(+:loglikelihood) over computeSumLogProb. Confirms the empty-
// range edge case (last < first) still returns (0,0) as the original unsigned-loop
// version did, and that deterministic=true (which gates the new pragma off) gives
// bit-identical repeated results, which a data race in the parallel path would be
// very likely to break.
TEST(EflomalAlignmentModelTest, loglikelihoodForPairRangeEdgeCasesAndDeterminism)
{
  EflomalAlignmentModel model;
  model.setNumSamplers(4);
  model.setDeterministic(true);
  model.setIterations(2, 2, 2);
  addTrainingData(model);
  train(model, 6);

  // Empty range: second < first.
  auto empty = model.loglikelihoodForPairRange({3, 1});
  EXPECT_EQ(empty.first, 0.0);
  EXPECT_EQ(empty.second, 0.0);

  auto run1 = model.loglikelihoodForPairRange({0, model.numSentencePairs() - 1});
  auto run2 = model.loglikelihoodForPairRange({0, model.numSentencePairs() - 1});
  EXPECT_EQ(run1.first, run2.first);
  EXPECT_EQ(run1.second, run2.second);
}

TEST(EflomalAlignmentModelTest, fertilityProbIsValid)
{
  EflomalAlignmentModel model;
  model.setIterations(2, 2, 2);
  addTrainingData(model);
  train(model, 6);

  WordIndex s = model.stringToSrcWordIndex("isthay");
  for (PositionIndex phi = 0; phi <= 3; ++phi)
  {
    Prob p = model.fertilityProb(s, phi);
    EXPECT_GE((double)p, 0.0);
    EXPECT_LE((double)p, 1.0);
  }
}

TEST(EflomalAlignmentModelTest, clearResetsState)
{
  EflomalAlignmentModel model;
  trainEflomal(model);

  WordIndex s = model.stringToSrcWordIndex("isthay");
  WordIndex t = model.stringToTrgWordIndex("this");
  EXPECT_GT((double)model.translationProb(s, t), 0.01);

  // clear() empties the lex table and vocabulary; the model can be retrained.
  model.clear();
  trainEflomal(model);

  WordIndex s2 = model.stringToSrcWordIndex("isthay");
  WordIndex t2 = model.stringToTrgWordIndex("this");
  EXPECT_GT((double)model.translationProb(s2, t2), 0.01);
}

TEST(EflomalAlignmentModelTest, configNonDefaultRoundTrip)
{
  EflomalAlignmentModel model;
  model.setSeed(12345u);
  model.setDeterministic(false);
  model.setIterations(3, 5, 7);
  model.setP0(0.15);
  model.setEflomalLexNorm(false);
  model.setAutoIterations(false);
  model.setDecodeParams(2, 8, 3);
  addTrainingData(model);
  train(model, 15);

  std::string prefix = "eflomal_config_roundtrip_test";
  ASSERT_EQ(model.print(prefix.c_str()), THOT_OK);

  EflomalAlignmentModel loaded;
  ASSERT_EQ(loaded.load(prefix.c_str()), THOT_OK);

  EXPECT_EQ(loaded.getSeed(), 12345u);
  EXPECT_FALSE(loaded.getDeterministic());
  EXPECT_NEAR(loaded.getP0(), 0.15, kEpsilon);
  EXPECT_FALSE(loaded.getEflomalLexNorm());
  EXPECT_FALSE(loaded.getAutoIterations());
}

TEST(EflomalAlignmentModelTest, decodeBurnInClampedWhenExceedsIters)
{
  EflomalAlignmentModel model;
  model.setIterations(2, 2, 2);
  // Set burn-in >= decodeIters: without the clamp, acc would stay all-zero
  // and every target would map to NULL.
  model.setDecodeParams(2, 4, 10);
  addTrainingData(model);
  train(model, 6);

  std::vector<PositionIndex> alignment;
  model.getBestAlignment("isthay isyay ayay esttay-N .", "this is a test N .", alignment);
  // At least some tokens should not be NULL-aligned (position 0).
  int nonNull = 0;
  for (PositionIndex a : alignment)
    if (a > 0)
      ++nonNull;
  EXPECT_GT(nonNull, 0);
}

TEST(EflomalAlignmentModelTest, accumulateDecodeMarginalAveragesModels)
{
  // Two identically-seeded models should average to the same result as one.
  EflomalAlignmentModel a, b;
  trainEflomal(a);
  trainEflomal(b);

  auto src = a.strVectorToSrcIndexVector({"isthay", "isyay", "ayay", "esttay-N", "."});
  auto trg = a.strVectorToTrgIndexVector({"this", "is", "a", "test", "N", "."});

  std::vector<std::vector<double>> acc;
  a.accumulateDecodeMarginal(src, trg, acc);
  b.accumulateDecodeMarginal(src, trg, acc);

  // acc is non-empty and has one row per target token.
  EXPECT_EQ(acc.size(), trg.size());
  // Each row has one entry per source position (slen+1 including NULL).
  EXPECT_EQ(acc[0].size(), src.size() + 1);
  // Each row has positive total weight.
  for (const auto& row : acc)
  {
    double total = 0;
    for (double v : row)
      total += v;
    EXPECT_GT(total, 0.0);
  }
}

// Timing harness for the optimization phase. Disabled by default; run with
// --gtest_also_run_disabled_tests --gtest_filter=*benchmark*.
TEST(EflomalAlignmentModelTest, DISABLED_benchmark)
{
  const int numSentences = 4000;
  const int vocab = 400;
  EflomalAlignmentModel model;
  model.setIterations(4, 4, 4);
  for (int n = 0; n < numSentences; ++n)
  {
    std::vector<std::string> src, trg;
    int len = 8 + (n % 12);
    for (int k = 0; k < len; ++k)
    {
      int w = (n * 7 + k * 13) % vocab;
      src.push_back("s" + std::to_string(w));
      trg.push_back("t" + std::to_string((w + (k % 3)) % vocab));
    }
    model.addSentencePair(src, trg, 1);
  }

  auto start = std::chrono::steady_clock::now();
  train(model, 12);
  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);
  std::cerr << "BENCHMARK: trained " << numSentences << " pairs x 12 sweeps in " << elapsed.count() << " ms\n";
  SUCCEED();
}

namespace
{
std::vector<std::vector<std::string>> readTokenizedCorpus(const std::string& path)
{
  std::vector<std::vector<std::string>> out;
  std::ifstream in(path);
  std::string line;
  while (std::getline(in, line))
  {
    std::vector<std::string> toks;
    std::istringstream iss(line);
    std::string t;
    while (iss >> t)
      toks.push_back(t);
    out.push_back(toks);
  }
  return out;
}

int envInt(const char* key, int dflt)
{
  const char* v = std::getenv(key);
  return v ? std::atoi(v) : dflt;
}
} // namespace

// Real-data ablation harness for matching eflomal. Configured entirely through
// environment variables so different flag combinations can be evaluated from a
// single build. Trains EFL_SAMPLERS independently-seeded models, averages their
// decode marginals (eflomal's n_samplers combination), and writes forward Pharaoh
// alignments. Disabled by default; run with:
//   --gtest_also_run_disabled_tests --gtest_filter=*realAblation*
TEST(EflomalAlignmentModelTest, DISABLED_realAblation)
{
  const char* srcEnv = std::getenv("EFL_SRC");
  const char* trgEnv = std::getenv("EFL_TRG");
  const char* outEnv = std::getenv("EFL_OUT");
  ASSERT_TRUE(srcEnv && trgEnv && outEnv) << "EFL_SRC/EFL_TRG/EFL_OUT must be set";
  std::string modelType = std::getenv("EFL_MODEL") ? std::getenv("EFL_MODEL") : "eflomal";
  const char* trainSrcEnv = std::getenv("EFL_TRAIN_SRC");
  const char* trainTrgEnv = std::getenv("EFL_TRAIN_TGT");

  int samplers = std::max(1, envInt("EFL_SAMPLERS", 1));
  bool lexNorm = envInt("EFL_LEXNORM", 1) != 0;
  bool autoIters = envInt("EFL_AUTOITERS", 0) != 0;
  int ibm1 = envInt("EFL_IBM1", 4), hmm = envInt("EFL_HMM", 4), fert = envInt("EFL_FERT", 4);

  // Test pairs to align.
  std::vector<std::vector<std::string>> testSrc = readTokenizedCorpus(srcEnv);
  std::vector<std::vector<std::string>> testTrg = readTokenizedCorpus(trgEnv);
  ASSERT_EQ(testSrc.size(), testTrg.size());
  size_t testCount = testSrc.size();

  // Combined corpus = optional training + test, so the (transductive) models see the
  // test pairs during training, as is standard for these aligners.
  std::vector<std::vector<std::string>> allSrc, allTrg;
  if (trainSrcEnv && trainTrgEnv)
  {
    allSrc = readTokenizedCorpus(trainSrcEnv);
    allTrg = readTokenizedCorpus(trainTrgEnv);
    ASSERT_EQ(allSrc.size(), allTrg.size());
  }
  for (size_t i = 0; i < testCount; ++i)
  {
    allSrc.push_back(testSrc[i]);
    allTrg.push_back(testTrg[i]);
  }
  size_t nAll = allSrc.size();

  std::ofstream out(outEnv);
  long links = 0;
  int sweeps = 0;
  long trainMs = 0;
  auto t0 = std::chrono::steady_clock::now();

  // Writes the forward Pharaoh alignment for one test pair: alig[j] is the 1-based
  // source position aligned to target j (0 = NULL, omitted).
  auto writeAlignment = [&](const std::vector<PositionIndex>& alig) {
    std::string lineOut;
    for (size_t j = 0; j < alig.size(); ++j)
      if (alig[j] > 0)
      {
        if (!lineOut.empty())
          lineOut += " ";
        lineOut += std::to_string(alig[j] - 1) + "-" + std::to_string(j);
        ++links;
      }
    out << lineOut << "\n";
  };

  if (modelType == "hmm")
  {
    int ibm1Iters = envInt("EFL_IBM1", 5), hmmIters = envInt("EFL_HMM", 5);
    Ibm1AlignmentModel ibm1m;
    for (size_t i = 0; i < nAll; ++i)
      ibm1m.addSentencePair(allSrc[i], allTrg[i], 1);
    ibm1m.startTraining();
    for (int it = 0; it < ibm1Iters; ++it)
      ibm1m.train();
    ibm1m.endTraining();
    HmmAlignmentModel hmmm(ibm1m);
    hmmm.startTraining();
    for (int it = 0; it < hmmIters; ++it)
      hmmm.train();
    hmmm.endTraining();
    sweeps = ibm1Iters + hmmIters;
    trainMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
    for (size_t i = 0; i < testCount; ++i)
    {
      std::vector<PositionIndex> alig;
      hmmm.getBestAlignment(hmmm.strVectorToSrcIndexVector(testSrc[i]), hmmm.strVectorToTrgIndexVector(testTrg[i]),
                            alig);
      writeAlignment(alig);
    }
  }
  else
  {
    // The model runs N independent Gibbs chains in parallel internally.
    // Each chain s uses seed + s*2654435761 (matching the old per-model seeding).
    std::unique_ptr<EflomalAlignmentModel> model(new EflomalAlignmentModel());
    model->setNumSamplers(samplers);
    model->setEflomalLexNorm(lexNorm);
    model->setAutoIterations(autoIters);
    model->setDecodeParams(envInt("EFL_DSAMP", 4), envInt("EFL_DITERS", 12), envInt("EFL_DBURN", 4));
    if (!autoIters)
      model->setIterations(ibm1, hmm, fert);
    for (size_t i = 0; i < nAll; ++i)
      model->addSentencePair(allSrc[i], allTrg[i], 1);
    model->startTraining();
    sweeps = model->getScheduledIterations();
    for (int it = 0; it < sweeps; ++it)
      model->train();
    model->endTraining();
    trainMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();

    // Pre-convert test sentences to index vectors sequentially; strVectorToSrcIndexVector
    // writes to the vocabulary (OOV words) so must not run concurrently on the same model.
    std::vector<std::vector<WordIndex>> testSrcIdx(testCount), testTrgIdx(testCount);
    for (size_t i = 0; i < testCount; ++i)
    {
      testSrcIdx[i] = model->strVectorToSrcIndexVector(testSrc[i]);
      testTrgIdx[i] = model->strVectorToTrgIndexVector(testTrg[i]);
    }

    // accumulateDecodeMarginal now loops over all chains internally; one call
    // per sentence gives the full N-chain accumulated marginal. Thread-safe:
    // each call only reads trained tables and uses local RNG state.
    std::vector<std::vector<PositionIndex>> allAligs(testCount);
#pragma omp parallel for schedule(dynamic, 16)
    for (int i = 0; i < (int)testCount; ++i)
    {
      PositionIndex slen = (PositionIndex)testSrc[i].size();
      std::vector<std::vector<double>> acc;
      model->accumulateDecodeMarginal(testSrcIdx[i], testTrgIdx[i], acc);
      allAligs[i].assign(testTrg[i].size(), 0);
      for (size_t j = 0; j < acc.size(); ++j)
      {
        PositionIndex bestK = 0;
        double bestP = acc[j][0];
        for (PositionIndex k = 1; k <= slen; ++k)
          if (acc[j][k] > bestP)
          {
            bestP = acc[j][k];
            bestK = k;
          }
        allAligs[i][j] = bestK == slen ? 0 : bestK + 1;
      }
    }
    for (size_t i = 0; i < testCount; ++i)
      writeAlignment(allAligs[i]);
  }
  auto alignMs =
      std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count() - trainMs;

  std::cerr << "ABLATION model=" << modelType << " samplers=" << samplers << " lexNorm=" << lexNorm
            << " autoIters=" << autoIters << " sweeps=" << sweeps << " | train " << trainMs << " ms, align " << alignMs
            << " ms, " << links << " links over " << testCount << " test pairs\n";
  SUCCEED();
}

// Decode hyperparameter tuning harness. Trains ONCE on a corpus (transductively,
// as these aligners are used) with the auto-iteration schedule and EFL_SAMPLERS
// training chains, then sweeps decode (samplers, iters, burnIn) configurations
// against a Pharaoh-format gold alignment, reporting forward AER per config.
// Decode is deterministic given the trained model, so the sweep is noise-free.
// Sure-only gold => AER = 1 - 2|A^G| / (|A|+|G|). Disabled by default; run with
// EFL_SRC / EFL_TRG / EFL_GOLD set and
//   --gtest_also_run_disabled_tests --gtest_filter=*decodeTune*
TEST(EflomalAlignmentModelTest, DISABLED_decodeTune)
{
  const char* srcEnv = std::getenv("EFL_SRC");
  const char* trgEnv = std::getenv("EFL_TRG");
  const char* goldEnv = std::getenv("EFL_GOLD");
  ASSERT_TRUE(srcEnv && trgEnv && goldEnv) << "EFL_SRC/EFL_TRG/EFL_GOLD must be set";

  std::vector<std::vector<std::string>> src = readTokenizedCorpus(srcEnv);
  std::vector<std::vector<std::string>> trg = readTokenizedCorpus(trgEnv);
  ASSERT_EQ(src.size(), trg.size());
  size_t n = src.size();

  // Gold links: set of 0-based (src,trg) pairs per line.
  std::vector<std::set<std::pair<int, int>>> gold(n);
  long goldTotal = 0;
  {
    std::ifstream gin(goldEnv);
    std::string line;
    size_t li = 0;
    while (li < n && std::getline(gin, line))
    {
      std::istringstream iss(line);
      std::string tok;
      while (iss >> tok)
      {
        size_t dash = tok.find('-');
        if (dash != std::string::npos)
          gold[li].insert({std::atoi(tok.substr(0, dash).c_str()), std::atoi(tok.substr(dash + 1).c_str())});
      }
      goldTotal += (long)gold[li].size();
      ++li;
    }
    ASSERT_EQ(li, n) << "gold line count != corpus line count";
  }

  std::unique_ptr<EflomalAlignmentModel> model(new EflomalAlignmentModel());
  model->setNumSamplers(std::max(1, envInt("EFL_SAMPLERS", 3)));
  model->setEflomalLexNorm(envInt("EFL_LEXNORM", 1) != 0);
  model->setAutoIterations(envInt("EFL_AUTOITERS", 1) != 0);
  model->setDeterministic(false); // independent chains => parallel is reproducible and faster
  for (size_t i = 0; i < n; ++i)
    model->addSentencePair(src[i], trg[i], 1);

  auto t0 = std::chrono::steady_clock::now();
  model->startTraining();
  int sweeps = model->getScheduledIterations();
  for (int it = 0; it < sweeps; ++it)
    model->train();
  model->endTraining();
  long trainMs =
      std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();

  // Pre-convert to index vectors sequentially (writes vocab for OOV words).
  std::vector<std::vector<WordIndex>> srcIdx(n), trgIdx(n);
  for (size_t i = 0; i < n; ++i)
  {
    srcIdx[i] = model->strVectorToSrcIndexVector(src[i]);
    trgIdx[i] = model->strVectorToTrgIndexVector(trg[i]);
  }
  std::cerr << "TUNE_TRAIN sweeps=" << sweeps << " train " << trainMs << " ms over " << n << " pairs, "
            << goldTotal << " gold links\n";

  // Grids are overridable via comma-separated env (EFL_TUNE_SAMPLERS / EFL_TUNE_ITERS)
  // so the ranges can be probed without rebuilding.
  auto parseGrid = [](const char* key, std::vector<int> dflt) {
    const char* v = std::getenv(key);
    if (!v)
      return dflt;
    std::vector<int> out;
    std::stringstream ss(v);
    std::string item;
    while (std::getline(ss, item, ','))
      if (!item.empty())
        out.push_back(std::atoi(item.c_str()));
    return out.empty() ? dflt : out;
  };
  std::vector<int> samplersGrid = parseGrid("EFL_TUNE_SAMPLERS", {1, 2, 4, 8});
  std::vector<int> itersGrid = parseGrid("EFL_TUNE_ITERS", {5, 10, 20, 40, 80});
  for (int dsamp : samplersGrid)
  {
    for (int diters : itersGrid)
    {
      for (int dburn : {0, diters / 4, diters / 2})
      {
        model->setDecodeParams(dsamp, diters, dburn);
        auto d0 = std::chrono::steady_clock::now();
        long hypTotal = 0, inter = 0;
#pragma omp parallel for schedule(dynamic, 16) reduction(+ : hypTotal, inter)
        for (int i = 0; i < (int)n; ++i)
        {
          PositionIndex slen = (PositionIndex)src[i].size();
          std::vector<std::vector<double>> acc;
          model->accumulateDecodeMarginal(srcIdx[i], trgIdx[i], acc);
          for (size_t j = 0; j < acc.size(); ++j)
          {
            PositionIndex bestK = 0;
            double bestP = acc[j][0];
            for (PositionIndex k = 1; k <= slen; ++k)
              if (acc[j][k] > bestP)
              {
                bestP = acc[j][k];
                bestK = k;
              }
            if (bestK != slen) // not NULL; 0-based source token == bestK
            {
              ++hypTotal;
              if (gold[i].count({(int)bestK, (int)j}))
                ++inter;
            }
          }
        }
        long decodeMs =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - d0).count();
        double aer = (hypTotal + goldTotal) ? 1.0 - (2.0 * inter) / (hypTotal + goldTotal) : 0.0;
        std::cerr << "TUNE samplers=" << dsamp << " iters=" << diters << " burnin=" << dburn << " AER="
                  << std::fixed << std::setprecision(4) << aer << " decode " << decodeMs << " ms\n";
      }
    }
  }
  SUCCEED();
}

// Validates the train-tied decode defaults: trains once (auto schedule, EFL_SAMPLERS
// chains) and reports forward AER for the auto-tied decode params (set inside
// startTraining since they are not pinned here) vs the old fixed (8,20,4) default.
// Disabled by default; run with EFL_SRC/EFL_TRG/EFL_GOLD and
//   --gtest_also_run_disabled_tests --gtest_filter=*decodeCompare*
TEST(EflomalAlignmentModelTest, DISABLED_decodeCompare)
{
  const char* srcEnv = std::getenv("EFL_SRC");
  const char* trgEnv = std::getenv("EFL_TRG");
  const char* goldEnv = std::getenv("EFL_GOLD");
  ASSERT_TRUE(srcEnv && trgEnv && goldEnv) << "EFL_SRC/EFL_TRG/EFL_GOLD must be set";

  std::vector<std::vector<std::string>> src = readTokenizedCorpus(srcEnv);
  std::vector<std::vector<std::string>> trg = readTokenizedCorpus(trgEnv);
  ASSERT_EQ(src.size(), trg.size());
  size_t n = src.size();

  std::vector<std::set<std::pair<int, int>>> gold(n);
  long goldTotal = 0;
  {
    std::ifstream gin(goldEnv);
    std::string line;
    size_t li = 0;
    while (li < n && std::getline(gin, line))
    {
      std::istringstream iss(line);
      std::string tok;
      while (iss >> tok)
      {
        size_t dash = tok.find('-');
        if (dash != std::string::npos)
          gold[li].insert({std::atoi(tok.substr(0, dash).c_str()), std::atoi(tok.substr(dash + 1).c_str())});
      }
      goldTotal += (long)gold[li].size();
      ++li;
    }
    ASSERT_EQ(li, n) << "gold line count != corpus line count";
  }

  std::unique_ptr<EflomalAlignmentModel> model(new EflomalAlignmentModel());
  int trainSamplers = std::max(1, envInt("EFL_SAMPLERS", 3));
  model->setNumSamplers(trainSamplers);
  model->setEflomalLexNorm(envInt("EFL_LEXNORM", 1) != 0);
  model->setAutoIterations(envInt("EFL_AUTOITERS", 1) != 0);
  model->setDeterministic(false);
  for (size_t i = 0; i < n; ++i)
    model->addSentencePair(src[i], trg[i], 1);

  // No setDecodeParams call => startTraining auto-ties the decode params.
  model->startTraining();
  int sweeps = model->getScheduledIterations();
  for (int it = 0; it < sweeps; ++it)
    model->train();
  model->endTraining();

  std::vector<std::vector<WordIndex>> srcIdx(n), trgIdx(n);
  for (size_t i = 0; i < n; ++i)
  {
    srcIdx[i] = model->strVectorToSrcIndexVector(src[i]);
    trgIdx[i] = model->strVectorToTrgIndexVector(trg[i]);
  }

  auto scoreAER = [&]() {
    long hyp = 0, inter = 0;
#pragma omp parallel for schedule(dynamic, 16) reduction(+ : hyp, inter)
    for (int i = 0; i < (int)n; ++i)
    {
      PositionIndex slen = (PositionIndex)src[i].size();
      std::vector<std::vector<double>> acc;
      model->accumulateDecodeMarginal(srcIdx[i], trgIdx[i], acc);
      for (size_t j = 0; j < acc.size(); ++j)
      {
        PositionIndex bestK = 0;
        double bestP = acc[j][0];
        for (PositionIndex k = 1; k <= slen; ++k)
          if (acc[j][k] > bestP)
          {
            bestP = acc[j][k];
            bestK = k;
          }
        if (bestK != slen)
        {
          ++hyp;
          if (gold[i].count({(int)bestK, (int)j}))
            ++inter;
        }
      }
    }
    return (hyp + goldTotal) ? 1.0 - (2.0 * inter) / (hyp + goldTotal) : 0.0;
  };

  double aerTied = scoreAER(); // tied params, active from startTraining
  model->setDecodeParams(8, 20, 4);
  double aerOld = scoreAER();

  int tiedIters = std::max(40, sweeps); // mirrors MinDecodeIters floor, for display
  std::cerr << "COMPARE tied(samplers=" << trainSamplers << ",iters=" << tiedIters << ",burnin=0)=" << std::fixed
            << std::setprecision(4) << aerTied << " old(8,20,4)=" << aerOld << " sweeps=" << sweeps << " over " << n
            << " pairs\n";
  SUCCEED();
}

// Training hyperparameter tuning harness. Trains with EFL_SAMPLERS chains and a
// schedule scaled by EFL_RELITERS (iters = max(2, round(rel*5000/sqrt(N))), then
// IBM1=max(2,iters/4), HMM=iters/4, fertility=iters), and reports forward AER two
// ways: AER_tied (decode auto-tied to the schedule => production-realistic) and
// AER_fixed (decode pinned to a generous (3,80,0) => isolates trained-table
// quality from the decode coupling). Disabled by default; run with
// EFL_SRC/EFL_TRG/EFL_GOLD and --gtest_also_run_disabled_tests --gtest_filter=*trainTune*
TEST(EflomalAlignmentModelTest, DISABLED_trainTune)
{
  const char* srcEnv = std::getenv("EFL_SRC");
  const char* trgEnv = std::getenv("EFL_TRG");
  const char* goldEnv = std::getenv("EFL_GOLD");
  ASSERT_TRUE(srcEnv && trgEnv && goldEnv) << "EFL_SRC/EFL_TRG/EFL_GOLD must be set";

  std::vector<std::vector<std::string>> src = readTokenizedCorpus(srcEnv);
  std::vector<std::vector<std::string>> trg = readTokenizedCorpus(trgEnv);
  ASSERT_EQ(src.size(), trg.size());
  size_t n = src.size();

  std::vector<std::set<std::pair<int, int>>> gold(n);
  long goldTotal = 0;
  {
    std::ifstream gin(goldEnv);
    std::string line;
    size_t li = 0;
    while (li < n && std::getline(gin, line))
    {
      std::istringstream iss(line);
      std::string tok;
      while (iss >> tok)
      {
        size_t dash = tok.find('-');
        if (dash != std::string::npos)
          gold[li].insert({std::atoi(tok.substr(0, dash).c_str()), std::atoi(tok.substr(dash + 1).c_str())});
      }
      goldTotal += (long)gold[li].size();
      ++li;
    }
    ASSERT_EQ(li, n) << "gold line count != corpus line count";
  }

  std::unique_ptr<EflomalAlignmentModel> model(new EflomalAlignmentModel());
  int trainSamplers = std::max(1, envInt("EFL_SAMPLERS", 3));
  const char* relEnv = std::getenv("EFL_RELITERS");
  double rel = relEnv ? std::atof(relEnv) : 1.0;
  model->setNumSamplers(trainSamplers);
  model->setEflomalLexNorm(envInt("EFL_LEXNORM", 1) != 0);
  model->setDeterministic(false);

  // Scaled corpus schedule (N approximated by the line count, fine for scaling).
  int iters = std::max(2, (int)std::llround(rel * 5000.0 / std::sqrt((double)n)));
  int i4 = std::max(1, iters / 4);
  model->setIterations(std::max(2, i4), i4, iters);
  model->setAutoIterations(false);
  for (size_t i = 0; i < n; ++i)
    model->addSentencePair(src[i], trg[i], 1);

  auto t0 = std::chrono::steady_clock::now();
  model->startTraining(); // ties decode to this schedule (decode params not pinned)
  int sweeps = model->getScheduledIterations();
  for (int it = 0; it < sweeps; ++it)
    model->train();
  model->endTraining();
  long trainMs =
      std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();

  std::vector<std::vector<WordIndex>> srcIdx(n), trgIdx(n);
  for (size_t i = 0; i < n; ++i)
  {
    srcIdx[i] = model->strVectorToSrcIndexVector(src[i]);
    trgIdx[i] = model->strVectorToTrgIndexVector(trg[i]);
  }

  auto scoreAER = [&]() {
    long hyp = 0, inter = 0;
#pragma omp parallel for schedule(dynamic, 16) reduction(+ : hyp, inter)
    for (int i = 0; i < (int)n; ++i)
    {
      PositionIndex slen = (PositionIndex)src[i].size();
      std::vector<std::vector<double>> acc;
      model->accumulateDecodeMarginal(srcIdx[i], trgIdx[i], acc);
      for (size_t j = 0; j < acc.size(); ++j)
      {
        PositionIndex bestK = 0;
        double bestP = acc[j][0];
        for (PositionIndex k = 1; k <= slen; ++k)
          if (acc[j][k] > bestP)
          {
            bestP = acc[j][k];
            bestK = k;
          }
        if (bestK != slen)
        {
          ++hyp;
          if (gold[i].count({(int)bestK, (int)j}))
            ++inter;
        }
      }
    }
    return (hyp + goldTotal) ? 1.0 - (2.0 * inter) / (hyp + goldTotal) : 0.0;
  };

  double aerTied = scoreAER();  // decode tied to this schedule
  model->setDecodeParams(3, 80, 0);
  double aerFixed = scoreAER(); // decode pinned -> isolates table quality

  std::cerr << "TRAINTUNE samplers=" << trainSamplers << " reliters=" << rel << " iters=" << sweeps << " AER_tied="
            << std::fixed << std::setprecision(4) << aerTied << " AER_fixed=" << aerFixed << " train " << trainMs
            << " ms over " << n << " pairs\n";
  SUCCEED();
}

// Verifies that decodeSamplers=1 matches the tied decodeSamplers=numSamplers: trains
// once (auto schedule, EFL_SAMPLERS chains) and reports forward AER for the tied decode
// (decodeSamplers=numSamplers, set by startTraining) vs decodeSamplers=1 at the SAME
// iters/burn-in. If equal, the inner decode-sampler loop is redundant with the training
// ensemble. Run with EFL_SRC/EFL_TRG/EFL_GOLD and --gtest_filter=*dsCheck*
TEST(EflomalAlignmentModelTest, DISABLED_dsCheck)
{
  const char* srcEnv = std::getenv("EFL_SRC");
  const char* trgEnv = std::getenv("EFL_TRG");
  const char* goldEnv = std::getenv("EFL_GOLD");
  ASSERT_TRUE(srcEnv && trgEnv && goldEnv) << "EFL_SRC/EFL_TRG/EFL_GOLD must be set";

  std::vector<std::vector<std::string>> src = readTokenizedCorpus(srcEnv);
  std::vector<std::vector<std::string>> trg = readTokenizedCorpus(trgEnv);
  ASSERT_EQ(src.size(), trg.size());
  size_t n = src.size();

  std::vector<std::set<std::pair<int, int>>> gold(n);
  long goldTotal = 0;
  {
    std::ifstream gin(goldEnv);
    std::string line;
    size_t li = 0;
    while (li < n && std::getline(gin, line))
    {
      std::istringstream iss(line);
      std::string tok;
      while (iss >> tok)
      {
        size_t dash = tok.find('-');
        if (dash != std::string::npos)
          gold[li].insert({std::atoi(tok.substr(0, dash).c_str()), std::atoi(tok.substr(dash + 1).c_str())});
      }
      goldTotal += (long)gold[li].size();
      ++li;
    }
    ASSERT_EQ(li, n) << "gold line count != corpus line count";
  }

  std::unique_ptr<EflomalAlignmentModel> model(new EflomalAlignmentModel());
  int trainSamplers = std::max(1, envInt("EFL_SAMPLERS", 3));
  model->setNumSamplers(trainSamplers);
  model->setEflomalLexNorm(envInt("EFL_LEXNORM", 1) != 0);
  model->setAutoIterations(envInt("EFL_AUTOITERS", 1) != 0);
  model->setDeterministic(false);
  for (size_t i = 0; i < n; ++i)
    model->addSentencePair(src[i], trg[i], 1);

  model->startTraining(); // ties decodeSamplers=numSamplers, decodeIters=max(40,total), burnin=0
  int sweeps = model->getScheduledIterations();
  for (int it = 0; it < sweeps; ++it)
    model->train();
  model->endTraining();

  std::vector<std::vector<WordIndex>> srcIdx(n), trgIdx(n);
  for (size_t i = 0; i < n; ++i)
  {
    srcIdx[i] = model->strVectorToSrcIndexVector(src[i]);
    trgIdx[i] = model->strVectorToTrgIndexVector(trg[i]);
  }

  auto scoreAER = [&]() {
    long hyp = 0, inter = 0;
#pragma omp parallel for schedule(dynamic, 16) reduction(+ : hyp, inter)
    for (int i = 0; i < (int)n; ++i)
    {
      PositionIndex slen = (PositionIndex)src[i].size();
      std::vector<std::vector<double>> acc;
      model->accumulateDecodeMarginal(srcIdx[i], trgIdx[i], acc);
      for (size_t j = 0; j < acc.size(); ++j)
      {
        PositionIndex bestK = 0;
        double bestP = acc[j][0];
        for (PositionIndex k = 1; k <= slen; ++k)
          if (acc[j][k] > bestP)
          {
            bestP = acc[j][k];
            bestK = k;
          }
        if (bestK != slen)
        {
          ++hyp;
          if (gold[i].count({(int)bestK, (int)j}))
            ++inter;
        }
      }
    }
    return (hyp + goldTotal) ? 1.0 - (2.0 * inter) / (hyp + goldTotal) : 0.0;
  };

  int tiedIters = std::max(40, sweeps);
  double aerTied = scoreAER();          // decodeSamplers = numSamplers (from startTraining)
  model->setDecodeParams(1, tiedIters, 0);
  double aerDs1 = scoreAER();           // decodeSamplers = 1, same iters/burn-in

  std::cerr << "DSCHECK samplers=" << trainSamplers << " iters=" << tiedIters << " AER_tied=" << std::fixed
            << std::setprecision(4) << aerTied << " AER_ds1=" << aerDs1 << " over " << n << " pairs\n";
  SUCCEED();
}

// Runtime (and AER) comparison of the thot aligners on one corpus. EFL_MODEL selects
// eflomal (auto schedule, EFL_SAMPLERS chains, tied decode), fast_align (EFL_FA iters)
// or hmm (EFL_HI IBM1 + EFL_HH HMM iters). Trains and aligns the full corpus
// (transductive) and reports train ms, align ms (sequential, for a fair cross-model
// number) and forward AER. Run with EFL_SRC/EFL_TRG/EFL_GOLD and --gtest_filter=*runtimeCompare*
TEST(EflomalAlignmentModelTest, DISABLED_runtimeCompare)
{
  const char* srcEnv = std::getenv("EFL_SRC");
  const char* trgEnv = std::getenv("EFL_TRG");
  const char* goldEnv = std::getenv("EFL_GOLD");
  ASSERT_TRUE(srcEnv && trgEnv && goldEnv) << "EFL_SRC/EFL_TRG/EFL_GOLD must be set";
  std::string mt = std::getenv("EFL_MODEL") ? std::getenv("EFL_MODEL") : "eflomal";

  std::vector<std::vector<std::string>> src = readTokenizedCorpus(srcEnv);
  std::vector<std::vector<std::string>> trg = readTokenizedCorpus(trgEnv);
  ASSERT_EQ(src.size(), trg.size());
  size_t n = src.size();

  std::vector<std::set<std::pair<int, int>>> gold(n);
  long goldTotal = 0;
  {
    std::ifstream gin(goldEnv);
    std::string line;
    size_t li = 0;
    while (li < n && std::getline(gin, line))
    {
      std::istringstream iss(line);
      std::string tok;
      while (iss >> tok)
      {
        size_t dash = tok.find('-');
        if (dash != std::string::npos)
          gold[li].insert({std::atoi(tok.substr(0, dash).c_str()), std::atoi(tok.substr(dash + 1).c_str())});
      }
      goldTotal += (long)gold[li].size();
      ++li;
    }
    ASSERT_EQ(li, n) << "gold line count != corpus line count";
  }

  // Aligns the full corpus sequentially against the given trained model and returns
  // forward AER; sets alignMs. Sequential keeps the cross-model comparison fair and
  // avoids per-model getBestAlignment thread-safety assumptions.
  auto alignScore = [&](AlignmentModel& m, long& alignMs) {
    std::vector<std::vector<WordIndex>> si(n), ti(n);
    for (size_t i = 0; i < n; ++i)
    {
      si[i] = m.strVectorToSrcIndexVector(src[i]);
      ti[i] = m.strVectorToTrgIndexVector(trg[i]);
    }
    long hyp = 0, inter = 0;
    auto a0 = std::chrono::steady_clock::now();
    // Parallel align over sentences (realistic deployment), matching how eflomal's
    // warm argmax parallelizes; getBestAlignment reads only the trained tables.
#pragma omp parallel for schedule(dynamic, 16) reduction(+ : hyp, inter)
    for (int i = 0; i < (int)n; ++i)
    {
      std::vector<PositionIndex> alig;
      m.getBestAlignment(si[i], ti[i], alig);
      for (size_t j = 0; j < alig.size(); ++j)
        if (alig[j] > 0)
        {
          ++hyp;
          if (gold[i].count({(int)alig[j] - 1, (int)j}))
            ++inter;
        }
    }
    alignMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - a0).count();
    return (hyp + goldTotal) ? 1.0 - (2.0 * inter) / (hyp + goldTotal) : 0.0;
  };

  long trainMs = 0, alignMs = 0;
  double aer = 0;
  int detail = 0;
  auto t0 = std::chrono::steady_clock::now();

  if (mt == "fast_align")
  {
    int faIters = envInt("EFL_FA", 5);
    detail = faIters;
    FastAlignModel m;
    for (size_t i = 0; i < n; ++i)
      m.addSentencePair(src[i], trg[i], 1);
    m.startTraining();
    for (int it = 0; it < faIters; ++it)
      m.train();
    m.endTraining();
    trainMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
    aer = alignScore(m, alignMs);
  }
  else if (mt == "hmm")
  {
    int hi = envInt("EFL_HI", 5), hh = envInt("EFL_HH", 5);
    detail = hi + hh;
    Ibm1AlignmentModel ibm1;
    for (size_t i = 0; i < n; ++i)
      ibm1.addSentencePair(src[i], trg[i], 1);
    ibm1.startTraining();
    for (int it = 0; it < hi; ++it)
      ibm1.train();
    ibm1.endTraining();
    HmmAlignmentModel m(ibm1);
    m.startTraining();
    for (int it = 0; it < hh; ++it)
      m.train();
    m.endTraining();
    trainMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
    aer = alignScore(m, alignMs);
  }
  else // eflomal (new defaults + warm persisted alignments for the transductive corpus)
  {
    EflomalAlignmentModel m;
    m.setNumSamplers(std::max(1, envInt("EFL_SAMPLERS", 5)));
    m.setEflomalLexNorm(envInt("EFL_LEXNORM", 1) != 0);
    m.setAutoIterations(true);
    m.setDeterministic(false);
    m.setEmitTrainingAlignments(true); // warm final-argmax produced in endTraining
    for (size_t i = 0; i < n; ++i)
      m.addSentencePair(src[i], trg[i], 1);
    m.startTraining();
    int sweeps = m.getScheduledIterations();
    detail = sweeps;
    for (int it = 0; it < sweeps; ++it)
      m.train();
    m.endTraining();
    // "train" now includes producing the warm alignments; aligning the training
    // corpus is then just reading them (no separate decode).
    trainMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
    auto a0 = std::chrono::steady_clock::now();
    long hyp = 0, inter = 0;
    for (size_t i = 0; i < n; ++i)
    {
      std::vector<PositionIndex> alig;
      m.getTrainingAlignment(i, alig); // per target: 1-based src, 0=NULL
      for (size_t j = 0; j < alig.size(); ++j)
        if (alig[j] > 0)
        {
          ++hyp;
          if (gold[i].count({(int)alig[j] - 1, (int)j}))
            ++inter;
        }
    }
    alignMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - a0).count();
    aer = (hyp + goldTotal) ? 1.0 - (2.0 * inter) / (hyp + goldTotal) : 0.0;
  }

  std::cerr << "RUNTIME model=" << mt << " detail=" << detail << " pairs=" << n << " train " << trainMs
            << " ms, align " << alignMs << " ms, AER=" << std::fixed << std::setprecision(4) << aer << "\n";
  SUCCEED();
}

// Phase-0 scaling grid for the eflomal-optimizations effort (see
// eflomal-optimizations.md at the repo root). Trains the same corpus repeatedly
// across a matrix of numSamplers x OMP thread count x deterministic, to measure the
// actual parallel scaling ceiling before any code changes. No gold alignment
// needed; this is a pure timing harness. Run with EFL_SRC/EFL_TRG and
// --gtest_filter=*scalingGrid*. Grids are comma-separated env overrides:
// EFL_GRID_SAMPLERS (default "1,3,8"), EFL_GRID_THREADS (default "1,2,4,8"),
// EFL_GRID_DET (default "0,1", 0=false/1=true).
namespace
{
std::vector<int> parseIntList(const char* env, const char* dflt)
{
  std::string s = env ? env : dflt;
  std::vector<int> out;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ','))
    if (!tok.empty())
      out.push_back(std::atoi(tok.c_str()));
  return out;
}
} // namespace

TEST(EflomalAlignmentModelTest, DISABLED_scalingGrid)
{
  const char* srcEnv = std::getenv("EFL_SRC");
  const char* trgEnv = std::getenv("EFL_TRG");
  ASSERT_TRUE(srcEnv && trgEnv) << "EFL_SRC/EFL_TRG must be set";

  std::vector<std::vector<std::string>> src = readTokenizedCorpus(srcEnv);
  std::vector<std::vector<std::string>> trg = readTokenizedCorpus(trgEnv);
  ASSERT_EQ(src.size(), trg.size());
  size_t n = src.size();

  std::vector<int> samplersGrid = parseIntList(std::getenv("EFL_GRID_SAMPLERS"), "1,3,8");
  std::vector<int> threadsGrid = parseIntList(std::getenv("EFL_GRID_THREADS"), "1,2,4,8");
  std::vector<int> detGrid = parseIntList(std::getenv("EFL_GRID_DET"), "0,1");
  // Off by default (matches the original grid so round-1 numbers stay comparable);
  // set EFL_GRID_EMIT=1 to also produce warm training alignments in endTraining,
  // exercising computeTrainingAlignments' decode-path cost (the fable-improvements.md
  // item-2 target), which end_ms otherwise never touches.
  bool emit = envInt("EFL_GRID_EMIT", 0) != 0;

  for (int det : detGrid)
  {
    for (int samplers : samplersGrid)
    {
      for (int threads : threadsGrid)
      {
        omp_set_num_threads(threads);

        EflomalAlignmentModel m;
        m.setNumSamplers(samplers);
        m.setDeterministic(det != 0);
        m.setAutoIterations(true);
        m.setEmitTrainingAlignments(emit);
        for (size_t i = 0; i < n; ++i)
          m.addSentencePair(src[i], trg[i], 1);

        auto t0 = std::chrono::steady_clock::now();
        m.startTraining();
        auto t1 = std::chrono::steady_clock::now();
        int sweeps = m.getScheduledIterations();
        for (int it = 0; it < sweeps; ++it)
          m.train();
        auto t2 = std::chrono::steady_clock::now();
        m.endTraining();
        auto t3 = std::chrono::steady_clock::now();

        long initMs = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        long trainMs = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        long endMs = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
        long totalMs = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t0).count();

        std::cerr << "GRID samplers=" << samplers << " threads=" << threads << " det=" << det
                  << " pairs=" << n << " sweeps=" << sweeps << " init_ms=" << initMs << " train_ms=" << trainMs
                  << " end_ms=" << endMs << " total_ms=" << totalMs << "\n";
      }
    }
  }
  SUCCEED();
}

// Phase-1 opt 5 for the eflomal-optimizations effort: SymmetrizedAligner/
// SymmetrizedAlignmentModel compose an already-trained direct + inverse model
// (decode-time only) but nothing trains the two directions concurrently, even
// though they are fully independent. Trains direct (EFL_SRC->EFL_TRG) and inverse
// (EFL_TRG->EFL_SRC) sequentially, then concurrently via std::async, at a fixed
// numSamplers (EFL_SYM_SAMPLERS, default 3 - kept well under half the machine's
// hardware_concurrency so 2*numSamplers threads don't oversubscribe). Run with
// EFL_SRC/EFL_TRG and --gtest_filter=*concurrentDirections*.
TEST(EflomalAlignmentModelTest, DISABLED_concurrentDirections)
{
  const char* srcEnv = std::getenv("EFL_SRC");
  const char* trgEnv = std::getenv("EFL_TRG");
  ASSERT_TRUE(srcEnv && trgEnv) << "EFL_SRC/EFL_TRG must be set";

  std::vector<std::vector<std::string>> src = readTokenizedCorpus(srcEnv);
  std::vector<std::vector<std::string>> trg = readTokenizedCorpus(trgEnv);
  ASSERT_EQ(src.size(), trg.size());
  size_t n = src.size();

  int samplers = envInt("EFL_SYM_SAMPLERS", 3);

  auto trainOneDirection = [&](bool swapped) {
    EflomalAlignmentModel m;
    m.setNumSamplers(samplers);
    m.setDeterministic(false);
    m.setAutoIterations(true);
    for (size_t i = 0; i < n; ++i)
    {
      if (swapped)
        m.addSentencePair(trg[i], src[i], 1);
      else
        m.addSentencePair(src[i], trg[i], 1);
    }
    m.startTraining();
    int sweeps = m.getScheduledIterations();
    for (int it = 0; it < sweeps; ++it)
      m.train();
    m.endTraining();
  };

  // Sequential: direct then inverse, back to back (today's implicit behaviour, since
  // nothing in the library trains both directions at once).
  auto s0 = std::chrono::steady_clock::now();
  trainOneDirection(false);
  trainOneDirection(true);
  long seqMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - s0).count();

  // Concurrent: both directions on separate threads. Each spawns its own numSamplers
  // OpenMP chains, so this only pays off wall-clock-wise while 2*numSamplers stays
  // within the machine's hardware concurrency; past that the two directions start
  // competing for the same cores instead of adding parallelism.
  auto c0 = std::chrono::steady_clock::now();
  auto futDirect = std::async(std::launch::async, trainOneDirection, false);
  auto futInverse = std::async(std::launch::async, trainOneDirection, true);
  futDirect.get();
  futInverse.get();
  long concMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - c0).count();

  std::cerr << "SYMDIR samplers=" << samplers << " hw_concurrency=" << std::thread::hardware_concurrency()
            << " pairs=" << n << " sequential_ms=" << seqMs << " concurrent_ms=" << concMs
            << " speedup=" << std::fixed << std::setprecision(2) << (double)seqMs / (double)concMs << "\n";
  SUCCEED();
}

// Benchmark for fable-improvements.md item 11: loglikelihoodForPairRange's new
// #pragma omp reduction over computeSumLogProb (previously a strictly serial loop).
// Trains briefly (this benchmarks decode/scoring, not training quality) then times
// loglikelihoodForPairRange over the whole corpus at threads=1 vs EFL_LL_THREADS.
// Run with EFL_SRC/EFL_TRG and --gtest_filter=*loglikelihoodParallel*.
TEST(EflomalAlignmentModelTest, DISABLED_loglikelihoodParallel)
{
  const char* srcEnv = std::getenv("EFL_SRC");
  const char* trgEnv = std::getenv("EFL_TRG");
  ASSERT_TRUE(srcEnv && trgEnv) << "EFL_SRC/EFL_TRG must be set";

  std::vector<std::vector<std::string>> src = readTokenizedCorpus(srcEnv);
  std::vector<std::vector<std::string>> trg = readTokenizedCorpus(trgEnv);
  ASSERT_EQ(src.size(), trg.size());
  size_t n = src.size();

  int threads = envInt("EFL_LL_THREADS", 8);

  EflomalAlignmentModel model;
  model.setNumSamplers(3);
  model.setDeterministic(false);
  model.setIterations(4, 4, 4); // short: this times decode/scoring, not training
  for (size_t i = 0; i < n; ++i)
    model.addSentencePair(src[i], trg[i], 1);
  omp_set_num_threads(threads);
  train(model, 12);

  omp_set_num_threads(1);
  auto t0 = std::chrono::steady_clock::now();
  auto serialResult = model.loglikelihoodForPairRange({0, model.numSentencePairs() - 1});
  long serialMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();

  omp_set_num_threads(threads);
  auto t1 = std::chrono::steady_clock::now();
  auto parallelResult = model.loglikelihoodForPairRange({0, model.numSentencePairs() - 1});
  long parallelMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t1).count();

  std::cerr << "LLPARALLEL pairs=" << n << " threads=" << threads << " serial_ms=" << serialMs
            << " parallel_ms=" << parallelMs << " speedup=" << std::fixed << std::setprecision(2)
            << (double)serialMs / (double)parallelMs << " ll_serial=" << serialResult.first
            << " ll_parallel=" << parallelResult.first << "\n";
  SUCCEED();
}

// Validates persisted training alignments + round-trip: trains with
// setEmitTrainingAlignments, computes AER of the in-memory warm alignments,
// saves, reloads, and checks the reloaded links are identical and score the same
// AER. Run with EFL_SRC/EFL_TRG/EFL_GOLD and --gtest_filter=*alignFile*
TEST(EflomalAlignmentModelTest, DISABLED_alignFile)
{
  const char* srcEnv = std::getenv("EFL_SRC");
  const char* trgEnv = std::getenv("EFL_TRG");
  const char* goldEnv = std::getenv("EFL_GOLD");
  ASSERT_TRUE(srcEnv && trgEnv && goldEnv) << "EFL_SRC/EFL_TRG/EFL_GOLD must be set";

  std::vector<std::vector<std::string>> src = readTokenizedCorpus(srcEnv);
  std::vector<std::vector<std::string>> trg = readTokenizedCorpus(trgEnv);
  ASSERT_EQ(src.size(), trg.size());
  size_t n = src.size();

  std::vector<std::set<std::pair<int, int>>> gold(n);
  long goldTotal = 0;
  {
    std::ifstream gin(goldEnv);
    std::string line;
    size_t li = 0;
    while (li < n && std::getline(gin, line))
    {
      std::istringstream iss(line);
      std::string tok;
      while (iss >> tok)
      {
        size_t dash = tok.find('-');
        if (dash != std::string::npos)
          gold[li].insert({std::atoi(tok.substr(0, dash).c_str()), std::atoi(tok.substr(dash + 1).c_str())});
      }
      goldTotal += (long)gold[li].size();
      ++li;
    }
    ASSERT_EQ(li, n) << "gold line count != corpus line count";
  }

  using AlignVec = std::vector<std::vector<PositionIndex>>; // per-target: 1-based src, 0=NULL
  auto aerOf = [&](const AlignVec& al) {
    long A = 0, inter = 0;
    for (size_t i = 0; i < al.size() && i < gold.size(); ++i)
      for (size_t j = 0; j < al[i].size(); ++j)
        if (al[i][j] > 0)
        {
          ++A;
          if (gold[i].count({(int)al[i][j] - 1, (int)j}))
            ++inter;
        }
    return (A + goldTotal) ? 1.0 - (2.0 * inter) / (A + goldTotal) : 0.0;
  };

  EflomalAlignmentModel model;
  model.setNumSamplers(std::max(1, envInt("EFL_SAMPLERS", 5)));
  model.setEflomalLexNorm(envInt("EFL_LEXNORM", 1) != 0);
  model.setAutoIterations(true);
  model.setDeterministic(false);
  model.setEmitTrainingAlignments(true);
  for (size_t i = 0; i < n; ++i)
    model.addSentencePair(src[i], trg[i], 1);
  model.startTraining();
  int sweeps = model.getScheduledIterations();
  for (int it = 0; it < sweeps; ++it)
    model.train();
  model.endTraining();

  AlignVec inmem(model.numSentencePairs()); // copy before reload
  for (unsigned int i = 0; i < model.numSentencePairs(); ++i)
    model.getTrainingAlignment(i, inmem[i]);
  ASSERT_EQ(inmem.size(), n) << "expected one alignment per training pair";
  double aerInmem = aerOf(inmem);

  std::string prefix = "eflomal_aligns_test";
  ASSERT_EQ(model.print(prefix.c_str()), THOT_OK);

  EflomalAlignmentModel loaded;
  ASSERT_EQ(loaded.load(prefix.c_str()), THOT_OK);
  AlignVec reloaded(loaded.numSentencePairs());
  for (unsigned int i = 0; i < loaded.numSentencePairs(); ++i)
    loaded.getTrainingAlignment(i, reloaded[i]);

  // The round-trip recovers each pair's target length from the sentence handler,
  // so the per-target vectors (including trailing NULLs) must match exactly.
  bool sizeMatch = reloaded.size() == inmem.size();
  size_t mismatches = 0;
  for (size_t i = 0; sizeMatch && i < inmem.size(); ++i)
    if (reloaded[i] != inmem[i])
      ++mismatches;
  double aerReloaded = aerOf(reloaded);

  std::cerr << "ALIGNFILE pairs=" << n << " sizeMatch=" << (sizeMatch ? 1 : 0) << " mismatches=" << mismatches
            << " AER_inmem=" << std::fixed << std::setprecision(4) << aerInmem << " AER_reloaded=" << aerReloaded
            << "\n";
  EXPECT_TRUE(sizeMatch);
  EXPECT_EQ(mismatches, 0u);
  EXPECT_NEAR(aerInmem, aerReloaded, 1e-9);
  SUCCEED();
}
