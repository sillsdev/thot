#include "sw_models/EflomalAlignmentModel.h"

#include "TestUtils.h"
#include "sw_models/HmmAlignmentModel.h"
#include "sw_models/Ibm1AlignmentModel.h"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <gtest/gtest.h>
#include <memory>
#include <sstream>
#include <string>

namespace
{
constexpr double kEpsilon = 1e-6;

void trainEflomal(EflomalAlignmentModel& model)
{
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
  model.setAlphaLex(0.002);
  model.setAlphaNull(0.0005);
  model.setAlphaJump(0.3);
  model.setAlphaFertility(0.7);
  model.setNullProb(0.15);
  model.setJumpWindow(50);
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
  EXPECT_EQ(loaded.getIbm1Iters(), 3);
  EXPECT_EQ(loaded.getHmmIters(), 5);
  EXPECT_EQ(loaded.getFertilityIters(), 7);
  EXPECT_NEAR(loaded.getAlphaLex(), 0.002, kEpsilon);
  EXPECT_NEAR(loaded.getAlphaNull(), 0.0005, kEpsilon);
  EXPECT_NEAR(loaded.getAlphaJump(), 0.3, kEpsilon);
  EXPECT_NEAR(loaded.getAlphaFertility(), 0.7, kEpsilon);
  EXPECT_NEAR(loaded.getNullProb(), 0.15, kEpsilon);
  EXPECT_EQ(loaded.getJumpWindow(), 50);
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

double envDouble(const char* key, double dflt)
{
  const char* v = std::getenv(key);
  return v ? std::atof(v) : dflt;
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
  double alphaLex = envDouble("EFL_ALPHA_LEX", -1.0);
  double alphaNull = envDouble("EFL_ALPHA_NULL", -1.0);
  double alphaJump = envDouble("EFL_ALPHA_JUMP", -1.0);
  double alphaFert = envDouble("EFL_ALPHA_FERT", -1.0);
  double nullProb = envDouble("EFL_NULL_PROB", -1.0);
  int jumpWindow = envInt("EFL_JUMP_WINDOW", -1);

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
    // EFL_DETERMINISTIC: 1=force deterministic (single-threaded, reproducible),
    // 0=parallel chains. Defaults to 0 (parallel) in ablation mode so multi-sampler
    // configs actually use OpenMP.
    model->setDeterministic(envInt("EFL_DETERMINISTIC", 0) != 0);
    model->setEflomalLexNorm(lexNorm);
    model->setAutoIterations(autoIters);
    model->setDecodeParams(envInt("EFL_DSAMP", 4), envInt("EFL_DITERS", 12), envInt("EFL_DBURN", 4));
    if (!autoIters)
      model->setIterations(ibm1, hmm, fert);
    if (alphaLex >= 0)
      model->setAlphaLex(alphaLex);
    if (alphaNull >= 0)
      model->setAlphaNull(alphaNull);
    if (alphaJump >= 0)
      model->setAlphaJump(alphaJump);
    if (alphaFert >= 0)
      model->setAlphaFertility(alphaFert);
    if (nullProb >= 0)
      model->setNullProb(nullProb);
    if (jumpWindow > 0)
      model->setJumpWindow(jumpWindow);
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
