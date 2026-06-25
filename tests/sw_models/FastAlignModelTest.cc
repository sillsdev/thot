#include "sw_models/FastAlignModel.h"

#include "TestUtils.h"
#include "nlp_common/ErrorDefs.h"
#include "nlp_common/MathDefs.h"

#include <gtest/gtest.h>

TEST(FastAlignModelTest, trainEmpty)
{
  FastAlignModel model;
  EXPECT_NO_THROW(train(model));
}

TEST(FastAlignModelTest, train)
{
  FastAlignModel model;
  addTrainingData(model);
  train(model, 2);

  std::vector<PositionIndex> alignment;
  model.getBestAlignment("isthay isyay ayay esttay-N .", "this is a test N .", alignment);
  EXPECT_EQ(alignment, (std::vector<PositionIndex>{1, 2, 3, 4, 4, 5}));

  model.getBestAlignment("isthay isyay otnay ayay esttay-N .", "this is not a test N .", alignment);
  EXPECT_EQ(alignment, (std::vector<PositionIndex>{1, 2, 3, 4, 5, 5, 6}));

  model.getBestAlignment("isthay isyay ayay esttay-N ardhay .", "this is a hard test N .", alignment);
  EXPECT_EQ(alignment, (std::vector<PositionIndex>{1, 2, 3, 5, 4, 4, 6}));
}

TEST(FastAlignModelTest, incrTrain)
{
  FastAlignModel model;
  addTrainingData(model);
  incrTrain(model, std::make_pair(0, model.numSentencePairs() - 1), 2);

  std::vector<PositionIndex> alignment;
  model.getBestAlignment("isthay isyay ayay esttay-N .", "this is a test N .", alignment);
  EXPECT_EQ(alignment, (std::vector<PositionIndex>{1, 2, 3, 4, 4, 5}));

  model.getBestAlignment("isthay isyay otnay ayay esttay-N .", "this is not a test N .", alignment);
  EXPECT_EQ(alignment, (std::vector<PositionIndex>{1, 2, 3, 4, 5, 5, 6}));

  model.getBestAlignment("isthay isyay ayay esttay-N ardhay .", "this is a hard test N .", alignment);
  EXPECT_EQ(alignment, (std::vector<PositionIndex>{1, 2, 3, 5, 4, 4, 6}));
}

TEST(FastAlignModelTest, trainingAlignments)
{
  // The training-alignment capability is generalized in AlignmentModelBase, so a
  // non-Eflomal model gains it for free: the base default emits getBestAlignment
  // for every training pair and persists/restores them via ".aligns".
  FastAlignModel model;
  addTrainingData(model);
  model.setEmitTrainingAlignments(true);
  train(model, 2);

  for (unsigned int n = 0; n < model.numSentencePairs(); ++n)
  {
    std::vector<std::string> src, trg;
    Count c;
    model.getSentencePair(n, src, trg, c);
    std::vector<PositionIndex> expected;
    model.getBestAlignment(src, trg, expected);

    // The single-pair accessor mirrors getBestAlignment: it fills the same
    // alignment (per-target form) and returns that alignment's log-probability.
    std::vector<PositionIndex> alig;
    LgProb prob = model.getTrainingAlignment(n, alig);
    EXPECT_EQ(alig, expected);
    EXPECT_EQ(alig.size(), trg.size());
    WordAlignmentMatrix waMatrix;
    waMatrix.init((PositionIndex)src.size(), (PositionIndex)trg.size());
    waMatrix.putAligVec(alig);
    EXPECT_NEAR((double)prob, (double)model.computeLogProb(src, trg, waMatrix), EPSILON);

    // The WordAlignmentMatrix overload yields the same alignment and probability.
    WordAlignmentMatrix trainMatrix;
    LgProb matrixProb = model.getTrainingAlignment(n, trainMatrix);
    std::vector<PositionIndex> fromMatrix;
    trainMatrix.getAligVec(fromMatrix);
    EXPECT_EQ(fromMatrix, alig);
    EXPECT_NEAR((double)matrixProb, (double)prob, EPSILON);
  }

  // Out-of-range index clears the alignment and returns the sentinel.
  std::vector<PositionIndex> oob{1};
  EXPECT_EQ((double)model.getTrainingAlignment(1000, oob), (double)SMALL_LG_NUM);
  EXPECT_TRUE(oob.empty());

  // Persisted ".aligns" round-trips exactly, and the restored alignments score
  // identically (the probability is recomputed from the loaded model).
  std::string prefix = "fast_align_aligns_test";
  ASSERT_EQ(model.print(prefix.c_str()), THOT_OK);
  FastAlignModel loaded;
  ASSERT_EQ(loaded.load(prefix.c_str()), THOT_OK);
  ASSERT_EQ(loaded.numSentencePairs(), model.numSentencePairs());
  for (unsigned int n = 0; n < model.numSentencePairs(); ++n)
  {
    std::vector<PositionIndex> a1, a2;
    LgProb p1 = model.getTrainingAlignment(n, a1);
    LgProb p2 = loaded.getTrainingAlignment(n, a2);
    EXPECT_EQ(a1, a2);
    EXPECT_NEAR((double)p1, (double)p2, EPSILON);
  }
}

TEST(FastAlignModelTest, trainingAlignmentIndexMatchesSentencePair)
{
  // A pair that fails the length filter (here, an empty source) is not trained on,
  // but getTrainingAlignment is still indexed by sentence-handler pair index, so
  // index n lines up with getSentencePair(n) regardless of filtered pairs.
  FastAlignModel model;
  addSentencePair(model, "isthay isyay ayay esttay-N .", "this is a test N .");
  addSentencePair(model, "", "this is filtered out"); // index 1: filtered (empty source)
  addSentencePair(model, "ouyay ouldshay esttay-V oftenyay .", "you should test V often .");
  addSentencePair(model, "isyay isthay orkingway ?", "is this working ?");
  model.setEmitTrainingAlignments(true);
  train(model, 2);

  // The filtered pair mimics getBestAlignment on length-invalid input: an all-NULL
  // alignment of the target length, scored SMALL_LG_NUM (here target "this is
  // filtered out" has 4 tokens).
  std::vector<std::string> fsrc, ftrg;
  Count fc;
  model.getSentencePair(1, fsrc, ftrg, fc);
  std::vector<PositionIndex> expectedFiltered;
  LgProb expectedFilteredProb = model.getBestAlignment(fsrc, ftrg, expectedFiltered);
  std::vector<PositionIndex> filteredAlig{7};
  LgProb filteredProb = model.getTrainingAlignment(1, filteredAlig);
  EXPECT_EQ(filteredAlig, expectedFiltered);
  EXPECT_EQ(filteredAlig, (std::vector<PositionIndex>{0, 0, 0, 0}));
  EXPECT_EQ((double)filteredProb, (double)expectedFilteredProb);
  EXPECT_EQ((double)filteredProb, (double)SMALL_LG_NUM);

  // Every other index lines up with getSentencePair(n): the stored alignment equals
  // getBestAlignment on that very pair (which would not hold under filtered-order
  // indexing, where index 1 would have been the third pair).
  for (unsigned int n = 0; n < model.numSentencePairs(); ++n)
  {
    if (n == 1)
      continue;
    std::vector<std::string> src, trg;
    Count c;
    model.getSentencePair(n, src, trg, c);
    std::vector<PositionIndex> expected;
    model.getBestAlignment(src, trg, expected);
    std::vector<PositionIndex> alig;
    model.getTrainingAlignment(n, alig);
    EXPECT_EQ(alig, expected);
  }

  // Round-trip preserves the per-pair indexing, including the filtered slot
  // (recovered as an empty stored alignment, surfaced as an all-NULL alignment).
  std::string prefix = "fast_align_aligns_filtered_test";
  ASSERT_EQ(model.print(prefix.c_str()), THOT_OK);
  FastAlignModel loaded;
  ASSERT_EQ(loaded.load(prefix.c_str()), THOT_OK);
  ASSERT_EQ(loaded.numSentencePairs(), model.numSentencePairs());
  for (unsigned int n = 0; n < model.numSentencePairs(); ++n)
  {
    std::vector<PositionIndex> a1, a2;
    model.getTrainingAlignment(n, a1);
    loaded.getTrainingAlignment(n, a2);
    EXPECT_EQ(a1, a2);
  }
}

TEST(FastAlignModelTest, computeLogProb)
{
  FastAlignModel model;
  addTrainingData(model);
  train(model);

  std::vector<PositionIndex> alignment;
  LgProb expectedLogProb = model.getBestAlignment("isthay isyay ayay esttay-N .", "this is a test N NULL .", alignment);
  WordAlignmentMatrix waMatrix{5, 7};
  waMatrix.putAligVec(alignment);
  LgProb logProb = model.computeLogProb("isthay isyay ayay esttay-N .", "this is a test N NULL .", waMatrix);
  EXPECT_NEAR(logProb, expectedLogProb, EPSILON);
}
