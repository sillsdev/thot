#include "sw_models/IncrHmmAlignmentModel.h"

#include "TestUtils.h"

#include <gtest/gtest.h>

TEST(IncrHmmAlignmentModelTest, train)
{
  IncrHmmAlignmentModel model;
  model.setHmmP0(0.1);
  addTrainingData(model);
  train(model, 2);

  std::vector<PositionIndex> alignment;
  model.getBestAlignment("isthay isyay ayay esttay-N .", "this is a test N .", alignment);
  EXPECT_EQ(alignment, (std::vector<PositionIndex>{1, 2, 3, 4, 4, 5}));

  model.getBestAlignment("isthay isyay otnay ayay esttay-N .", "this is not a test N .", alignment);
  EXPECT_EQ(alignment, (std::vector<PositionIndex>{1, 2, 4, 4, 5, 5, 5}));

  model.getBestAlignment("isthay isyay ayay esttay-N ardhay .", "this is a hard test N .", alignment);
  EXPECT_EQ(alignment, (std::vector<PositionIndex>{1, 2, 3, 5, 4, 4, 4}));
}

TEST(IncrHmmAlignmentModelTest, incrTrain)
{
  IncrHmmAlignmentModel model;
  model.setHmmP0(0.1);
  addTrainingData(model);
  incrTrain(model, std::make_pair(0, model.numSentencePairs() - 1), 2);

  std::vector<PositionIndex> alignment;
  model.getBestAlignment("isthay isyay ayay esttay-N .", "this is a test N .", alignment);
  EXPECT_EQ(alignment, (std::vector<PositionIndex>{1, 2, 3, 4, 4, 5}));

  model.getBestAlignment("isthay isyay otnay ayay esttay-N .", "this is not a test N .", alignment);
  EXPECT_EQ(alignment, (std::vector<PositionIndex>{1, 2, 4, 4, 5, 5, 5}));

  model.getBestAlignment("isthay isyay ayay esttay-N ardhay .", "this is a hard test N .", alignment);
  EXPECT_EQ(alignment, (std::vector<PositionIndex>{1, 2, 3, 5, 4, 4, 4}));
}

TEST(IncrHmmAlignmentModelTest, computeLogProb)
{
  IncrHmmAlignmentModel model;
  model.setHmmP0(0.2);
  addTrainingData(model);
  train(model);

  std::vector<PositionIndex> alignment;
  LgProb expectedLogProb = model.getBestAlignment("isthay isyay ayay esttay-N .", "this is a test N NULL .", alignment);
  WordAlignmentMatrix waMatrix{5, 7};
  waMatrix.putAligVec(alignment);
  LgProb logProb = model.computeLogProb("isthay isyay ayay esttay-N .", "this is a test N NULL .", waMatrix);
  EXPECT_NEAR(logProb, expectedLogProb, EPSILON);
}
