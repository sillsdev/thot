#include "sw_models/IncrHmmAligModel.h"

#include "TestUtils.h"

#include <gtest/gtest.h>
#include <utility>

using namespace std;

TEST(IncrHmmAligModelTest, train)
{
  IncrHmmAligModel model;
  addTrainingData(model);
  train(model, 2);

  vector<PositionIndex> alignment;
  obtainBestAlignment(model, "isthay isyay ayay esttay-N .", "this is a test N .", alignment);
  EXPECT_EQ(alignment, (vector<PositionIndex>{1, 2, 3, 4, 4, 5}));

  obtainBestAlignment(model, "isthay isyay otnay ayay esttay-N .", "this is not a test N .", alignment);
  EXPECT_EQ(alignment, (vector<PositionIndex>{1, 2, 3, 4, 5, 5, 6}));

  obtainBestAlignment(model, "isthay isyay ayay esttay-N ardhay .", "this is a hard test N .", alignment);
  EXPECT_EQ(alignment, (vector<PositionIndex>{1, 2, 3, 5, 4, 4, 6}));
}

TEST(IncrHmmAligModelTest, incrTrain)
{
  IncrHmmAligModel model;
  addTrainingData(model);
  incrTrain(model, make_pair(0, model.numSentPairs() - 1), 2);

  vector<PositionIndex> alignment;
  obtainBestAlignment(model, "isthay isyay ayay esttay-N .", "this is a test N .", alignment);
  EXPECT_EQ(alignment, (vector<PositionIndex>{1, 2, 3, 4, 4, 5}));

  obtainBestAlignment(model, "isthay isyay otnay ayay esttay-N .", "this is not a test N .", alignment);
  EXPECT_EQ(alignment, (vector<PositionIndex>{1, 2, 3, 4, 5, 5, 6}));

  obtainBestAlignment(model, "isthay isyay ayay esttay-N ardhay .", "this is a hard test N .", alignment);
  EXPECT_EQ(alignment, (vector<PositionIndex>{1, 2, 3, 5, 4, 4, 6}));
}

TEST(IncrHmmAligModelTest, calcLgProbForAlig)
{
  IncrHmmAligModel model;
  addTrainingData(model);
  train(model, 2);

  WordAligMatrix waMatrix;
  LgProb expectedLogProb =
      model.obtainBestAlignmentChar("isthay isyay ayay esttay-N .", "this is a test N .", waMatrix);
  LgProb logProb = model.calcLgProbForAligChar("isthay isyay ayay esttay-N .", "this is a test N .", waMatrix);
  EXPECT_NEAR(logProb, expectedLogProb, EPSILON);
}
