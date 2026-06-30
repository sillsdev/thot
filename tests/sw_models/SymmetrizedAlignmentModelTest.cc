#include "sw_models/SymmetrizedAlignmentModel.h"

#include "TestUtils.h"
#include "nlp_common/ErrorDefs.h"
#include "nlp_common/MathDefs.h"
#include "sw_models/FastAlignModel.h"

#include <gtest/gtest.h>

#include <memory>

using namespace std;

namespace
{
// Builds a direct model (trained on the test corpus) and an inverse model trained
// on the same pairs with source/target swapped, both emitting training
// alignments, then wraps them in a SymmetrizedAlignmentModel.
shared_ptr<SymmetrizedAlignmentModel> buildSymmetrizedModel(shared_ptr<FastAlignModel>& direct,
                                                            shared_ptr<FastAlignModel>& inverse)
{
  direct = make_shared<FastAlignModel>();
  direct->setEmitTrainingAlignments(true);
  addTrainingData(*direct);
  train(*direct, 2);

  inverse = make_shared<FastAlignModel>();
  inverse->setEmitTrainingAlignments(true);
  for (unsigned int n = 0; n < direct->numSentencePairs(); ++n)
  {
    vector<string> src, trg;
    Count c;
    direct->getSentencePair(n, src, trg, c);
    inverse->addSentencePair(trg, src, c); // swapped
  }
  train(*inverse, 2);

  return make_shared<SymmetrizedAlignmentModel>(direct, inverse);
}
} // namespace

TEST(SymmetrizedAlignmentModelTest, sentencePairAccessors)
{
  shared_ptr<FastAlignModel> direct, inverse;
  shared_ptr<SymmetrizedAlignmentModel> model = buildSymmetrizedModel(direct, inverse);

  EXPECT_EQ(model->numSentencePairs(), direct->numSentencePairs());

  // get_sentence_pair returns the direct model's pairs, in src->trg order (not
  // the swapped inverse order).
  for (unsigned int n = 0; n < model->numSentencePairs(); ++n)
  {
    vector<string> src, trg, expectedSrc, expectedTrg;
    Count c, expectedC;
    direct->getSentencePair(n, expectedSrc, expectedTrg, expectedC);
    model->getSentencePair(n, src, trg, c);
    EXPECT_EQ(src, expectedSrc);
    EXPECT_EQ(trg, expectedTrg);
  }
}

TEST(SymmetrizedAlignmentModelTest, heuristicNoneReturnsDirect)
{
  shared_ptr<FastAlignModel> direct, inverse;
  shared_ptr<SymmetrizedAlignmentModel> model = buildSymmetrizedModel(direct, inverse);
  model->setHeuristic(SymmetrizationHeuristic::None);

  for (unsigned int n = 0; n < model->numSentencePairs(); ++n)
  {
    vector<PositionIndex> expected;
    LgProb expectedProb = direct->getTrainingAlignment(n, expected);

    vector<PositionIndex> alig;
    LgProb prob = model->getTrainingAlignment(n, alig);
    EXPECT_EQ(alig, expected);
    EXPECT_NEAR((double)prob, (double)expectedProb, EPSILON);
  }
}

TEST(SymmetrizedAlignmentModelTest, combinesDirectAndInverse)
{
  shared_ptr<FastAlignModel> direct, inverse;
  shared_ptr<SymmetrizedAlignmentModel> model = buildSymmetrizedModel(direct, inverse);

  for (SymmetrizationHeuristic h :
       {SymmetrizationHeuristic::Union, SymmetrizationHeuristic::Intersection, SymmetrizationHeuristic::Och,
        SymmetrizationHeuristic::GrowDiagFinalAnd})
  {
    model->setHeuristic(h);
    for (unsigned int n = 0; n < model->numSentencePairs(); ++n)
    {
      // Reconstruct the expected combination directly from the two underlying
      // training alignments: direct (src x trg) and inverse transposed (src x trg).
      WordAlignmentMatrix expected;
      LgProb directProb = direct->getTrainingAlignment(n, expected);
      WordAlignmentMatrix invMatrix;
      LgProb invProb = inverse->getTrainingAlignment(n, invMatrix);
      invMatrix.transpose();
      switch (h)
      {
      case SymmetrizationHeuristic::Union:
        expected |= invMatrix;
        break;
      case SymmetrizationHeuristic::Intersection:
        expected &= invMatrix;
        break;
      case SymmetrizationHeuristic::Och:
        expected.symmetr1(invMatrix);
        break;
      case SymmetrizationHeuristic::GrowDiagFinalAnd:
        expected.growDiagFinalAnd(invMatrix);
        break;
      default:
        break;
      }

      WordAlignmentMatrix actual;
      LgProb prob = model->getTrainingAlignment(n, actual);
      EXPECT_TRUE(actual == expected);
      EXPECT_NEAR((double)prob, (double)max(directProb, invProb), EPSILON);

      // The per-target-vector overload agrees with the matrix overload.
      vector<PositionIndex> aligVec, expectedVec;
      model->getTrainingAlignment(n, aligVec);
      expected.getAligVec(expectedVec);
      EXPECT_EQ(aligVec, expectedVec);
    }
  }
}

TEST(SymmetrizedAlignmentModelTest, outOfRangeIndex)
{
  shared_ptr<FastAlignModel> direct, inverse;
  shared_ptr<SymmetrizedAlignmentModel> model = buildSymmetrizedModel(direct, inverse);
  model->setHeuristic(SymmetrizationHeuristic::GrowDiagFinalAnd);

  vector<PositionIndex> oob{1, 2, 3};
  LgProb prob = model->getTrainingAlignment(1000, oob);
  EXPECT_TRUE(oob.empty());
  EXPECT_EQ((double)prob, (double)SMALL_LG_NUM);
}
