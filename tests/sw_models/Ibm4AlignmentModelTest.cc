#include "sw_models/Ibm4AlignmentModel.h"

#include "TestUtils.h"
#include "nlp_common/MathDefs.h"
#include "nlp_common/StrProcUtils.h"
#include "sw_models/SwDefs.h"

#include <gtest/gtest.h>
#include <memory>

class Ibm4AlignmentModelTest : public testing::Test
{
protected:
  void createTrainedModel()
  {
    model.reset(new Ibm4AlignmentModel);
    addSrcWordClass(*model, "1", {"räucherschinken"});
    addSrcWordClass(*model, "2", {"ja"});
    addSrcWordClass(*model, "3", {"ich"});
    addSrcWordClass(*model, "4", {"esse"});
    addSrcWordClass(*model, "5", {"gern"});

    addTrgWordClass(*model, "1", {"ham"});
    addTrgWordClass(*model, "2", {"smoked"});
    addTrgWordClass(*model, "3", {"to"});
    addTrgWordClass(*model, "4", {"i"});
    addTrgWordClass(*model, "5", {"love", "eat"});

    setHeadDistortionProb(NULL_WORD_CLASS, 4, 1, 0.97);
    setHeadDistortionProb(3, 5, 3, 0.97);
    setHeadDistortionProb(4, 5, -2, 0.97);
    setHeadDistortionProb(5, 2, 3, 0.97);

    setNonheadDistortionProb(1, 1, 0.96);

    setTranslationProb("ich", "i", 0.98);
    setTranslationProb("gern", "love", 0.98);
    setTranslationProb(NULL_WORD_STR, "to", 0.98);
    setTranslationProb("esse", "eat", 0.98);
    setTranslationProb("räucherschinken", "smoked", 0.98);
    setTranslationProb("räucherschinken", "ham", 0.98);

    setFertilityProb("ich", 1, 0.99);
    setFertilityProb("esse", 1, 0.99);
    setFertilityProb("ja", 0, 0.99);
    setFertilityProb("gern", 1, 0.99);
    setFertilityProb("räucherschinken", 2, 0.999);
    setFertilityProb(NULL_WORD_STR, 1, 0.99);

    *model->p1 = 0.167;
  }

  void setHeadDistortionProb(WordClassIndex srcWordClass, WordClassIndex trgWordClass, int dj, double prob)
  {
    model->headDistortionTable->set(srcWordClass, trgWordClass, dj, log(prob), 0);
  }

  void setNonheadDistortionProb(WordClassIndex trgWordClass, int dj, double prob)
  {
    model->nonheadDistortionTable->set(trgWordClass, dj, log(prob), 0);
  }

  void setTranslationProb(const std::string& s, const std::string& t, double prob)
  {
    model->lexTable->set(model->addSrcSymbol(s), model->addTrgSymbol(t), log(prob), 0);
  }

  void setFertilityProb(const std::string& s, PositionIndex phi, double prob)
  {
    model->fertilityTable->set(model->addSrcSymbol(s), phi, log(prob), 0);
  }

  LgProb computeLogProb(const std::string& srcSentence, const std::string& trgSentence,
                        const std::vector<PositionIndex>& alignment)
  {
    std::vector<std::string> srcTokens = StrProcUtils::stringToStringVector(srcSentence);
    std::vector<std::string> trgTokens = StrProcUtils::stringToStringVector(trgSentence);
    auto slen = PositionIndex(srcTokens.size());
    auto tlen = PositionIndex(trgTokens.size());
    WordAlignmentMatrix waMatrix{slen, tlen};
    waMatrix.putAligVec(alignment);
    LgProb logProb = model->computeLogProb(srcTokens, trgTokens, waMatrix);
    logProb -= model->sentenceLengthLogProb(slen, tlen);
    return logProb;
  }

  std::unique_ptr<Ibm4AlignmentModel> model;
};

TEST_F(Ibm4AlignmentModelTest, getBestAlignment)
{
  createTrainedModel();
  std::vector<PositionIndex> alignment;
  model->getBestAlignment("ich esse ja gern räucherschinken", "i love to eat smoked ham", alignment);
  EXPECT_EQ(alignment, (std::vector<PositionIndex>{1, 4, 0, 2, 5, 5}));
}

TEST_F(Ibm4AlignmentModelTest, computeLogProb)
{
  createTrainedModel();
  model->setDistortionSmoothFactor(0);
  std::vector<PositionIndex> alignment = {1, 4, 0, 2, 5, 5};
  LgProb logProb = computeLogProb("ich esse ja gern räucherschinken", "i love to eat smoked ham", alignment);
  EXPECT_NEAR(logProb.get_p(), 0.2905, 0.0001);
}

TEST_F(Ibm4AlignmentModelTest, trainIbm2)
{
  Ibm1AlignmentModel model1;
  addTrainingDataWordClasses(model1);
  addTrainingData(model1);
  train(model1, 2);

  std::vector<PositionIndex> alignment;
  model1.getBestAlignment("isthay isyay ayay esttay-N .", "this is a test N .", alignment);
  EXPECT_EQ(alignment, (std::vector<PositionIndex>{1, 2, 3, 4, 4, 5}));

  model1.getBestAlignment("isthay isyay otnay ayay esttay-N .", "this is not a test N .", alignment);
  EXPECT_EQ(alignment, (std::vector<PositionIndex>{1, 2, 0, 4, 5, 5, 6}));

  model1.getBestAlignment("isthay isyay ayay esttay-N ardhay .", "this is a hard test N .", alignment);
  EXPECT_EQ(alignment, (std::vector<PositionIndex>{1, 2, 3, 5, 4, 4, 6}));

  Ibm2AlignmentModel model2{model1};
  train(model2, 2);

  model2.getBestAlignment("isthay isyay ayay esttay-N .", "this is a test N .", alignment);
  EXPECT_EQ(alignment, (std::vector<PositionIndex>{1, 2, 3, 4, 4, 5}));

  model2.getBestAlignment("isthay isyay otnay ayay esttay-N .", "this is not a test N .", alignment);
  EXPECT_EQ(alignment, (std::vector<PositionIndex>{1, 2, 2, 4, 5, 5, 6}));

  model2.getBestAlignment("isthay isyay ayay esttay-N ardhay .", "this is a hard test N .", alignment);
  EXPECT_EQ(alignment, (std::vector<PositionIndex>{1, 2, 3, 5, 4, 4, 6}));

  Ibm3AlignmentModel model3{model2};
  train(model3, 2);

  model3.getBestAlignment("isthay isyay ayay esttay-N .", "this is a test N .", alignment);
  EXPECT_EQ(alignment, (std::vector<PositionIndex>{1, 2, 3, 4, 4, 5}));

  model3.getBestAlignment("isthay isyay otnay ayay esttay-N .", "this is not a test N .", alignment);
  EXPECT_EQ(alignment, (std::vector<PositionIndex>{1, 2, 0, 4, 3, 5, 6}));

  model3.getBestAlignment("isthay isyay ayay esttay-N ardhay .", "this is a hard test N .", alignment);
  EXPECT_EQ(alignment, (std::vector<PositionIndex>{1, 2, 3, 5, 4, 0, 6}));

  model.reset(new Ibm4AlignmentModel{model3});
  train(*model, 2);

  model->getBestAlignment("isthay isyay ayay esttay-N .", "this is a test N .", alignment);
  EXPECT_EQ(alignment, (std::vector<PositionIndex>{1, 2, 3, 4, 4, 5}));

  model->getBestAlignment("isthay isyay otnay ayay esttay-N .", "this is not a test N .", alignment);
  EXPECT_EQ(alignment, (std::vector<PositionIndex>{1, 2, 0, 4, 5, 5, 6}));

  model->getBestAlignment("isthay isyay ayay esttay-N ardhay .", "this is a hard test N .", alignment);
  EXPECT_EQ(alignment, (std::vector<PositionIndex>{1, 2, 3, 5, 4, 4, 6}));
}

TEST_F(Ibm4AlignmentModelTest, trainHmm)
{
  Ibm1AlignmentModel model1;
  addTrainingDataWordClasses(model1);
  addTrainingData(model1);
  train(model1, 2);

  std::vector<PositionIndex> alignment;
  model1.getBestAlignment("isthay isyay ayay esttay-N .", "this is a test N .", alignment);
  EXPECT_EQ(alignment, (std::vector<PositionIndex>{1, 2, 3, 4, 4, 5}));

  model1.getBestAlignment("isthay isyay otnay ayay esttay-N .", "this is not a test N .", alignment);
  EXPECT_EQ(alignment, (std::vector<PositionIndex>{1, 2, 0, 4, 5, 5, 6}));

  model1.getBestAlignment("isthay isyay ayay esttay-N ardhay .", "this is a hard test N .", alignment);
  EXPECT_EQ(alignment, (std::vector<PositionIndex>{1, 2, 3, 5, 4, 4, 6}));

  HmmAlignmentModel modelHmm{model1};
  modelHmm.setHmmP0(0.1);
  train(modelHmm, 2);

  modelHmm.getBestAlignment("isthay isyay ayay esttay-N .", "this is a test N .", alignment);
  EXPECT_EQ(alignment, (std::vector<PositionIndex>{1, 2, 3, 4, 4, 5}));

  modelHmm.getBestAlignment("isthay isyay otnay ayay esttay-N .", "this is not a test N .", alignment);
  EXPECT_EQ(alignment, (std::vector<PositionIndex>{1, 2, 3, 4, 5, 5, 6}));

  modelHmm.getBestAlignment("isthay isyay ayay esttay-N ardhay .", "this is a hard test N .", alignment);
  EXPECT_EQ(alignment, (std::vector<PositionIndex>{1, 2, 3, 5, 4, 4, 4}));

  Ibm3AlignmentModel model3{modelHmm};
  train(model3, 2);

  model3.getBestAlignment("isthay isyay ayay esttay-N .", "this is a test N .", alignment);
  EXPECT_EQ(alignment, (std::vector<PositionIndex>{1, 2, 3, 4, 4, 5}));

  model3.getBestAlignment("isthay isyay otnay ayay esttay-N .", "this is not a test N .", alignment);
  EXPECT_EQ(alignment, (std::vector<PositionIndex>{1, 2, 3, 4, 5, 5, 6}));

  model3.getBestAlignment("isthay isyay ayay esttay-N ardhay .", "this is a hard test N .", alignment);
  EXPECT_EQ(alignment, (std::vector<PositionIndex>{1, 2, 3, 5, 4, 0, 6}));

  model.reset(new Ibm4AlignmentModel{model3});
  train(*model, 2);

  model->getBestAlignment("isthay isyay ayay esttay-N .", "this is a test N .", alignment);
  EXPECT_EQ(alignment, (std::vector<PositionIndex>{1, 2, 3, 4, 4, 5}));

  model->getBestAlignment("isthay isyay otnay ayay esttay-N .", "this is not a test N .", alignment);
  EXPECT_EQ(alignment, (std::vector<PositionIndex>{1, 2, 0, 4, 5, 5, 6}));

  model->getBestAlignment("isthay isyay ayay esttay-N ardhay .", "this is a hard test N .", alignment);
  EXPECT_EQ(alignment, (std::vector<PositionIndex>{1, 2, 3, 5, 4, 4, 6}));
}

TEST_F(Ibm4AlignmentModelTest, headDistortionProbSmoothing)
{
  createTrainedModel();
  Prob prob = model->headDistortionProb(3, 5, 6, 3);
  EXPECT_NEAR(prob, 0.7942, 0.0001);
}

TEST_F(Ibm4AlignmentModelTest, headDistortionProbNoSmoothing)
{
  createTrainedModel();
  model->setDistortionSmoothFactor(0);
  Prob prob = model->headDistortionProb(3, 5, 6, 3);
  EXPECT_NEAR(prob, 0.97, EPSILON);
}

TEST_F(Ibm4AlignmentModelTest, headDistortionProbDefaultSmoothing)
{
  createTrainedModel();
  Prob prob = model->headDistortionProb(3, 5, 6, 2);
  EXPECT_NEAR(prob, 0.0182, 0.0001);
}

TEST_F(Ibm4AlignmentModelTest, headDistortionProbDefaultNoSmoothing)
{
  createTrainedModel();
  model->setDistortionSmoothFactor(0);
  Prob prob = model->headDistortionProb(3, 5, 6, 2);
  EXPECT_NEAR(prob, SW_PROB_SMOOTH, EPSILON);
}

TEST_F(Ibm4AlignmentModelTest, nonheadDistortionProbSmoothing)
{
  createTrainedModel();
  Prob prob = model->nonheadDistortionProb(1, 6, 1);
  EXPECT_NEAR(prob, 0.8079, 0.0001);
}

TEST_F(Ibm4AlignmentModelTest, nonheadDistortionProbNoSmoothing)
{
  createTrainedModel();
  model->setDistortionSmoothFactor(0);
  Prob prob = model->nonheadDistortionProb(1, 6, 1);
  EXPECT_NEAR(prob, 0.96, EPSILON);
}

TEST_F(Ibm4AlignmentModelTest, nonheadDistortionProbDefaultSmoothing)
{
  createTrainedModel();
  Prob prob = model->nonheadDistortionProb(1, 6, 0);
  EXPECT_NEAR(prob, 0.04, EPSILON);
}

TEST_F(Ibm4AlignmentModelTest, nonheadDistortionProbDefaultNoSmoothing)
{
  createTrainedModel();
  model->setDistortionSmoothFactor(0);
  Prob prob = model->nonheadDistortionProb(1, 6, 0);
  EXPECT_NEAR(prob, SW_PROB_SMOOTH, EPSILON);
}
