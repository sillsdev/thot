#include "sw_models/Ibm4AligModel.h"

#include "nlp_common/MathDefs.h"
#include "nlp_common/StrProcUtils.h"
#include "sw_models/SwDefs.h"

#include <gtest/gtest.h>
#include <memory>
#include <unordered_set>

using namespace std;

void addSentencePair(BaseSwAligModel& model, const string& srcSentence, const string& trgSentence)
{
  vector<string> srcTokens = StrProcUtils::stringToStringVector(srcSentence);
  vector<string> trgTokens = StrProcUtils::stringToStringVector(trgSentence);
  pair<unsigned int, unsigned int> range;
  model.addSentPair(srcTokens, trgTokens, 1, range);
}

void addTrainingData(BaseSwAligModel& model)
{
  addSentencePair(model, "isthay isyay ayay esttay-N .", "this is a test N .");
  addSentencePair(model, "ouyay ouldshay esttay-V oftenyay .", "you should test V often .");
  addSentencePair(model, "isyay isthay orkingway ?", "is this working ?");
  addSentencePair(model, "isthay ouldshay orkway-V .", "this should work V .");
  addSentencePair(model, "ityay isyay orkingway .", "it is working .");
  addSentencePair(model, "orkway-N ancay ebay ardhay !", "work N can be hard !");
  addSentencePair(model, "ayay esttay-N ancay ebay ardhay .", "a test N can be hard .");
  addSentencePair(model, "isthay isyay ayay ordway !", "this is a word !");
}

void train(BaseSwAligModel& model, int numIters = 1)
{
  model.startTraining();
  for (int i = 0; i < numIters; ++i)
    model.train();
  model.endTraining();
}

LgProb obtainBestAlignment(BaseSwAligModel& model, const string& srcSentence, const string& trgSentence,
                           vector<PositionIndex>& alignment)
{
  WordAligMatrix waMatrix;
  LgProb lgProb = model.obtainBestAlignmentChar(srcSentence.c_str(), trgSentence.c_str(), waMatrix);
  waMatrix.getAligVec(alignment);
  return lgProb;
}

class Ibm4AligModelTest : public testing::Test
{
protected:
  void createTrainedModel()
  {
    model.reset(new Ibm4AligModel);
    addSrcWordClass(1, {"räucherschinken"});
    addSrcWordClass(2, {"ja"});
    addSrcWordClass(3, {"ich"});
    addSrcWordClass(4, {"esse"});
    addSrcWordClass(5, {"gern"});

    addTrgWordClass(1, {"ham"});
    addTrgWordClass(2, {"smoked"});
    addTrgWordClass(3, {"to"});
    addTrgWordClass(4, {"i"});
    addTrgWordClass(5, {"love", "eat"});

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

    model->p1 = 0.167;
  }

  void addTrainingDataWordClasses()
  {
    // pronouns
    addSrcWordClass(1, {"isthay", "ouyay", "ityay"});
    // verbs
    addSrcWordClass(2, {"isyay", "ouldshay", "orkway-V", "ancay", "ebay", "esttay-V"});
    // articles
    addSrcWordClass(3, {"ayay"});
    // nouns
    addSrcWordClass(4, {"esttay-N", "orkway-N", "ordway"});
    // punctuation
    addSrcWordClass(5, {".", "?", "!"});
    // adverbs
    addSrcWordClass(6, {"oftenyay"});
    // adjectives
    addSrcWordClass(7, {"ardhay", "orkingway"});

    // pronouns
    addTrgWordClass(1, {"this", "you", "it"});
    // verbs
    addTrgWordClass(2, {"is", "should", "can", "be"});
    // articles
    addTrgWordClass(3, {"a"});
    // nouns
    addTrgWordClass(4, {"word"});
    // punctuations
    addTrgWordClass(5, {".", "?", "!"});
    // adverbs
    addTrgWordClass(6, {"often"});
    // adjectives
    addTrgWordClass(7, {"hard", "working"});
    // nouns/verbs
    addTrgWordClass(8, {"test", "work"});
    // disambiguators
    addTrgWordClass(9, {"N", "V"});
  }

  void addSrcWordClass(WordClassIndex c, const unordered_set<string>& words)
  {
    for (auto& w : words)
      model->addSrcWordClass(model->addSrcSymbol(w), c);
  }

  void addTrgWordClass(WordClassIndex c, const unordered_set<string>& words)
  {
    for (auto& w : words)
      model->addTrgWordClass(model->addTrgSymbol(w), c);
  }

  void setHeadDistortionProb(WordClassIndex srcWordClass, WordClassIndex trgWordClass, int dj, double prob)
  {
    model->headDistortionTable->set(srcWordClass, trgWordClass, dj, log(prob), 0);
  }

  void setNonheadDistortionProb(WordClassIndex trgWordClass, int dj, double prob)
  {
    model->nonheadDistortionTable->set(trgWordClass, dj, log(prob), 0);
  }

  void setTranslationProb(const string& s, const string& t, double prob)
  {
    model->lexTable->set(model->addSrcSymbol(s), model->addTrgSymbol(t), log(prob), 0);
  }

  void setFertilityProb(const string& s, PositionIndex phi, double prob)
  {
    model->fertilityTable->set(model->addSrcSymbol(s), phi, log(prob), 0);
  }

  LgProb calcLgProbForAlig(const string& srcSentence, const string& trgSentence, const vector<PositionIndex>& alignment)
  {
    vector<string> srcTokens = StrProcUtils::stringToStringVector(srcSentence);
    vector<string> trgTokens = StrProcUtils::stringToStringVector(trgSentence);
    auto slen = PositionIndex(srcTokens.size());
    auto tlen = PositionIndex(trgTokens.size());
    WordAligMatrix waMatrix{slen, tlen};
    waMatrix.putAligVec(alignment);
    LgProb logProb = model->calcLgProbForAligVecStr(srcTokens, trgTokens, waMatrix);
    logProb -= model->sentLenLgProb(slen, tlen);
    return logProb;
  }

  unique_ptr<Ibm4AligModel> model;
};

TEST_F(Ibm4AligModelTest, obtainBestAlignment)
{
  createTrainedModel();
  vector<PositionIndex> alignment;
  obtainBestAlignment(*model, "ich esse ja gern räucherschinken", "i love to eat smoked ham", alignment);
  EXPECT_EQ(alignment, (vector<PositionIndex>{1, 4, 0, 2, 5, 5}));
}

TEST_F(Ibm4AligModelTest, calcLgProbForAlig)
{
  createTrainedModel();
  model->setDistortionSmoothFactor(0);
  vector<PositionIndex> alignment = {1, 4, 0, 2, 5, 5};
  LgProb logProb = calcLgProbForAlig("ich esse ja gern räucherschinken", "i love to eat smoked ham", alignment);
  EXPECT_NEAR(logProb.get_p(), 0.2905, 0.0001);
}

TEST_F(Ibm4AligModelTest, train)
{
  Ibm1AligModel model1;
  addTrainingData(model1);
  train(model1);

  vector<PositionIndex> alignment;
  obtainBestAlignment(model1, "isthay isyay ayay esttay-N .", "this is a test N .", alignment);
  EXPECT_EQ(alignment, (vector<PositionIndex>{1, 2, 3, 4, 4, 5}));

  obtainBestAlignment(model1, "isthay isyay otnay ayay esttay-N .", "this is not a test N .", alignment);
  EXPECT_EQ(alignment, (vector<PositionIndex>{1, 2, 6, 4, 5, 5, 6}));

  obtainBestAlignment(model1, "isthay isyay ayay esttay-N ardhay .", "this is a hard test N .", alignment);
  EXPECT_EQ(alignment, (vector<PositionIndex>{1, 2, 3, 5, 4, 5, 6}));

  Ibm2AligModel model2{model1};
  train(model2);

  obtainBestAlignment(model2, "isthay isyay ayay esttay-N .", "this is a test N .", alignment);
  EXPECT_EQ(alignment, (vector<PositionIndex>{1, 2, 3, 4, 4, 5}));

  obtainBestAlignment(model2, "isthay isyay otnay ayay esttay-N .", "this is not a test N .", alignment);
  EXPECT_EQ(alignment, (vector<PositionIndex>{1, 2, 5, 4, 5, 5, 6}));

  obtainBestAlignment(model2, "isthay isyay ayay esttay-N ardhay .", "this is a hard test N .", alignment);
  EXPECT_EQ(alignment, (vector<PositionIndex>{1, 2, 3, 5, 4, 4, 6}));

  Ibm3AligModel model3{model2};
  train(model3);

  obtainBestAlignment(model3, "isthay isyay ayay esttay-N .", "this is a test N .", alignment);
  EXPECT_EQ(alignment, (vector<PositionIndex>{1, 2, 3, 4, 4, 5}));

  obtainBestAlignment(model3, "isthay isyay otnay ayay esttay-N .", "this is not a test N .", alignment);
  EXPECT_EQ(alignment, (vector<PositionIndex>{1, 2, 0, 4, 5, 5, 6}));

  obtainBestAlignment(model3, "isthay isyay ayay esttay-N ardhay .", "this is a hard test N .", alignment);
  EXPECT_EQ(alignment, (vector<PositionIndex>{1, 2, 3, 5, 4, 4, 6}));

  model.reset(new Ibm4AligModel{model3});
  addTrainingDataWordClasses();
  train(*model);

  obtainBestAlignment(*model, "isthay isyay ayay esttay-N .", "this is a test N .", alignment);
  EXPECT_EQ(alignment, (vector<PositionIndex>{1, 2, 3, 4, 4, 5}));

  obtainBestAlignment(*model, "isthay isyay otnay ayay esttay-N .", "this is not a test N .", alignment);
  EXPECT_EQ(alignment, (vector<PositionIndex>{1, 2, 3, 4, 5, 5, 6}));

  obtainBestAlignment(*model, "isthay isyay ayay esttay-N ardhay .", "this is a hard test N .", alignment);
  EXPECT_EQ(alignment, (vector<PositionIndex>{1, 2, 3, 5, 4, 4, 6}));
}

TEST_F(Ibm4AligModelTest, headDistortionProbSmoothing)
{
  createTrainedModel();
  Prob prob = model->headDistortionProb(3, 5, 6, 3);
  EXPECT_NEAR(prob, 0.8159, 0.0001);
}

TEST_F(Ibm4AligModelTest, headDistortionProbNoSmoothing)
{
  createTrainedModel();
  model->setDistortionSmoothFactor(0);
  Prob prob = model->headDistortionProb(3, 5, 6, 3);
  EXPECT_NEAR(prob, 0.97, EPSILON);
}

TEST_F(Ibm4AligModelTest, headDistortionProbDefaultSmoothing)
{
  createTrainedModel();
  Prob prob = model->headDistortionProb(3, 5, 6, 2);
  EXPECT_NEAR(prob, 0.04, EPSILON);
}

TEST_F(Ibm4AligModelTest, headDistortionProbDefaultNoSmoothing)
{
  createTrainedModel();
  model->setDistortionSmoothFactor(0);
  Prob prob = model->headDistortionProb(3, 5, 6, 2);
  EXPECT_NEAR(prob, SW_PROB_SMOOTH, EPSILON);
}

TEST_F(Ibm4AligModelTest, nonheadDistortionProbSmoothing)
{
  createTrainedModel();
  Prob prob = model->nonheadDistortionProb(1, 6, 1);
  EXPECT_NEAR(prob, 0.8079, 0.0001);
}

TEST_F(Ibm4AligModelTest, nonheadDistortionProbNoSmoothing)
{
  createTrainedModel();
  model->setDistortionSmoothFactor(0);
  Prob prob = model->nonheadDistortionProb(1, 6, 1);
  EXPECT_NEAR(prob, 0.96, EPSILON);
}

TEST_F(Ibm4AligModelTest, nonheadDistortionProbDefaultSmoothing)
{
  createTrainedModel();
  Prob prob = model->nonheadDistortionProb(1, 6, 0);
  EXPECT_NEAR(prob, 0.04, EPSILON);
}

TEST_F(Ibm4AligModelTest, nonheadDistortionProbDefaultNoSmoothing)
{
  createTrainedModel();
  model->setDistortionSmoothFactor(0);
  Prob prob = model->nonheadDistortionProb(1, 6, 0);
  EXPECT_NEAR(prob, SW_PROB_SMOOTH, EPSILON);
}
