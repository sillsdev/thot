#include "sw_models/Ibm4AligModel.h"

#include "nlp_common/MathDefs.h"
#include "nlp_common/StrProcUtils.h"

#include <gtest/gtest.h>
#include <unordered_set>

using namespace std;

class Ibm4AligModelTest : public testing::Test
{
protected:
  void addSentencePairs()
  {
    addSentencePair("isthay isyay ayay esttay-N .", "this is a test N .");
    addSentencePair("ouyay ouldshay esttay-V oftenyay .", "you should test V often .");
    addSentencePair("isyay isthay orkingway ?", "is this working ?");
    addSentencePair("isthay ouldshay orkway-V .", "this should work V .");
    addSentencePair("ityay isyay orkingway .", "it is working .");
    addSentencePair("orkway-N ancay ebay ardhay !", "work N can be hard !");
    addSentencePair("ayay esttay-N ancay ebay ardhay .", "a test N can be hard .");
    addSentencePair("isthay isyay ayay ordway !", "this is a word !");

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

  void initTables()
  {
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

    model_.p1 = 0.167;
  }

  void addSrcWordClass(WordClassIndex c, const unordered_set<string>& words)
  {
    for (auto& w : words)
      model_.addSrcWordClass(model_.addSrcSymbol(w), c);
  }

  void addTrgWordClass(WordClassIndex c, const unordered_set<string>& words)
  {
    for (auto& w : words)
      model_.addTrgWordClass(model_.addTrgSymbol(w), c);
  }

  void setHeadDistortionProb(WordClassIndex srcWordClass, WordClassIndex trgWordClass, int dj, double prob)
  {
    model_.headDistortionTable.set(srcWordClass, trgWordClass, dj, log(prob), 0);
  }

  void setNonheadDistortionProb(WordClassIndex trgWordClass, int dj, double prob)
  {
    model_.nonheadDistortionTable.set(trgWordClass, dj, log(prob), 0);
  }

  void setTranslationProb(const string& s, const string& t, double prob)
  {
    model_.lexTable.setLexNumDen(model_.addSrcSymbol(s), model_.addTrgSymbol(t), log(prob), 0);
  }

  void setFertilityProb(const string& s, PositionIndex phi, double prob)
  {
    model_.fertilityTable.set(model_.addSrcSymbol(s), phi, log(prob), 0);
  }

  void addSentencePair(const string& srcSentence, const string& trgSentence)
  {
    vector<string> srcTokens = StrProcUtils::stringToStringVector(srcSentence);
    vector<string> trgTokens = StrProcUtils::stringToStringVector(trgSentence);
    pair<unsigned int, unsigned int> range;
    model_.addSentPair(srcTokens, trgTokens, 1, range);
  }

  LgProb obtainBestAlignment(const string& srcSentence, const string& trgSentence, vector<PositionIndex>& alignment)
  {
    WordAligMatrix waMatrix;
    LgProb lgProb = model_.obtainBestAlignmentChar(srcSentence.c_str(), trgSentence.c_str(), waMatrix);
    waMatrix.getAligVec(alignment);
    return lgProb;
  }

  LgProb calcLgProbForAlig(const string& srcSentence, const string& trgSentence, const vector<PositionIndex>& alignment)
  {
    vector<string> srcTokens = StrProcUtils::stringToStringVector(srcSentence);
    vector<string> trgTokens = StrProcUtils::stringToStringVector(trgSentence);
    auto slen = PositionIndex(srcTokens.size());
    auto tlen = PositionIndex(trgTokens.size());
    WordAligMatrix waMatrix{slen, tlen};
    waMatrix.putAligVec(alignment);
    LgProb logProb = model_.calcLgProbForAligVecStr(srcTokens, trgTokens, waMatrix);
    logProb -= model_.sentLenLgProb(slen, tlen);
    return logProb;
  }

  Ibm4AligModel model_;
};

TEST_F(Ibm4AligModelTest, obtainBestAlignment)
{
  initTables();
  vector<PositionIndex> alignment;
  obtainBestAlignment("ich esse ja gern räucherschinken", "i love to eat smoked ham", alignment);
  EXPECT_EQ(alignment, (vector<PositionIndex>{1, 4, 0, 2, 5, 5}));
}

TEST_F(Ibm4AligModelTest, calcLgProbForAlig)
{
  model_.setDistortionSmoothFactor(0);
  initTables();
  vector<PositionIndex> alignment = {1, 4, 0, 2, 5, 5};
  LgProb logProb = calcLgProbForAlig("ich esse ja gern räucherschinken", "i love to eat smoked ham", alignment);
  EXPECT_NEAR(logProb.get_p(), 0.2905, 0.0001);
}

TEST_F(Ibm4AligModelTest, trainAllSents)
{
  addSentencePairs();
  model_.trainAllSents();
  vector<PositionIndex> alignment;
  obtainBestAlignment("isthay isyay ayay esttay-N .", "this is a test N .", alignment);
  EXPECT_EQ(alignment, (vector<PositionIndex>{1, 2, 3, 4, 4, 5}));

  obtainBestAlignment("isthay isyay otnay ayay esttay-N .", "this is not a test N .", alignment);
  EXPECT_EQ(alignment, (vector<PositionIndex>{1, 2, 0, 4, 5, 5, 6}));

  obtainBestAlignment("isthay isyay ayay esttay-N ardhay .", "this is a hard test N .", alignment);
  EXPECT_EQ(alignment, (vector<PositionIndex>{1, 2, 3, 5, 4, 4, 6}));
}

TEST_F(Ibm4AligModelTest, headDistortionProbSmoothing)
{
  initTables();
  Prob prob = model_.headDistortionProb(3, 5, 6, 3);
  EXPECT_NEAR(prob, 0.8159, 0.0001);
}

TEST_F(Ibm4AligModelTest, headDistortionProbNoSmoothing)
{
  model_.setDistortionSmoothFactor(0);
  initTables();
  Prob prob = model_.headDistortionProb(3, 5, 6, 3);
  EXPECT_NEAR(prob, 0.97, EPSILON);
}

TEST_F(Ibm4AligModelTest, headDistortionProbDefaultSmoothing)
{
  initTables();
  Prob prob = model_.headDistortionProb(3, 5, 6, 2);
  EXPECT_NEAR(prob, 0.04, EPSILON);
}

TEST_F(Ibm4AligModelTest, headDistortionProbDefaultNoSmoothing)
{
  model_.setDistortionSmoothFactor(0);
  initTables();
  Prob prob = model_.headDistortionProb(3, 5, 6, 2);
  EXPECT_NEAR(prob, SW_PROB_SMOOTH, EPSILON);
}

TEST_F(Ibm4AligModelTest, nonheadDistortionProbSmoothing)
{
  initTables();
  Prob prob = model_.nonheadDistortionProb(1, 6, 1);
  EXPECT_NEAR(prob, 0.8079, 0.0001);
}

TEST_F(Ibm4AligModelTest, nonheadDistortionProbNoSmoothing)
{
  model_.setDistortionSmoothFactor(0);
  initTables();
  Prob prob = model_.nonheadDistortionProb(1, 6, 1);
  EXPECT_NEAR(prob, 0.96, EPSILON);
}

TEST_F(Ibm4AligModelTest, nonheadDistortionProbDefaultSmoothing)
{
  initTables();
  Prob prob = model_.nonheadDistortionProb(1, 6, 0);
  EXPECT_NEAR(prob, 0.04, EPSILON);
}

TEST_F(Ibm4AligModelTest, nonheadDistortionProbDefaultNoSmoothing)
{
  model_.setDistortionSmoothFactor(0);
  initTables();
  Prob prob = model_.nonheadDistortionProb(1, 6, 0);
  EXPECT_NEAR(prob, SW_PROB_SMOOTH, EPSILON);
}
