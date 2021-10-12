#include "TestUtils.h"

#include "nlp_common/StrProcUtils.h"

using namespace std;

pair<unsigned int, unsigned int> addSentencePair(AlignmentModel& model, const string& srcSentence,
                                                 const string& trgSentence)
{
  vector<string> srcTokens = StrProcUtils::stringToStringVector(srcSentence);
  vector<string> trgTokens = StrProcUtils::stringToStringVector(trgSentence);
  return model.addSentencePair(srcTokens, trgTokens, 1);
}

void addTrainingData(AlignmentModel& model)
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

void addTrainingDataWordClasses(AlignmentModel& model)
{
  // pronouns
  addSrcWordClass(model, "1", {"isthay", "ouyay", "ityay"});
  // verbs
  addSrcWordClass(model, "2", {"isyay", "ouldshay", "orkway-V", "ancay", "ebay", "esttay-V"});
  // articles
  addSrcWordClass(model, "3", {"ayay"});
  // nouns
  addSrcWordClass(model, "4", {"esttay-N", "orkway-N", "ordway"});
  // punctuation
  addSrcWordClass(model, "5", {".", "?", "!"});
  // adverbs
  addSrcWordClass(model, "6", {"oftenyay"});
  // adjectives
  addSrcWordClass(model, "7", {"ardhay", "orkingway"});

  // pronouns
  addTrgWordClass(model, "1", {"this", "you", "it"});
  // verbs
  addTrgWordClass(model, "2", {"is", "should", "can", "be"});
  // articles
  addTrgWordClass(model, "3", {"a"});
  // nouns
  addTrgWordClass(model, "4", {"word"});
  // punctuations
  addTrgWordClass(model, "5", {".", "?", "!"});
  // adverbs
  addTrgWordClass(model, "6", {"often"});
  // adjectives
  addTrgWordClass(model, "7", {"hard", "working"});
  // nouns/verbs
  addTrgWordClass(model, "8", {"test", "work"});
  // disambiguators
  addTrgWordClass(model, "9", {"N", "V"});
}

void addSrcWordClass(AlignmentModel& model, const std::string& c, const std::unordered_set<std::string>& words)
{
  WordClassIndex wordClassIndex = model.addSrcWordClass(c);
  for (auto& w : words)
    model.mapSrcWordToWordClass(model.addSrcSymbol(w), wordClassIndex);
}

void addTrgWordClass(AlignmentModel& model, const std::string& c, const std::unordered_set<std::string>& words)
{
  WordClassIndex wordClassIndex = model.addTrgWordClass(c);
  for (auto& w : words)
    model.mapTrgWordToWordClass(model.addTrgSymbol(w), wordClassIndex);
}

void train(AlignmentModel& model, int numIters)
{
  model.startTraining();
  for (int i = 0; i < numIters; ++i)
    model.train();
  model.endTraining();
}

void incrTrain(IncrAlignmentModel& model, pair<unsigned int, unsigned int> range, int numIters)
{
  model.startIncrTraining(range);
  for (int i = 0; i < numIters; ++i)
    model.incrTrain(range);
  model.endTraining();
}
