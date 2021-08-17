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

LgProb obtainBestAlignment(AlignmentModel& model, const string& srcSentence, const string& trgSentence,
                           vector<PositionIndex>& alignment)
{
  WordAlignmentMatrix waMatrix;
  LgProb lgProb = model.getBestAlignment(srcSentence.c_str(), trgSentence.c_str(), waMatrix);
  waMatrix.getAligVec(alignment);
  return lgProb;
}
