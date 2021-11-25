
#pragma once

#include "nlp_common/SingleWordVocab.h"
#include "sw_models/SentenceHandler.h"
#include "sw_models/SentenceLengthModel.h"

class SentenceLengthModelBase : public SentenceLengthModel
{
public:
  void linkVocabPtr(SingleWordVocab* _swVocabPtr);
  void linkSentPairInfo(SentenceHandler* _sentenceHandlerPtr);
  void trainSentencePairRange(std::pair<unsigned int, unsigned int> sentPairRange, int verbosity = 0) override;
  int getSentencePair(unsigned int n, std::vector<std::string>& srcSentStr, std::vector<std::string>& trgSentStr,
                      Count& c);
  using SentenceLengthModel::trainSentencePair;
  void trainSentencePair(std::vector<std::string> srcSentVec, std::vector<std::string> trgSentVec,
                         Count c = 1) override;

protected:
  SingleWordVocab* swVocabPtr = NULL;
  SentenceHandler* sentenceHandlerPtr = NULL;
};
