
#include "sw_models/SentenceLengthModelBase.h"

void SentenceLengthModelBase::linkVocabPtr(SingleWordVocab* _swVocabPtr)
{
  swVocabPtr = _swVocabPtr;
}

void SentenceLengthModelBase::linkSentPairInfo(SentenceHandler* _sentenceHandlerPtr)
{
  sentenceHandlerPtr = _sentenceHandlerPtr;
}

void SentenceLengthModelBase::trainSentencePairRange(std::pair<unsigned int, unsigned int> sentPairRange,
                                                     int /*verbosity=0*/)
{
  // Iterate over the training samples
  for (unsigned int n = sentPairRange.first; n <= sentPairRange.second; ++n)
  {
    std::vector<std::string> srcSentStrVec;
    std::vector<std::string> trgSentStrVec;
    Count c;
    getSentencePair(n, srcSentStrVec, trgSentStrVec, c);

    if (!srcSentStrVec.empty() && !trgSentStrVec.empty())
      trainSentencePair(srcSentStrVec, trgSentStrVec, c);
  }
}

int SentenceLengthModelBase::getSentencePair(unsigned int n, std::vector<std::string>& srcSentStr,
                                             std::vector<std::string>& trgSentStr, Count& c)
{
  return sentenceHandlerPtr->getSentencePair(n, srcSentStr, trgSentStr, c);
}

void SentenceLengthModelBase::trainSentencePair(std::vector<std::string> srcSentVec,
                                                std::vector<std::string> trgSentVec, Count c)
{
  trainSentencePair(srcSentVec.size(), trgSentVec.size(), c);
}
