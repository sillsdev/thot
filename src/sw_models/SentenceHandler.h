#pragma once

#include "nlp_common/Count.h"

#include <string>
#include <vector>

class SentenceHandler
{
public:
  // Functions to read and add sentence pairs
  virtual bool readSentencePairs(const char* srcFileName, const char* trgFileName, const char* sentCountsFile,
                                 std::pair<unsigned int, unsigned int>& sentRange, int verbose = 0) = 0;
  // NOTE: when function readSentencePairs() is invoked, previously
  //       seen sentence pairs are removed

  virtual std::pair<unsigned int, unsigned int> addSentencePair(std::vector<std::string> srcSentStr,
                                                                std::vector<std::string> trgSentStr, Count c,
                                                                int verbose = 0) = 0;
  virtual unsigned int numSentencePairs() = 0;
  // NOTE: the whole valid range in a given moment is
  // [ 0 , numSentPairs() )
  virtual int getSentencePair(unsigned int n, std::vector<std::string>& srcSentStr,
                              std::vector<std::string>& trgSentStr, Count& c) = 0;
  virtual int getSrcSentence(unsigned int n, std::vector<std::string>& srcSentStr) = 0;
  virtual int getTrgSentence(unsigned int n, std::vector<std::string>& trgSentStr) = 0;
  virtual int getCount(unsigned int n, Count& c) = 0;

  // Functions to print sentence pairs
  virtual bool printSentencePairs(const char* srcSentFile, const char* trgSentFile, const char* sentCountsFile) = 0;

  // Clear function
  virtual void clear() = 0;

  // Destructor
  virtual ~SentenceHandler()
  {
  }
};
