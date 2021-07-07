#pragma once

#include "nlp_common/Count.h"

#include <string>
#include <vector>

class BaseSentenceHandler
{
public:
  // Functions to read and add sentence pairs
  virtual bool readSentencePairs(const char* srcFileName, const char* trgFileName, const char* sentCountsFile,
                                 std::pair<unsigned int, unsigned int>& sentRange, int verbose = 0) = 0;
  // NOTE: when function readSentencePairs() is invoked, previously
  //       seen sentence pairs are removed

  virtual void addSentPair(std::vector<std::string> srcSentStr, std::vector<std::string> trgSentStr, Count c,
                           std::pair<unsigned int, unsigned int>& sentRange) = 0;
  virtual unsigned int numSentPairs(void) = 0;
  // NOTE: the whole valid range in a given moment is
  // [ 0 , numSentPairs() )
  virtual int nthSentPair(unsigned int n, std::vector<std::string>& srcSentStr, std::vector<std::string>& trgSentStr,
                          Count& c) = 0;
  virtual int getSrcSent(unsigned int n, std::vector<std::string>& srcSentStr) = 0;
  virtual int getTrgSent(unsigned int n, std::vector<std::string>& trgSentStr) = 0;
  virtual int getCount(unsigned int n, Count& c) = 0;

  // Functions to print sentence pairs
  virtual bool printSentPairs(const char* srcSentFile, const char* trgSentFile, const char* sentCountsFile) = 0;

  // Clear function
  virtual void clear(void) = 0;

  // Destructor
  virtual ~BaseSentenceHandler()
  {
  }
};
