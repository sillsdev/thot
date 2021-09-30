#pragma once

#include "nlp_common/AwkInputStream.h"
#include "sw_models/SentenceHandler.h"

#include <fstream>
#include <string.h>

class LightSentenceHandler : public SentenceHandler
{
public:
  // Constructor
  LightSentenceHandler();

  // Functions to read and add sentence pairs
  bool readSentencePairs(const char* srcFileName, const char* trgFileName, const char* sentCountsFile,
                         std::pair<unsigned int, unsigned int>& sentRange, int verbose = 0) override;
  // NOTE: when function readSentencePairs() is invoked, previously
  //       seen sentence pairs are removed

  std::pair<unsigned int, unsigned int> addSentencePair(std::vector<std::string> srcSentStr,
                                                        std::vector<std::string> trgSentStr, Count c,
                                                        int verbose = 0) override;
  unsigned int numSentencePairs() override;
  // NOTE: the whole valid range in a given moment is
  // [ 0 , numSentPairs() )
  int getSentencePair(unsigned int n, std::vector<std::string>& srcSentStr, std::vector<std::string>& trgSentStr,
                      Count& c) override;
  int getSrcSentence(unsigned int n, std::vector<std::string>& srcSentStr) override;
  int getTrgSentence(unsigned int n, std::vector<std::string>& trgSentStr) override;
  int getCount(unsigned int n, Count& c) override;

  // Functions to print sentence pairs
  bool printSentencePairs(const char* srcSentFile, const char* trgSentFile, const char* sentCountsFile) override;

  void clear() override;

protected:
  AwkInputStream awkSrc;
  AwkInputStream awkTrg;
  AwkInputStream awkSrcTrgC;

  bool countFileExists;
  size_t nsPairsInFiles;
  size_t currFileSentIdx;

  std::vector<std::pair<std::vector<std::string>, std::vector<std::string>>> sentPairCont;
  std::vector<Count> sentPairCount;

  void rewindFiles();
  bool getNextLineFromFiles();
  int nthSentPairFromFiles(unsigned int n, std::vector<std::string>& srcSentStr, std::vector<std::string>& trgSentStr,
                           Count& c);
};
