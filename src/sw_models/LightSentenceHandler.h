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
                         std::pair<unsigned int, unsigned int>& sentRange, int verbose = 0);
  // NOTE: when function readSentencePairs() is invoked, previously
  //       seen sentence pairs are removed

  void addSentencePair(std::vector<std::string> srcSentStr, std::vector<std::string> trgSentStr, Count c,
                       std::pair<unsigned int, unsigned int>& sentRange);
  unsigned int numSentencePairs();
  // NOTE: the whole valid range in a given moment is
  // [ 0 , numSentPairs() )
  int getSentencePair(unsigned int n, std::vector<std::string>& srcSentStr, std::vector<std::string>& trgSentStr,
                      Count& c);
  int getSrcSentence(unsigned int n, std::vector<std::string>& srcSentStr);
  int getTrgSentence(unsigned int n, std::vector<std::string>& trgSentStr);
  int getCount(unsigned int n, Count& c);

  // Functions to print sentence pairs
  bool printSentencePairs(const char* srcSentFile, const char* trgSentFile, const char* sentCountsFile);

  void clear();

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
