#pragma once

#include "nlp_common/Count.h"
#include "nlp_common/Prob.h"

#include <vector>

class SentenceLengthModel
{
public:
  // Load model parameters
  virtual bool load(const char* filename, int verbose = 0) = 0;

  // Print model parameters
  virtual bool print(const char* filename) = 0;

  // Sentence length model functions

  // returns p(tl=tlen|sl=slen)
  virtual Prob sentenceLengthProb(unsigned int slen, unsigned int tlen) = 0;
  virtual LgProb sentenceLengthLogProb(unsigned int slen, unsigned int tlen) = 0;

  // Sum sentence length model functions

  // returns p(tl<=tlen|sl=slen)
  virtual Prob sumSentenceLengthProb(unsigned int slen, unsigned int tlen) = 0;
  virtual LgProb sumSentenceLengthLogProb(unsigned int slen, unsigned int tlen) = 0;

  // Functions to train the sentence length model
  virtual void trainSentencePairRange(std::pair<unsigned int, unsigned int> sentPairRange, int verbosity = 0) = 0;
  virtual void trainSentencePair(std::vector<std::string> srcSentVec, std::vector<std::string> trgSentVec,
                                 Count c = 1) = 0;
  virtual void trainSentencePair(unsigned int slen, unsigned int tlen, Count c = 1) = 0;

  // clear function
  virtual void clear() = 0;

  // Destructor
  virtual ~SentenceLengthModel()
  {
  }
};
