#pragma once

#include "nlp_common/Count.h"
#include "nlp_common/Prob.h"

class BaseSentLengthModel
{
public:
  // Load model parameters
  virtual bool load(const char* filename, int verbose = 0) = 0;

  // Print model parameters
  virtual bool print(const char* filename) = 0;

  // Sentence length model functions
  virtual Prob sentLenProb(unsigned int slen, unsigned int tlen) = 0;
  // returns p(tl=tlen|sl=slen)
  virtual LgProb sentLenLgProb(unsigned int slen, unsigned int tlen) = 0;

  // Sum sentence length model functions
  virtual Prob sumSentLenProb(unsigned int slen, unsigned int tlen) = 0;
  // returns p(tl<=tlen|sl=slen)
  virtual LgProb sumSentLenLgProb(unsigned int slen, unsigned int tlen) = 0;

  // Functions to train the sentence length model
  virtual void trainSentPairRange(std::pair<unsigned int, unsigned int> sentPairRange, int verbosity = 0) = 0;
  virtual void trainSentPair(std::vector<std::string> srcSentVec, std::vector<std::string> trgSentVec, Count c = 1) = 0;

  // clear function
  virtual void clear(void) = 0;

  // Destructor
  virtual ~BaseSentLengthModel(void)
  {
  }
};
