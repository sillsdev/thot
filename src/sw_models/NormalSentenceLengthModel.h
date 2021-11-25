#pragma once

#include "nlp_common/AwkInputStream.h"
#include "nlp_common/MathFuncs.h"
#include "sw_models/SentenceLengthModelBase.h"

#include <map>
#include <string.h>
#include <utility>

/*
 * Implements a weighted incremental gaussian sentence length model
 */
class NormalSentenceLengthModel : public SentenceLengthModelBase
{
public:
  // Load model parameters
  bool load(const char* filename, int verbose = 0) override;

  // Print model parameters
  bool print(const char* filename) override;

  // Sentence length model functions

  // returns p(tl=tlen|sl=slen)
  Prob sentenceLengthProb(unsigned int slen, unsigned int tlen) override;
  LgProb sentenceLengthLogProb(unsigned int slen, unsigned int tlen) override;

  // Sum sentence length model functions

  // returns p(tl<=tlen|sl=slen)
  Prob sumSentenceLengthProb(unsigned int slen, unsigned int tlen) override;
  LgProb sumSentenceLengthLogProb(unsigned int slen, unsigned int tlen) override;

  // Functions to train the sentence length model
  using SentenceLengthModelBase::trainSentencePair;
  void trainSentencePair(unsigned int slen, unsigned int tlen, Count c = 1) override;

  void clear() override;

protected:
  unsigned int numSents = 0;
  unsigned int slenSum = 0;
  unsigned int tlenSum = 0;
  std::vector<unsigned int> kVec;
  std::vector<float> swkVec;
  std::vector<float> mkVec;
  std::vector<float> skVec;

  // Auxiliary functions
  std::ostream& print(std::ostream& outS);
  LgProb sentLenLgProbNorm(unsigned int slen, unsigned int tlen);
  Prob sumSentLenProbNorm(unsigned int slen, unsigned int tlen);
  bool readNormalPars(const char* normParsFileName, int verbose);
  bool get_mean_stddev(unsigned int slen, float& mean, float& stddev);
  unsigned int get_k(unsigned int slen, bool& found);
  void set_k(unsigned int slen, unsigned int k_val);
  float get_swk(unsigned int slen);
  void set_swk(unsigned int slen, float swk_val);
  float get_mk(unsigned int slen);
  void set_mk(unsigned int slen, float mk_val);
  float get_sk(unsigned int slen);
  void set_sk(unsigned int slen, float sk_val);
};
