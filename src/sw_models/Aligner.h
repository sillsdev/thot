#pragma once

#include "nlp_common/Prob.h"
#include "nlp_common/WordAlignmentMatrix.h"
#include "nlp_common/WordIndex.h"

#include <vector>

class Aligner
{
public:
  // Obtains the best alignments for the sentence pairs given in
  // the files 'sourceTestFileName' and 'targetTestFilename'. The
  // results are stored in the file 'outFileName'
  virtual LgProb getBestAlignment(const char* srcSentence, const char* trgSentence,
                                  WordAlignmentMatrix& bestWaMatrix) = 0;
  // Obtains the best alignment for the given sentence pair
  virtual LgProb getBestAlignment(const std::vector<std::string>& srcSentence,
                                  const std::vector<std::string>& trgSentence, WordAlignmentMatrix& bestWaMatrix) = 0;
  // Obtains the best alignment for the given sentence pair (input
  // parameters are now string vectors)
  virtual LgProb getBestAlignment(const std::vector<WordIndex>& srcSentence, const std::vector<WordIndex>& trgSentence,
                                  WordAlignmentMatrix& bestWaMatrix) = 0;

  virtual WordIndex stringToSrcWordIndex(std::string s) const = 0;
  virtual std::vector<WordIndex> strVectorToSrcIndexVector(std::vector<std::string> s) = 0;

  virtual WordIndex stringToTrgWordIndex(std::string t) const = 0;
  virtual std::vector<WordIndex> strVectorToTrgIndexVector(std::vector<std::string> t) = 0;

  virtual ~Aligner()
  {
  }
};
