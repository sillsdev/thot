/*
thot package for statistical machine translation
Copyright (C) 2013 Daniel Ortiz-Mart\'inez

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public License
as published by the Free Software Foundation; either version 3
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program; If not, see <http://www.gnu.org/licenses/>.
*/

/**
 * @file Ibm1AligModel.h
 *
 * @brief Defines the Ibm1AligModel class. Ibm1AligModel class
 * allows to generate and access to the data of an IBM 1 statistical
 * alignment model.
 */

#ifndef _Ibm1AligModel_h
#define _Ibm1AligModel_h

#if HAVE_CONFIG_H
#  include <thot_config.h>
#endif /* HAVE_CONFIG_H */
#include <unordered_map>

#include "_incrSwAligModel.h"
#include "WeightedIncrNormSlm.h"
#include "anjiMatrix.h"
#include "IncrLexTable.h"
#include "BestLgProbForTrgWord.h"
#include "LexCounts.h"

class Ibm1AligModel : public _swAligModel
{
  friend class IncrIbm1AligTrainer;

public:
  // Constructor
  Ibm1AligModel();

  // Functions to train model
  void trainSentPairRange(std::pair<unsigned int, unsigned int> sentPairRange, int verbosity = 0);
    // train model for range [uint,uint]
  std::pair<double, double> loglikelihoodForPairRange(std::pair<unsigned int, unsigned int> sentPairRange,
    int verbosity = 0);
    // Returns log-likelihood. The first double contains the
    // loglikelihood for all sentences, and the second one, the same
    // loglikelihood normalized by the number of sentences
  void clearInfoAboutSentRange();
    // clear info about the whole sentence range without clearing
    // information about current model parameters

  // Functions to access model parameters

  // lexical model functions
  Prob pts(WordIndex s, WordIndex t);
    // returns p(t|s)
  LgProb logpts(WordIndex s, WordIndex t);
    // returns log(p(t|s))

  // alignment model functions
  Prob aProbIbm1(PositionIndex slen, PositionIndex tlen);
  LgProb logaProbIbm1(PositionIndex slen, PositionIndex tlen);

  // Sentence length model functions
  Prob sentLenProb(PositionIndex slen, PositionIndex tlen);
    // returns p(tlen|slen)
  LgProb sentLenLgProb(PositionIndex slen, PositionIndex tlen);

  // Functions to get translations for word
  bool getEntriesForSource(WordIndex s, NbestTableNode<WordIndex>& trgtn);

  // Functions to generate alignments 
  LgProb obtainBestAlignment(const std::vector<WordIndex>& srcSentIndexVector,
    const std::vector<WordIndex>& trgSentIndexVector, WordAligMatrix& bestWaMatrix);
  LgProb lexM1LpForBestAlig(const std::vector<WordIndex>& nSrcSentIndexVector,
    const std::vector<WordIndex>& trgSentIndexVector, std::vector<PositionIndex>& bestAlig);

  // Functions to calculate probabilities for alignments
  LgProb calcLgProbForAlig(const std::vector<WordIndex>& sSent, const std::vector<WordIndex>& tSent,
    const WordAligMatrix& aligMatrix, int verbose = 0);
  LgProb calcIbm1LgProbForAlig(const std::vector<WordIndex>& nsSent, const std::vector<WordIndex>& tSent,
    const std::vector<PositionIndex>& alig, int verbose = 0);

  // Scoring functions without giving an alignment
  LgProb calcLgProb(const std::vector<WordIndex>& sSent, const std::vector<WordIndex>& tSent, int verbose = 0);
  LgProb calcSumIbm1LgProb(const char* sSent, const char* tSent, int verbose = 0);
  LgProb calcSumIbm1LgProb(const std::vector<std::string>& nsSent, const std::vector<std::string>& tSent,
    int verbose = 0);
  LgProb calcSumIbm1LgProb(const std::vector<WordIndex>& nsSent, const std::vector<WordIndex>& tSent, int verbose = 0);

  // load function
  bool load(const char* prefFileName, int verbose = 0);

  // print function
  bool print(const char* prefFileName, int verbose = 0);

  // clear() function
  void clear();

  // clearTempVars() function
  void clearTempVars();

  void clearSentLengthModel();

  // Destructor
  ~Ibm1AligModel();

protected:
  const std::size_t ThreadBufferSize = 10000;
  const float SmoothingAnjiNum = 1e-6f;
  const double ArbitraryProb = 0.1;

  // Functions to get sentence pairs
  std::vector<WordIndex> getSrcSent(unsigned int n);
    // get n-th source sentence
  std::vector<WordIndex> extendWithNullWord(const std::vector<WordIndex>& srcWordIndexVec);
    // given a vector with source words, returns a extended vector
    // including extra NULL words

  std::vector<WordIndex> getTrgSent(unsigned int n);
    // get n-th target sentence

  bool sentenceLengthIsOk(const std::vector<WordIndex> sentence);

  // Auxiliar scoring functions
  double unsmoothed_pts(WordIndex s, WordIndex t);
    // Returns p(t|s) without smoothing
  double unsmoothed_logpts(WordIndex s, WordIndex t);
    // Returns log(p(t|s)) without smoothing
  double ptsOrDefault(WordIndex s, WordIndex t);

  // Batch EM functions
  void initialBatchPass(std::pair<unsigned int, unsigned int> sentPairRange, int verbose);
  virtual void initSourceWord(const Sentence& nsrc, const Sentence& trg, PositionIndex i);
  virtual void initTargetWord(const Sentence& nsrc, const Sentence& trg, PositionIndex j);
  virtual void addTranslationOptions(std::vector<std::vector<WordIndex>>& insertBuffer);
  void batchUpdateCounts(const SentPairCont& pairs);
  virtual double calc_anji_num(const std::vector<WordIndex>& nsrc, const std::vector<WordIndex>& trg,
    PositionIndex i, PositionIndex j);
  virtual void incrementWordPairCounts(const Sentence& nsrc, const Sentence& trg, PositionIndex i, PositionIndex j,
    double count);
  virtual void batchMaximizeProbs();

  WeightedIncrNormSlm sentLengthModel;

  // EM counts
  LexCounts lexCounts;

  IncrLexTable lexTable;
  int iter = 0;
};

#endif
