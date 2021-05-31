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
 * @file Ibm2AligModel.h
 *
 * @brief Defines the Ibm2AligModel class. Ibm2AligModel class
 * allows to generate and access to the data of an IBM 2 statistical
 * alignment model.
 *
 */

#ifndef _Ibm2AligModel_h
#define _Ibm2AligModel_h

#if HAVE_CONFIG_H
#  include <thot_config.h>
#endif /* HAVE_CONFIG_H */
#include <unordered_map>

#include "Ibm1AligModel.h"
#include "aSource.h"
#include "IncrIbm2AligTable.h"

class Ibm2AligModel : public Ibm1AligModel
{
  friend class IncrIbm2AligTrainer;

public:
  // Constructor
  Ibm2AligModel();

  // Functions to access model parameters

  // alignment model functions
  virtual Prob aProb(PositionIndex j, PositionIndex slen, PositionIndex tlen, PositionIndex i);
    // Returns p(i|j,slen,tlen)
  LgProb logaProb(PositionIndex j, PositionIndex slen, PositionIndex tlen, PositionIndex i);
    // Returns log(p(i|j,slen,tlen))

  // Functions to generate alignments 
  LgProb obtainBestAlignment(const std::vector<WordIndex>& srcSentIndexVector,
    const std::vector<WordIndex>& trgSentIndexVector, WordAligMatrix& bestWaMatrix);

  LgProb lexAligM2LpForBestAlig(const std::vector<WordIndex>& nSrcSentIndexVector,
    const std::vector<WordIndex>& trgSentIndexVector, std::vector<PositionIndex>& bestAlig,
    std::vector<PositionIndex>& fertility);

  // Functions to calculate probabilities for alignments
  LgProb calcLgProbForAlig(const std::vector<WordIndex>& sSent, const std::vector<WordIndex>& tSent,
    const WordAligMatrix& aligMatrix, int verbose = 0);
  LgProb calcIbm2LgProbForAlig(const std::vector<WordIndex>& nsSent, const std::vector<WordIndex>& tSent,
    const std::vector<PositionIndex>& alig, int verbose = 0);

  // Scoring functions without giving an alignment
  LgProb calcLgProb(const std::vector<WordIndex>& sSent, const std::vector<WordIndex>& tSent, int verbose = 0);
  LgProb calcSumIbm2LgProb(const std::vector<WordIndex>& nsSent, const std::vector<WordIndex>& tSent, int verbose = 0);

  // load function
  bool load(const char* prefFileName, int verbose = 0);

  // print function
  bool print(const char* prefFileName, int verbose = 0);

  // clear() function
  void clear();

  void clearTempVars();

  void clearInfoAboutSentRange();

  // Destructor
  ~Ibm2AligModel();

protected:
  typedef std::vector<double> AligCountsEntry;
  typedef OrderedVector<aSource, AligCountsEntry> AligCounts;

  // Auxiliar scoring functions
  virtual double unsmoothed_aProb(PositionIndex j, PositionIndex slen, PositionIndex tlen, PositionIndex i);
    // Returns p(i|j,slen,tlen) without smoothing
  double unsmoothed_logaProb(PositionIndex j, PositionIndex slen, PositionIndex tlen, PositionIndex i);
    // Returns log(p(i|j,slen,tlen)) without smoothing
  double aProbOrDefault(PositionIndex j, PositionIndex slen, PositionIndex tlen, PositionIndex i);

  void initTargetWord(const Sentence& nsrc, const Sentence& trg, PositionIndex j);
  double calc_anji_num(const std::vector<WordIndex>& nsrcSent, const std::vector<WordIndex>& trgSent, unsigned int i,
    unsigned int j);
  double calc_anji_num_alig(PositionIndex i, PositionIndex j, PositionIndex slen, PositionIndex tlen);
  void incrementWordPairCounts(const Sentence& nsrc, const Sentence& trg, PositionIndex i, PositionIndex j,
    double count);
  void batchMaximizeProbs();

  // Mask for aSource. This function makes it possible to affect the
  // estimation of the alignment probabilities by setting to zero the
  // components of 'as'
  virtual void aSourceMask(aSource& as);

  IncrIbm2AligTable aligTable;

  AligCounts aligCounts;
};

#endif
