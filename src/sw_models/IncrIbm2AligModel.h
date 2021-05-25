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
 * @file IncrIbm2AligModel.h
 *
 * @brief Defines the IncrIbm2AligModel class.  IncrIbm2AligModel class
 * allows to generate and access to the data of an IBM 2 statistical
 * alignment model.
 *
 */

#pragma once

#include "sw_models/IncrIbm1AligModel.h"
#include "sw_models/IncrIbm2AligTable.h"
#include "sw_models/aSource.h"

#include <unordered_map>

class IncrIbm2AligModel : public IncrIbm1AligModel
{
public:
  // Constructor
  IncrIbm2AligModel();

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
  ~IncrIbm2AligModel();

protected:
  const double ArbitraryAp = 0.1;
  IncrIbm2AligTable aligTable;

  typedef std::vector<std::pair<float, float>> IncrAligCountsEntry;
  typedef OrderedVector<aSource, IncrAligCountsEntry> IncrAligCounts;
  typedef std::vector<double> AligCountsEntry;
  typedef OrderedVector<aSource, AligCountsEntry> AligCounts;

  AligCounts aligCounts;
  IncrAligCounts incrAligCounts;
  // EM algorithm auxiliary variables

  // Auxiliar scoring functions
  virtual double unsmoothed_aProb(PositionIndex j, PositionIndex slen, PositionIndex tlen, PositionIndex i);
  // Returns p(i|j,slen,tlen) without smoothing
  double unsmoothed_logaProb(PositionIndex j, PositionIndex slen, PositionIndex tlen, PositionIndex i);
  // Returns log(p(i|j,slen,tlen)) without smoothing

  // EM-related functions
  double calc_anji_num(const std::vector<WordIndex>& nsrcSent, const std::vector<WordIndex>& trgSent, unsigned int i,
                       unsigned int j);
  double calc_anji_num_alig(PositionIndex i, PositionIndex j, PositionIndex slen, PositionIndex tlen);
  void incrUpdateCounts(unsigned int mapped_n, unsigned int mapped_n_aux, PositionIndex i, PositionIndex j,
                     const std::vector<WordIndex>& nsrcSent, const std::vector<WordIndex>& trgSent,
  void incrUpdateCountsAlig(unsigned int mapped_n, unsigned int mapped_n_aux, PositionIndex i, PositionIndex j,
                         PositionIndex slen, PositionIndex tlen, const Count& weight);
  void incrMaximizeProbs();
  void incrMaximizeProbsAlig();

  void initTargetWord(const Sentence& nsrc, const Sentence& trg, PositionIndex j);
  void initWordPair(const Sentence& nsrc, const Sentence& trg, PositionIndex i, PositionIndex j);
  void incrementWordPairCounts(const Sentence& nsrc, const Sentence& trg, PositionIndex i, PositionIndex j, double count);
  void batchMaximizeProbs();

  // Mask for aSource. This function makes it possible to affect the
  // estimation of the alignment probabilities by setting to zero the
  // components of 'as'
  virtual void aSourceMask(aSource& as);
};

