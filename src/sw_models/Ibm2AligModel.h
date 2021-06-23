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

#include "sw_models/AlignmentTable.h"
#include "sw_models/Ibm1AligModel.h"

#include <unordered_map>

class Ibm2AligModel : public Ibm1AligModel
{
  friend class IncrIbm2AligTrainer;

public:
  // Constructor
  Ibm2AligModel();

  // Returns p(i|j,slen,tlen)
  virtual Prob aProb(PositionIndex j, PositionIndex slen, PositionIndex tlen, PositionIndex i);
  // Returns log(p(i|j,slen,tlen))
  LgProb logaProb(PositionIndex j, PositionIndex slen, PositionIndex tlen, PositionIndex i);

  LgProb obtainBestAlignment(const std::vector<WordIndex>& srcSentIndexVector,
                             const std::vector<WordIndex>& trgSentIndexVector, WordAligMatrix& bestWaMatrix);
  LgProb calcLgProbForAlig(const std::vector<WordIndex>& sSent, const std::vector<WordIndex>& tSent,
                           const WordAligMatrix& aligMatrix, int verbose = 0);
  LgProb calcLgProb(const std::vector<WordIndex>& sSent, const std::vector<WordIndex>& tSent, int verbose = 0);

  bool load(const char* prefFileName, int verbose = 0);
  bool print(const char* prefFileName, int verbose = 0);

  void clear();
  void clearTempVars();
  void clearInfoAboutSentRange();

  ~Ibm2AligModel();

protected:
  typedef std::vector<double> AlignmentCountsElem;
  typedef OrderedVector<AlignmentKey, AlignmentCountsElem> AlignmentCounts;

  virtual double unsmoothed_aProb(PositionIndex j, PositionIndex slen, PositionIndex tlen, PositionIndex i);
  double unsmoothed_logaProb(PositionIndex j, PositionIndex slen, PositionIndex tlen, PositionIndex i);
  double aProb(PositionIndex j, PositionIndex slen, PositionIndex tlen, PositionIndex i, bool training);

  LgProb lexAligM2LpForBestAlig(const std::vector<WordIndex>& nSrcSentIndexVector,
                                const std::vector<WordIndex>& trgSentIndexVector, std::vector<PositionIndex>& bestAlig);
  LgProb calcIbm2LgProbForAlig(const std::vector<WordIndex>& nsSent, const std::vector<WordIndex>& tSent,
                               const std::vector<PositionIndex>& alig, int verbose = 0);
  LgProb calcSumIbm2LgProb(const std::vector<WordIndex>& nsSent, const std::vector<WordIndex>& tSent, int verbose = 0);

  void initTargetWord(const Sentence& nsrc, const Sentence& trg, PositionIndex j);
  double calc_anji_num(const std::vector<WordIndex>& nsrcSent, const std::vector<WordIndex>& trgSent, unsigned int i,
                       unsigned int j);
  void incrementWordPairCounts(const Sentence& nsrc, const Sentence& trg, PositionIndex i, PositionIndex j,
                               double count);
  void batchMaximizeProbs();

  AlignmentTable alignmentTable;

  AlignmentCounts alignmentCounts;
};

#endif
