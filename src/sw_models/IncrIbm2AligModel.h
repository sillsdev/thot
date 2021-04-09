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

#ifndef _IncrIbm2AligModel_h
#define _IncrIbm2AligModel_h

#if HAVE_CONFIG_H
#  include <thot_config.h>
#endif /* HAVE_CONFIG_H */
#include <unordered_map>

#include "IncrIbm1AligModel.h"
#include "aSource.h"
#include "IncrIbm2AligTable.h"

#define ARBITRARY_AP 0.1

class IncrIbm2AligModel : public IncrIbm1AligModel
{
public:
  // Constructor
  IncrIbm2AligModel();

  // Functions to train model
  //void efficientBatchTrainingForRange(std::pair<unsigned int, unsigned int> sentPairRange,
  //  int verbosity = 0);

  // Functions to access model parameters

  // alignment model functions
  virtual Prob aProb(PositionIndex j, PositionIndex slen, PositionIndex tlen, PositionIndex i);
    // Returns p(i|j,slen,tlen)
  LgProb logaProb(PositionIndex j, PositionIndex slen, PositionIndex tlen, PositionIndex i);
    // Returns log(p(i|j,slen,tlen))

  // Functions to generate alignments 
  LgProb obtainBestAlignment(std::vector<WordIndex> srcSentIndexVector, std::vector<WordIndex> trgSentIndexVector,
    WordAligMatrix& bestWaMatrix);

  LgProb lexAligM2LpForBestAlig(std::vector<WordIndex> nSrcSentIndexVector, std::vector<WordIndex> trgSentIndexVector,
    std::vector<PositionIndex>& bestAlig);

  // Functions to calculate probabilities for alignments
  LgProb calcLgProbForAlig(const std::vector<WordIndex>& sSent, const std::vector<WordIndex>& tSent,
    WordAligMatrix aligMatrix, int verbose = 0);
  LgProb incrIBM2LgProb(std::vector<WordIndex> nsSent, std::vector<WordIndex> tSent, std::vector<PositionIndex> alig,
    int verbose = 0);

  // Scoring functions without giving an alignment
  LgProb calcLgProb(const std::vector<WordIndex>& sSent, const std::vector<WordIndex>& tSent, int verbose = 0);
  LgProb calcSumIBM2LgProb(std::vector<WordIndex> nsSent, std::vector<WordIndex> tSent, int verbose = 0);

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

  IncrIbm2AligTable incrIbm2AligTable;

  typedef std::vector<std::pair<float, float>> IncrAligAuxVarElem;
  typedef OrderedVector<aSource, IncrAligAuxVarElem> IncrAligAuxVar;
  typedef std::vector<double> AligAuxVarElem;
  typedef OrderedVector<aSource, AligAuxVarElem> AligAuxVar;

  AligAuxVar aligAuxVar;
  IncrAligAuxVar incrAligAuxVar;
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
  void fillEmAuxVars(unsigned int mapped_n, unsigned int mapped_n_aux, PositionIndex i, PositionIndex j,
    const std::vector<WordIndex>& nsrcSent, const std::vector<WordIndex>& trgSent, const Count& weight);
  void fillEmAuxVarsAlig(unsigned int mapped_n, unsigned int mapped_n_aux, PositionIndex i, PositionIndex j,
    PositionIndex slen, PositionIndex tlen, const Count& weight);
  void updatePars();
  void updateParsAlig();

  void initTargetWord(const Sentence& nsrc, const Sentence& trg, PositionIndex j);
  void initWordPair(const Sentence& nsrc, const Sentence& trg, PositionIndex i, PositionIndex j);
  void incrementCount(const Sentence& nsrc, const Sentence& trg, PositionIndex i, PositionIndex j, double count);
  void normalizeCounts();

  // Mask for aSource. This function makes it possible to affect the
  // estimation of the alignment probabilities by setting to zero the
  // components of 'as'
  virtual void aSourceMask(aSource& as);
};

#endif
