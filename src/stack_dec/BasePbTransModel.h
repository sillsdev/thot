/*
thot package for statistical machine translation
Copyright (C) 2013 Daniel Ortiz-Mart\'inez and SIL International

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

#pragma once

#include "nlp_common/SmtDefs.h"
#include "nlp_common/StrProcUtils.h"
#include "stack_dec/BasePbTransModelStats.h"
#include "stack_dec/PbTransModelPars.h"
#include "stack_dec/_phraseHypothesis.h"
#include "stack_dec/_phraseHypothesisRec.h"
#include "stack_dec/_smtModel.h"

#define PBM_W_DEFAULT 10
#define PBM_A_DEFAULT 10
#define PBM_E_DEFAULT 10
#define PBM_U_DEFAULT 10

/**
 * @brief The BasePbTransModel class is a predecessor of the
 * _phraseBasedTransModel class. In this class it is assumed that the
 * template parameter HYPOTHESIS is a class derived from the
 * BasePhraseHypothesis or the BasePhraseHypothesisRec classes.
 */
template <class HYPOTHESIS>
class BasePbTransModel : public _smtModel<HYPOTHESIS>
{
public:
  typedef typename _smtModel<HYPOTHESIS>::Hypothesis Hypothesis;
  typedef typename _smtModel<HYPOTHESIS>::HypScoreInfo HypScoreInfo;
  typedef typename _smtModel<HYPOTHESIS>::HypDataType HypDataType;

  // Constructor
  BasePbTransModel(void);

  virtual void clear(void) = 0;

  // Heuristic-related functions
  virtual void setHeuristic(unsigned int _heuristicId) = 0;

  ////// Hypotheses-related functions

  // Specific phrase-based functions
  virtual void extendHypData(PositionIndex srcLeft, PositionIndex srcRight, const std::vector<std::string>& trgPhrase,
                             HypDataType& hypd) = 0;

  // Misc. operations with hypothesis
  unsigned int distToNullHyp(const Hypothesis& hyp);
  virtual void aligMatrix(const Hypothesis& hyp, std::vector<std::pair<PositionIndex, PositionIndex>>& amatrix);
  // Returns an alignment matrix for 'hyp' hypothesis
  virtual std::pair<PositionIndex, PositionIndex> getLastSourceSegment(const Hypothesis& hyp);

  // Printing functions and data conversion
  unsigned int partialTransLength(const Hypothesis& hyp) const;

  // Expansion-related parameters
  void set_W_par(float W_par);
  float get_W_par() const;
  void set_A_par(unsigned int A_par);
  unsigned int get_A_par() const;
  void set_E_par(unsigned int E_par);
  unsigned int get_E_par() const;
  void set_U_par(unsigned int U_par);
  unsigned int get_U_par() const;
  bool monotoneSearch() const;
  // Returns true if the search is monotone

  // Set verbosity level
  void setVerbosity(int _verbosity);

  // Utility functions
  void getPhraseAlignment(const std::vector<std::pair<PositionIndex, PositionIndex>>& amatrix,
                          SourceSegmentation& sourceSegmentation, std::vector<PositionIndex>& targetSegmentCuts);
  std::vector<std::vector<std::string>> getSrcPhrases(const std::vector<std::string>& srcSentVec,
                                                      const Hypothesis& hyp);
  std::vector<std::vector<std::string>> getTrgPhrases(const Hypothesis& hyp);

  // Destructor
  ~BasePbTransModel();

#ifdef THOT_STATS
  virtual std::ostream& printStats(std::ostream& outS);
  virtual void clearStats(void);
  BasePbTransModelStats basePbTmStats;
#endif

protected:
  PbTransModelPars pbTransModelPars; // Model parameters

  int verbosity; // Verbosity level

  ////// Hypotheses-related functions

  // Misc. operations with hypothesis
  virtual unsigned int numberOfUncoveredSrcWords(const Hypothesis& hyp) const;
  virtual unsigned int numberOfUncoveredSrcWordsHypData(const HypDataType& hypd) const = 0;
};

template <class HYPOTHESIS>
BasePbTransModel<HYPOTHESIS>::BasePbTransModel(void) : _smtModel<HYPOTHESIS>()
{
  // Set verbosity level
  verbosity = 0;
}

template <class HYPOTHESIS>
void BasePbTransModel<HYPOTHESIS>::getPhraseAlignment(
    const std::vector<std::pair<PositionIndex, PositionIndex>>& amatrix, SourceSegmentation& sourceSegmentation,
    std::vector<PositionIndex>& targetSegmentCuts)
{
  sourceSegmentation.clear();
  targetSegmentCuts.clear();

  if (amatrix.size() > 0)
  {
    std::vector<std::pair<PositionIndex, PositionIndex>> temp;
    std::pair<PositionIndex, PositionIndex> pip;

    // Create temporary data structure 'temp' from 'amatrix'
    for (unsigned int i = 0; i < amatrix.size(); ++i)
    {
      unsigned int j = amatrix[i].second;
      while (temp.size() <= j)
        temp.push_back(std::make_pair(MAX_SENTENCE_LENGTH_ALLOWED + 1, 0));
      if (temp[j].first > amatrix[i].first)
        temp[j].first = amatrix[i].first;
      if (temp[j].second < amatrix[i].first)
        temp[j].second = amatrix[i].first;
    }
    // Set contents of 'sourceSegmentation' and 'targetSegmentCuts'
    // data structures from 'temp'
    pip = temp[1];
    for (unsigned int j = 1; j < temp.size(); ++j)
    {
      if (j == temp.size() - 1)
      {
        sourceSegmentation.push_back(temp[j]);
        targetSegmentCuts.push_back(j);
      }
      else
      {
        if (pip != temp[j + 1])
        {
          sourceSegmentation.push_back(temp[j]);
          targetSegmentCuts.push_back(j);
          pip = temp[j + 1];
        }
      }
    }
  }
}

template <class HYPOTHESIS>
std::vector<std::vector<std::string>> BasePbTransModel<HYPOTHESIS>::getSrcPhrases(
    const std::vector<std::string>& srcSentVec, const Hypothesis& hyp)
{
  std::vector<std::pair<PositionIndex, PositionIndex>> amatrix;
  SourceSegmentation sourceSegmentation;
  std::vector<PositionIndex> targetSegmentCuts;
  std::vector<std::vector<std::string>> srcPhrases;
  std::vector<std::string> srcPhrase;

  aligMatrix(hyp, amatrix);

  getPhraseAlignment(amatrix, sourceSegmentation, targetSegmentCuts);

  for (unsigned int i = 0; i < sourceSegmentation.size(); ++i)
  {
    srcPhrase.clear();
    for (unsigned int j = sourceSegmentation[i].first; j <= sourceSegmentation[i].second; ++j)
    {
      srcPhrase.push_back(srcSentVec[j - 1]);
    }
    srcPhrases.push_back(srcPhrase);
  }

  return srcPhrases;
}

template <class HYPOTHESIS>
std::vector<std::vector<std::string>> BasePbTransModel<HYPOTHESIS>::getTrgPhrases(const Hypothesis& hyp)
{
  std::vector<std::pair<PositionIndex, PositionIndex>> amatrix;
  SourceSegmentation sourceSegmentation;
  std::vector<PositionIndex> targetSegmentCuts;
  std::vector<std::vector<std::string>> trgPhrases;
  std::vector<std::string> trgPhrase;
  std::vector<std::string> trgSentVec = getTransInPlainTextVec(hyp);

  aligMatrix(hyp, amatrix);

  getPhraseAlignment(amatrix, sourceSegmentation, targetSegmentCuts);

  for (unsigned int i = 0; i < targetSegmentCuts.size(); ++i)
  {
    trgPhrase.clear();
    unsigned int phrStart = 1;
    if (i > 0)
      phrStart = targetSegmentCuts[i - 1] + 1;
    for (unsigned int j = phrStart; j <= targetSegmentCuts[i]; ++j)
    {
      trgPhrase.push_back(trgSentVec[j - 1]);
    }
    trgPhrases.push_back(trgPhrase);
  }

  return trgPhrases;
}

template <class HYPOTHESIS>
void BasePbTransModel<HYPOTHESIS>::set_W_par(float W_par)
{
  pbTransModelPars.W = W_par;
}

template <class HYPOTHESIS>
float BasePbTransModel<HYPOTHESIS>::get_W_par() const
{
  return pbTransModelPars.W;
}

template <class HYPOTHESIS>
void BasePbTransModel<HYPOTHESIS>::set_A_par(unsigned int A_par)
{
  pbTransModelPars.A = A_par;
}

template <class HYPOTHESIS>
unsigned int BasePbTransModel<HYPOTHESIS>::get_A_par() const
{
  return pbTransModelPars.A;
}

template <class HYPOTHESIS>
void BasePbTransModel<HYPOTHESIS>::set_E_par(unsigned int E_par)
{
  pbTransModelPars.E = E_par;
}

template <class HYPOTHESIS>
unsigned int BasePbTransModel<HYPOTHESIS>::get_E_par() const
{
  return pbTransModelPars.E;
}

template <class HYPOTHESIS>
void BasePbTransModel<HYPOTHESIS>::set_U_par(unsigned int U_par)
{
  pbTransModelPars.U = U_par;
}

template <class HYPOTHESIS>
unsigned int BasePbTransModel<HYPOTHESIS>::get_U_par() const
{
  return pbTransModelPars.U;
}

template <class HYPOTHESIS>
bool BasePbTransModel<HYPOTHESIS>::monotoneSearch() const
{
  if (pbTransModelPars.U == 0)
    return true;
  else
    return false;
}

template <class HYPOTHESIS>
void BasePbTransModel<HYPOTHESIS>::setVerbosity(int _verbosity)
{
  verbosity = _verbosity;
}

template <class HYPOTHESIS>
BasePbTransModel<HYPOTHESIS>::~BasePbTransModel()
{
}

template <class HYPOTHESIS>
void BasePbTransModel<HYPOTHESIS>::aligMatrix(const Hypothesis& hyp,
                                              std::vector<std::pair<PositionIndex, PositionIndex>>& amatrix)
{
  Hypothesis nullHyp = this->nullHypothesis();
  unsigned int numSrcWords = numberOfUncoveredSrcWords(nullHyp);
  unsigned int numTrgWords = hyp.partialTransLength();

  amatrix.clear();
  for (unsigned int i = 0; i <= numSrcWords; ++i)
  {
    for (unsigned int j = 0; j <= numTrgWords; ++j)
    {
      if (hyp.areAligned(i, j))
        amatrix.push_back(std::make_pair(i, j));
    }
  }
}

template <class HYPOTHESIS>
std::pair<PositionIndex, PositionIndex> BasePbTransModel<HYPOTHESIS>::getLastSourceSegment(const Hypothesis& hyp)
{
  if (hyp.getData().sourceSegmentation.size() == 0)
    return std::pair<PositionIndex, PositionIndex>(0, 0);
  return hyp.getData().sourceSegmentation.back();
}

template <class HYPOTHESIS>
unsigned int BasePbTransModel<HYPOTHESIS>::distToNullHyp(const Hypothesis& hyp)
{
  return numberOfUncoveredSrcWordsHypData(this->nullHypothesisHypData())
       - numberOfUncoveredSrcWordsHypData(hyp.getData());
}

template <class HYPOTHESIS>
unsigned int BasePbTransModel<HYPOTHESIS>::partialTransLength(const Hypothesis& hyp) const
{
  return hyp.partialTransLength();
}

template <class HYPOTHESIS>
unsigned int BasePbTransModel<HYPOTHESIS>::numberOfUncoveredSrcWords(const Hypothesis& hyp) const
{
  HypDataType dataType;

  dataType = hyp.getData();
  return numberOfUncoveredSrcWordsHypData(dataType);
}

#ifdef THOT_STATS
template <class HYPOTHESIS>
std::ostream& BasePbTransModel<HYPOTHESIS>::printStats(std::ostream& outS)
{
  return basePbTmStats.print(outS);
}

template <class HYPOTHESIS>
void BasePbTransModel<HYPOTHESIS>::clearStats(void)
{
  basePbTmStats.clear();
}

#endif
