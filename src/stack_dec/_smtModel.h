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

#include "stack_dec/BaseLogLinWeightUpdater.h"
#include "stack_dec/BaseSmtModel.h"
#include "stack_dec/BaseTranslationMetadata.h"

#include <memory>

#define NBEST_LIST_SIZE_FOR_LLWEIGHT_UPDATE 1000
#define SMALL_LLWEIGHT 1e-6

/**
 * @brief Base predecessor class that starts the development of the
 * interface defined in the BaseSmtModel abstract class.
 */

template <class HYPOTHESIS>
class _smtModel : public BaseSmtModel<HYPOTHESIS>
{
public:
  typedef typename BaseSmtModel<HYPOTHESIS>::Hypothesis Hypothesis;
  typedef typename BaseSmtModel<HYPOTHESIS>::HypScoreInfo HypScoreInfo;
  typedef typename BaseSmtModel<HYPOTHESIS>::HypDataType HypDataType;

  // Constructor
  _smtModel();

  // Link translation constraints with model
  virtual void setTranslationMetadata(BaseTranslationMetadata<HypScoreInfo>* trMetadata);
  BaseTranslationMetadata<HypScoreInfo>* getTranslationMetadata()
  {
    return transMetadata.get();
  }

  // Actions to be executed before the translation and before using
  // hypotheses-related functions
  virtual void pre_trans_actions(std::string srcsent) = 0;
  virtual void pre_trans_actions_ref(std::string srcsent, std::string refsent) = 0;
  virtual void pre_trans_actions_ver(std::string srcsent, std::string refsent) = 0;
  virtual void pre_trans_actions_prefix(std::string srcsent, std::string prefix) = 0;

  // Function to obtain current source sentence (it may differ from
  // that provided when calling pre_trans_actions since information
  // about translation constraints is removed)
  virtual std::string getCurrentSrcSent() = 0;

  ////// Hypotheses-related functions

  // Expansion-related functions
  virtual void expand(const Hypothesis& hyp, std::vector<Hypothesis>& hypVec,
                      std::vector<std::vector<Score>>& scrCompVec) = 0;
  virtual void expand_ref(const Hypothesis& hyp, std::vector<Hypothesis>& hypVec,
                          std::vector<std::vector<Score>>& scrCompVec) = 0;
  virtual void expand_ver(const Hypothesis& hyp, std::vector<Hypothesis>& hypVec,
                          std::vector<std::vector<Score>>& scrCompVec) = 0;
  virtual void expand_prefix(const Hypothesis& hyp, std::vector<Hypothesis>& hypVec,
                             std::vector<std::vector<Score>>& scrCompVec) = 0;

  // Functions for performing on-line training
  void setOnlineTrainingPars(OnlineTrainingPars _onlineTrainingPars, int verbose);
  const OnlineTrainingPars& getOnlineTrainingPars() const;

  // Misc. operations with hypothesis
  virtual void obtainHypFromHypData(const HypDataType& hypDataType, Hypothesis& hyp);
  virtual bool obtainPredecessor(Hypothesis& hyp);
  virtual unsigned int distToNullHyp(const Hypothesis& hyp) = 0;
  virtual void printHyp(const Hypothesis& hyp, std::ostream& outS, int verbose = false) = 0;
  virtual void diffScoreCompsForHyps(const Hypothesis& pred_hyp, const Hypothesis& succ_hyp,
                                     std::vector<Score>& scoreComponents);

  // IMPORTANT NOTE: Before using the hypothesis-related functions
  // it is mandatory the previous invocation of one of the
  // pre_trans_actionsXXXX functions

  // Destructor
  virtual ~_smtModel(){};

#ifdef THOT_STATS
  virtual std::ostream& printStats(std::ostream& outS) = 0;
  virtual void clearStats(void) = 0;
#endif

protected:
  OnlineTrainingPars onlineTrainingPars;

  std::shared_ptr<BaseTranslationMetadata<HypScoreInfo>> transMetadata;

  // Scoring functions
  virtual Score incrScore(const Hypothesis& prev_hyp, const HypDataType& new_hypd, Hypothesis& new_hyp,
                          std::vector<Score>& scoreComponents) = 0;

  // Helper functions
  float smoothLlWeight(float weight);
};

template <class HYPOTHESIS>
_smtModel<HYPOTHESIS>::_smtModel()
{
}

template <class HYPOTHESIS>
void _smtModel<HYPOTHESIS>::setTranslationMetadata(BaseTranslationMetadata<HypScoreInfo>* trMetadata)
{
  transMetadata.reset(trMetadata);
}

template <class HYPOTHESIS>
void _smtModel<HYPOTHESIS>::setOnlineTrainingPars(OnlineTrainingPars _onlineTrainingPars, int /*verbose*/)
{
  onlineTrainingPars = _onlineTrainingPars;
}

template <class HYPOTHESIS>
const OnlineTrainingPars& _smtModel<HYPOTHESIS>::getOnlineTrainingPars() const
{
  return onlineTrainingPars;
}

template <class HYPOTHESIS>
bool _smtModel<HYPOTHESIS>::obtainPredecessor(Hypothesis& hyp)
{
  typename Hypothesis::DataType predData;
  std::vector<Score> scoreComponents;

  predData = hyp.getData();
  if (!this->obtainPredecessorHypData(predData))
    return false;
  {
    Hypothesis null_hyp = this->nullHypothesis();

    incrScore(null_hyp, predData, hyp, scoreComponents);

    return true;
  }
}

template <class HYPOTHESIS>
void _smtModel<HYPOTHESIS>::obtainHypFromHypData(const HypDataType& hypDataType, Hypothesis& hyp)
{
  std::vector<Score> scoreComponents;

  incrScore(this->nullHypothesis(), hypDataType, hyp, scoreComponents);
}

template <class HYPOTHESIS>
void _smtModel<HYPOTHESIS>::diffScoreCompsForHyps(const Hypothesis& pred_hyp, const Hypothesis& succ_hyp,
                                                  std::vector<Score>& scoreComponents)
{
  typename Hypothesis::DataType succ_hypd = succ_hyp.getData();
  Hypothesis aux;
  incrScore(pred_hyp, succ_hypd, aux, scoreComponents);
}

template <class HYPOTHESIS>
float _smtModel<HYPOTHESIS>::smoothLlWeight(float weight)
{
  if (weight <= SMALL_LLWEIGHT && weight >= 0)
    return SMALL_LLWEIGHT;
  else
  {
    if (weight >= -SMALL_LLWEIGHT && weight < 0)
      return -SMALL_LLWEIGHT;
    else
      return weight;
  }
}
