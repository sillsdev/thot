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

extern "C"
{
#include "downhill_simplex/step_by_step_dhs.h"
}

#include "error_correction/EditDistForVec.h"
#include "phrase_models/BaseIncrPhraseModel.h"
#include "phrase_models/PhraseExtractParameters.h"
#include "phrase_models/PhrasePair.h"
#include "phrase_models/WbaIncrPhraseModel.h"
#include "phrase_models/_wbaIncrPhraseModel.h"
#include "stack_dec/PhrHypNumcovJumps01EqClassF.h"
#include "stack_dec/PhrLocalSwLiTmHypRec.h"
#include "stack_dec/_phrSwTransModel.h"
#include "sw_models/StepwiseAlignmentModel.h"

#define PHRSWLITM_LGPROB_SMOOTH -9999999
#define PHRSWLITM_DEFAULT_LR 0.5
#define PHRSWLITM_DEFAULT_LR_ALPHA_PAR 0.75
#define PHRSWLITM_DEFAULT_LR_PAR1 0.99
#define PHRSWLITM_DEFAULT_LR_PAR2 0.75
#define PHRSWLITM_LR_RESID_WER 0.2
#define PHRSWLITM_DHS_FTOL 0.001
#define PHRSWLITM_DHS_SCALE_PAR 1

typedef PhrHypNumcovJumps01EqClassF HypEqClassF;

/**
 * @brief The PhrLocalSwLiTm class implements a statistical translation
 * model specialized for phrase-based translation that combines a phrase
 * model with a single word model via linear interpolation. Training of
 * new samples is carried out using an interlaced training scheme.
 */
class PhrLocalSwLiTm : public _phrSwTransModel<PhrLocalSwLiTmHypRec<HypEqClassF>>
{
public:
  typedef _phrSwTransModel<PhrLocalSwLiTmHypRec<HypEqClassF>>::Hypothesis Hypothesis;
  typedef _phrSwTransModel<PhrLocalSwLiTmHypRec<HypEqClassF>>::HypScoreInfo HypScoreInfo;
  typedef _phrSwTransModel<PhrLocalSwLiTmHypRec<HypEqClassF>>::HypDataType HypDataType;

  // class functions

  // Constructor
  PhrLocalSwLiTm();

  // Virtual object copy
  BaseSmtModel<PhrLocalSwLiTmHypRec<HypEqClassF>>* clone();

  // Init alignment model
  bool loadAligModel(const char* prefixFileName, int verbose = 0);

  // Print models
  bool printAligModel(std::string printPrefix);

  void clear();

  // Functions to update linear interpolation weights
  int updateLinInterpWeights(std::string srcDevCorpusFileName, std::string trgDevCorpusFileName, int verbose = 0);

  ////// Hypotheses-related functions

  // Misc. operations with hypothesis
  Hypothesis nullHypothesis();
  HypDataType nullHypothesisHypData();
  bool obtainPredecessorHypData(HypDataType& hypd);
  bool isCompleteHypData(const HypDataType& hypd) const;

  // Model weights functions
  void setWeights(std::vector<float> wVec);
  void getWeights(std::vector<std::pair<std::string, float>>& compWeights);
  unsigned int getNumWeights();
  void printWeights(std::ostream& outS);

  // Functions for performing on-line training
  void setOnlineTrainingPars(OnlineTrainingPars _onlineTrainingPars, int verbose = 0);
  int onlineTrainFeatsSentPair(const char* srcSent, const char* refSent, const char* sysSent, int verbose = 0);

  // Destructor
  ~PhrLocalSwLiTm();

protected:
  // Training-related data members
  std::vector<std::vector<std::string>> vecSrcSent;
  std::vector<std::vector<std::string>> vecTrgSent;
  std::vector<std::vector<std::string>> vecSysSent;
  std::vector<std::vector<PhrasePair>> vecVecInvPhPair;
  unsigned int stepNum;

  // Weight auxiliary functions
  void setPmWeights(std::vector<float> wVec);
  void getPmWeights(std::vector<std::pair<std::string, float>>& compWeights);
  void printPmWeights(std::ostream& outS);

  // Functions related to linear interpolation weights updating
  int extractPhrPairsFromDevCorpus(std::string srcDevCorpusFileName, std::string trgDevCorpusFileName,
                                   std::vector<std::vector<PhrasePair>>& invPhrPairs, int verbose /*=0*/);
  double phraseModelPerplexity(const std::vector<std::vector<PhrasePair>>& invPhrPairs, int verbose = 0);
  int new_dhs_eval(const std::vector<std::vector<PhrasePair>>& invPhrPairs, FILE* tmp_file, double* x,
                   double& obj_func);

  // Function lo load and print lambda values
  bool load_lambdas(const char* lambdaFileName, int verbose);
  bool print_lambdas(const char* lambdaFileName);
  std::ostream& print_lambdas(std::ostream& outS);

  // Misc. operations with hypothesis
  unsigned int numberOfUncoveredSrcWordsHypData(const HypDataType& hypd) const;

  // Scoring functions
  Score incrScore(const Hypothesis& prev_hyp, const HypDataType& new_hypd, Hypothesis& new_hyp,
                  std::vector<Score>& scoreComponents);
  // Phrase model scoring functions
  Score smoothedPhrScore_s_t_(const std::vector<WordIndex>& s_, const std::vector<WordIndex>& t_);
  Score regularSmoothedPhrScore_s_t_(const std::vector<WordIndex>& s_, const std::vector<WordIndex>& t_);
  std::vector<Score> smoothedPhrScoreVec_s_t_(const std::vector<WordIndex>& s_, const std::vector<WordIndex>& t_);

  Score smoothedPhrScore_t_s_(const std::vector<WordIndex>& s_, const std::vector<WordIndex>& t_);
  Score regularSmoothedPhrScore_t_s_(const std::vector<WordIndex>& s_, const std::vector<WordIndex>& t_);
  std::vector<Score> smoothedPhrScoreVec_t_s_(const std::vector<WordIndex>& s_, const std::vector<WordIndex>& t_);

  // Vocabulary related functions
  void obtainSrcSwVocWordIdxVec(const std::vector<WordIndex>& s_, std::vector<WordIndex>& swVoc_s_);
  void obtainTrgSwVocWordIdxVec(const std::vector<WordIndex>& t_, std::vector<WordIndex>& swVoc_t_);

  // Functions to score n-best translations lists
  Score nbestTransScore(const std::vector<WordIndex>& s_, const std::vector<WordIndex>& t_);
  Score nbestTransScoreLast(const std::vector<WordIndex>& s_, const std::vector<WordIndex>& t_);

  PositionIndex getLastSrcPosCoveredHypData(const HypDataType& hypd);
  // Get the index of last source position which was covered

  // Functions for translating with references or prefixes
  bool hypDataTransIsPrefixOfTargetRef(const HypDataType& hypd, bool& equal) const;

  // Specific phrase-based functions
  void extendHypDataIdx(PositionIndex srcLeft, PositionIndex srcRight, const std::vector<WordIndex>& trgPhraseIdx,
                        HypDataType& hypd);

  // Functions for performing on-line training
  int extractConsistentPhrasePairs(const std::vector<std::string>& srcSentStrVec,
                                   const std::vector<std::string>& refSentStrVec, std::vector<PhrasePair>& vecInvPhPair,
                                   bool verbose = 0);
  int incrTrainFeatsSentPair(const char* srcSent, const char* refSent, int verbose = 0);
  int minibatchTrainFeatsSentPair(const char* srcSent, const char* refSent, const char* sysSent, int verbose = 0);
  int batchRetrainFeatsSentPair(const char* srcSent, const char* refSent, int verbose = 0);
  float calculateNewLearningRate(int verbose = 0);
  float werBasedLearningRate(int verbose = 0);
  unsigned int map_n_am_suff_stats(unsigned int n);
  int addNewTransOpts(unsigned int n, int verbose = 0);

  // Helper functions
  _wbaIncrPhraseModel* getWbaIncrPhraseModelPtr();
};
