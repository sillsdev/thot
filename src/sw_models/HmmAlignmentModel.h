#pragma once

#include "nlp_common/Matrix.h"
#include "sw_models/AlignmentInfo.h"
#include "sw_models/CachedHmmAligLgProb.h"
#include "sw_models/HmmAlignmentTable.h"
#include "sw_models/Ibm1AlignmentModel.h"

#include <memory>
#include <unordered_map>

#define DEFAULT_ALIG_SMOOTH_INTERP_FACTOR 0.3
#define DEFAULT_LEX_SMOOTH_INTERP_FACTOR 0.1
#define DEFAULT_HMM_P0 0.1

class HmmAlignmentKey
{
public:
  PositionIndex prev_i;
  PositionIndex slen;

  bool operator==(const HmmAlignmentKey& right) const
  {
    if (right.prev_i != prev_i)
      return 0;
    if (right.slen != slen)
      return 0;
    return 1;
  }

  bool operator<(const HmmAlignmentKey& right) const
  {
    if (right.prev_i < prev_i)
      return 0;
    if (prev_i < right.prev_i)
      return 1;
    if (right.slen < slen)
      return 0;
    if (slen < right.slen)
      return 1;
    return 0;
  }
};

class HmmAligInfo
{
public:
  bool validAlig;
  bool nullAlig;
  PositionIndex modified_ip;
};

class HmmAlignmentModel : public Ibm1AlignmentModel
{
  friend class IncrHmmAlignmentTrainer;
  friend class Ibm3AlignmentModel;

public:
  HmmAlignmentModel();
  HmmAlignmentModel(Ibm1AlignmentModel& model);

  // Sets lexical smoothing interpolation factor
  void setLexSmIntFactor(double _lexSmoothInterpFactor, int verbose = 0);
  // Sets alignment smoothing interpolation factor
  void setAlSmIntFactor(double _aligSmoothInterpFactor, int verbose = 0);

  // Set hmm p0 value
  void set_hmm_p0(Prob _hmm_p0);

  void startTraining(int verbosity = 0) override;

  // returns p(t|s)
  Prob pts(WordIndex s, WordIndex t) override;
  // returns log(p(t|s))
  LgProb logpts(WordIndex s, WordIndex t) override;
  // Returns p(i|prev_i,slen)
  Prob aProb(PositionIndex prev_i, PositionIndex slen, PositionIndex i);
  // Returns log(p(i|prev_i,slen))
  LgProb logaProb(PositionIndex prev_i, PositionIndex slen, PositionIndex i);

  using AlignmentModel::getBestAlignment;
  LgProb getBestAlignment(const std::vector<WordIndex>& srcSentence, const std::vector<WordIndex>& trgSentence,
                          std::vector<PositionIndex>& bestAlignment) override;
  using AlignmentModel::getAlignmentLgProb;
  LgProb getAlignmentLgProb(const std::vector<WordIndex>& srcSentence, const std::vector<WordIndex>& trgSentence,
                            const WordAlignmentMatrix& aligMatrix, int verbose = 0) override;
  using AlignmentModel::getSumLgProb;
  LgProb getSumLgProb(const std::vector<WordIndex>& srcSentence, const std::vector<WordIndex>& trgSentence,
                      int verbose = 0) override;

  Prob searchForBestAlignment(PositionIndex maxFertility, const std::vector<WordIndex>& src,
                              const std::vector<WordIndex>& trg, AlignmentInfo& bestAlignment,
                              Matrix<double>* moveScores = nullptr, Matrix<double>* swapScores = nullptr);

  bool load(const char* prefFileName, int verbose = 0) override;
  bool print(const char* prefFileName, int verbose = 0) override;

  void clear() override;
  void clearTempVars() override;

  virtual ~HmmAlignmentModel()
  {
  }

protected:
  typedef std::vector<double> HmmAlignmentCountsElem;
  typedef OrderedVector<HmmAlignmentKey, HmmAlignmentCountsElem> HmmAlignmentCounts;

  const double ExpValLogMax = -0.01;
  const double ExpValLogMin = -9;
  const double Log1 = log(1.0);

  HmmAlignmentModel(HmmAlignmentModel& model);

  double unsmoothed_logaProb(PositionIndex prev_i, PositionIndex slen, PositionIndex i);
  double cached_logaProb(PositionIndex prev_i, PositionIndex slen, PositionIndex i);
  std::vector<WordIndex> extendWithNullWord(const std::vector<WordIndex>& srcWordIndexVec) override;
  LgProb getBestAlignmentCached(const std::vector<WordIndex>& srcSentence, const std::vector<WordIndex>& trgSentence,
                                CachedHmmAligLgProb& cached_logap, std::vector<PositionIndex>& bestAlignment);
  // Execute the Viterbi algorithm to obtain the best HMM word alignment
  void viterbiAlgorithm(const std::vector<WordIndex>& nSrcSentIndexVector,
                        const std::vector<WordIndex>& trgSentIndexVector, std::vector<std::vector<double>>& vitMatrix,
                        std::vector<std::vector<PositionIndex>>& predMatrix);
  // Cached version of viterbiAlgorithm()
  void viterbiAlgorithmCached(const std::vector<WordIndex>& nSrcSentIndexVector,
                              const std::vector<WordIndex>& trgSentIndexVector, CachedHmmAligLgProb& cached_logap,
                              std::vector<std::vector<double>>& vitMatrix,
                              std::vector<std::vector<PositionIndex>>& predMatrix);
  // Obtain best alignment vector from Viterbi algorithm matrices, index of null word depends on how the source index
  // vector is transformed
  double bestAligGivenVitMatricesRaw(const std::vector<std::vector<double>>& vitMatrix,
                                     const std::vector<std::vector<PositionIndex>>& predMatrix,
                                     std::vector<PositionIndex>& bestAlig);
  // Obtain best alignment vector from Viterbi algorithm matrices, index of null word is zero
  double bestAligGivenVitMatrices(PositionIndex slen, const std::vector<std::vector<double>>& vitMatrix,
                                  const std::vector<std::vector<PositionIndex>>& predMatrix,
                                  std::vector<PositionIndex>& bestAlig);
  // Execute Forward algorithm to obtain the log-probability of a sentence pair
  double forwardAlgorithm(const std::vector<WordIndex>& nSrcSentIndexVector,
                          const std::vector<WordIndex>& trgSentIndexVector, int verbose = 0);
  double lgProbGivenForwardMatrix(const std::vector<std::vector<double>>& forwardMatrix);
  void calcAlphaBetaMatrices(const std::vector<WordIndex>& nsrcSent, const std::vector<WordIndex>& trgSent,
                             PositionIndex slen, std::vector<std::vector<double>>& cachedLexLogProbs,
                             std::vector<std::vector<double>>& alphaMatrix,
                             std::vector<std::vector<double>>& betaMatrix);
  PositionIndex getSrcLen(const std::vector<WordIndex>& nsrcWordIndexVec);
  Prob calcProbOfAlignment(CachedHmmAligLgProb& cached_logap, const std::vector<WordIndex>& nsrc,
                           const std::vector<WordIndex>& trg, AlignmentInfo& alignment, int verbose = 0);
  double swapScore(CachedHmmAligLgProb& cached_logap, const std::vector<WordIndex>& nsrc,
                   const std::vector<WordIndex>& trg, PositionIndex j1, PositionIndex j2, AlignmentInfo& alignment);
  double moveScore(CachedHmmAligLgProb& cached_logap, const std::vector<WordIndex>& nsrc,
                   const std::vector<WordIndex>& trg, PositionIndex iNew, PositionIndex j, AlignmentInfo& alignment);

  bool isFirstNullAlignmentPar(PositionIndex ip, unsigned int slen, PositionIndex i);
  double calc_lanji_num(PositionIndex i, PositionIndex j, const std::vector<std::vector<double>>& alphaMatrix,
                        const std::vector<std::vector<double>>& betaMatrix);
  double calc_lanjm1ip_anji_num_je1(PositionIndex slen, PositionIndex i,
                                    const std::vector<std::vector<double>>& cachedLexLogProbs,
                                    const std::vector<std::vector<double>>& betaMatrix);
  double calc_lanjm1ip_anji_num_jg1(PositionIndex ip, PositionIndex slen, PositionIndex i, PositionIndex j,
                                    const std::vector<std::vector<double>>& cachedLexLogProbs,
                                    const std::vector<std::vector<double>>& alphaMatrix,
                                    const std::vector<std::vector<double>>& betaMatrix);
  void getHmmAlignmentInfo(PositionIndex ip, PositionIndex slen, PositionIndex i, HmmAligInfo& hmmAligInfo);
  bool isValidAlignment(PositionIndex ip, PositionIndex slen, PositionIndex i);
  bool isNullAlignment(PositionIndex ip, PositionIndex slen, PositionIndex i);
  PositionIndex getModifiedIp(PositionIndex ip, PositionIndex slen, PositionIndex i);
  void batchUpdateCounts(const std::vector<std::pair<std::vector<WordIndex>, std::vector<WordIndex>>>& pairs) override;
  void batchMaximizeProbs() override;

  // Auxiliary functions to load and print models
  bool loadLexSmIntFactor(const char* lexSmIntFactorFile, int verbose);
  bool printLexSmIntFactor(const char* lexSmIntFactorFile, int verbose);
  bool loadAlSmIntFactor(const char* alSmIntFactorFile, int verbose);
  bool printAlSmIntFactor(const char* alSmIntFactorFile, int verbose);
  bool loadHmmP0(const char* hmmP0FileName, int verbose);
  bool printHmmP0(const char* hmmP0FileName);

  double aligSmoothInterpFactor = DEFAULT_ALIG_SMOOTH_INTERP_FACTOR;
  double lexSmoothInterpFactor = DEFAULT_LEX_SMOOTH_INTERP_FACTOR;

  CachedHmmAligLgProb cachedAligLogProbs;

  // model parameters
  std::shared_ptr<HmmAlignmentTable> hmmAlignmentTable;
  Prob hmm_p0 = DEFAULT_HMM_P0;

  // EM counts
  HmmAlignmentCounts hmmAlignmentCounts;
};
