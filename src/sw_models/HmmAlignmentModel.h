#pragma once

#include "nlp_common/Matrix.h"
#include "sw_models/AlignmentInfo.h"
#include "sw_models/CachedHmmAligLgProb.h"
#include "sw_models/HmmAlignmentTable.h"
#include "sw_models/Ibm2AlignmentModel.h"

#include <memory>
#include <unordered_map>

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

class HmmAlignmentModel : public Ibm2AlignmentModel
{
  friend class IncrHmmAlignmentTrainer;
  friend class Ibm3AlignmentModel;

public:
  HmmAlignmentModel();
  HmmAlignmentModel(Ibm1AlignmentModel& model);
  HmmAlignmentModel(HmmAlignmentModel& model);

  AlignmentModelType getModelType() const override
  {
    return Hmm;
  }

  // Get/set lexical smoothing interpolation factor
  double getLexicalSmoothFactor();
  void setLexicalSmoothFactor(double factor);
  // Get/set alignment smoothing interpolation factor
  double getHmmAlignmentSmoothFactor();
  void setHmmAlignmentSmoothFactor(double factor);

  // Get/set p0
  Prob getHmmP0();
  void setHmmP0(Prob p0);

  unsigned int startTraining(int verbosity = 0) override;

  // returns p(t|s)
  Prob translationProb(WordIndex s, WordIndex t) override;
  // returns log(p(t|s))
  LgProb translationLogProb(WordIndex s, WordIndex t) override;
  // Returns p(i|prev_i,slen)
  Prob hmmAlignmentProb(PositionIndex prev_i, PositionIndex slen, PositionIndex i);
  // Returns log(p(i|prev_i,slen))
  LgProb hmmAlignmentLogProb(PositionIndex prev_i, PositionIndex slen, PositionIndex i);

  using AlignmentModel::getBestAlignment;
  LgProb getBestAlignment(const std::vector<WordIndex>& srcSentence, const std::vector<WordIndex>& trgSentence,
                          std::vector<PositionIndex>& bestAlignment) override;
  using AlignmentModel::computeLogProb;
  LgProb computeLogProb(const std::vector<WordIndex>& srcSentence, const std::vector<WordIndex>& trgSentence,
                        const WordAlignmentMatrix& aligMatrix, int verbose = 0) override;
  using AlignmentModel::computeSumLogProb;
  LgProb computeSumLogProb(const std::vector<WordIndex>& srcSentence, const std::vector<WordIndex>& trgSentence,
                           int verbose = 0) override;

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

  const double ExpValMax = exp(-0.01);
  const double ExpValMin = exp(-9);
  const PositionIndex MaxSentenceLength = 200;
  const double DefaultHmmAlignmentSmoothFactor = 0.2;
  const double DefaultLexicalSmoothFactor = 0.0;
  const double DefaultHmmP0 = 0.4;

  std::string getModelTypeStr() const override
  {
    return "hmm";
  }

  Prob searchForBestAlignment(const std::vector<WordIndex>& src, const std::vector<WordIndex>& trg,
                              AlignmentInfo& bestAlignment, CachedHmmAligLgProb& cachedAligLogProbs);
  void populateMoveSwapScores(const std::vector<WordIndex>& src, const std::vector<WordIndex>& trg,
                              AlignmentInfo& bestAlignment, double alignmentProb,
                              CachedHmmAligLgProb& cachedAligLogProbs, Matrix<double>& moveScores,
                              Matrix<double>& swapScores);

  double unsmoothedHmmAlignmentLogProb(PositionIndex prev_i, PositionIndex slen, PositionIndex i);
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
                             PositionIndex slen, std::vector<std::vector<double>>& lexProbs,
                             std::vector<std::vector<double>>& alignProbs,
                             std::vector<std::vector<double>>& alphaMatrix,
                             std::vector<std::vector<double>>& betaMatrix);
  PositionIndex getSrcLen(const std::vector<WordIndex>& nsrcWordIndexVec);
  Prob calcProbOfAlignment(CachedHmmAligLgProb& cached_logap, const std::vector<WordIndex>& nsrc,
                           const std::vector<WordIndex>& trg, AlignmentInfo& alignment, int verbose = 0);
  double swapScore(CachedHmmAligLgProb& cached_logap, const std::vector<WordIndex>& nsrc,
                   const std::vector<WordIndex>& trg, PositionIndex j1, PositionIndex j2, AlignmentInfo& alignment,
                   double alignmentProb);
  double moveScore(CachedHmmAligLgProb& cached_logap, const std::vector<WordIndex>& nsrc,
                   const std::vector<WordIndex>& trg, PositionIndex iNew, PositionIndex j, AlignmentInfo& alignment,
                   double alignmentProb);

  bool isFirstNullAlignmentPar(PositionIndex ip, unsigned int slen, PositionIndex i);
  void getHmmAlignmentInfo(PositionIndex ip, PositionIndex slen, PositionIndex i, HmmAligInfo& hmmAligInfo);
  bool isValidAlignment(PositionIndex ip, PositionIndex slen, PositionIndex i);
  bool isNullAlignment(PositionIndex ip, PositionIndex slen, PositionIndex i);
  PositionIndex getModifiedIp(PositionIndex ip, PositionIndex slen, PositionIndex i);
  void batchUpdateCounts(const std::vector<std::pair<std::vector<WordIndex>, std::vector<WordIndex>>>& pairs) override;
  void batchMaximizeProbs() override;

  // Auxiliary functions to load and print models
  bool loadLexSmIntFactor(const char* lexSmIntFactorFile, int verbose);
  bool loadAlSmIntFactor(const char* alSmIntFactorFile, int verbose);
  bool loadHmmP0(const char* hmmP0FileName, int verbose);

  void loadConfig(const YAML::Node& config) override;
  bool loadOldConfig(const char* prefFileName, int verbose = 0) override;
  void createConfig(YAML::Emitter& out) override;

  double hmmAlignmentSmoothFactor = DefaultHmmAlignmentSmoothFactor;
  double lexicalSmoothFactor = DefaultLexicalSmoothFactor;

  Prob hmmP0 = DefaultHmmP0;

  // model parameters
  std::shared_ptr<HmmAlignmentTable> hmmAlignmentTable;

  // EM counts
  HmmAlignmentCounts hmmAlignmentCounts;
};
