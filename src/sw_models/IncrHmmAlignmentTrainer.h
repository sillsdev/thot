#pragma once

#include "sw_models/HmmAlignmentModel.h"
#include "sw_models/anjiMatrix.h"
#include "sw_models/anjm1ip_anjiMatrix.h"

class IncrHmmAlignmentCountsKeyHash
{
public:
  enum
  {
    bucket_size = 1
  };

  std::size_t operator()(const std::pair<HmmAlignmentKey, PositionIndex>& a1) const
  {
    return (size_t)(a1.second * 16384) + ((size_t)256 * a1.first.prev_i) + a1.first.slen;
  }

  bool operator()(const std::pair<HmmAlignmentKey, PositionIndex>& left,
                  const std::pair<HmmAlignmentKey, PositionIndex>& right) const
  {
    return left < right;
  }
};

class IncrHmmAlignmentTrainer
{
public:
  IncrHmmAlignmentTrainer(HmmAlignmentModel& model, anjiMatrix& lanji, anjm1ip_anjiMatrix& lanjm1ip_anji);

  void incrTrain(std::pair<unsigned int, unsigned int> sentPairRange, int verbosity);
  void clear();

  virtual ~IncrHmmAlignmentTrainer()
  {
  }

protected:
  void calcNewLocalSuffStats(std::pair<unsigned int, unsigned int> sentPairRange, int verbosity = 0);
  void calcNewLocalSuffStatsVit(std::pair<unsigned int, unsigned int> sentPairRange, int verbosity = 0);
  void calc_lanji(unsigned int n, const std::vector<WordIndex>& nsrcSent, const std::vector<WordIndex>& trgSent,
                  PositionIndex slen, const Count& weight, const std::vector<std::vector<double>>& alphaMatrix,
                  const std::vector<std::vector<double>>& betaMatrix);
  void calc_lanji_vit(unsigned int n, const std::vector<WordIndex>& nsrcSent, const std::vector<WordIndex>& trgSent,
                      const std::vector<PositionIndex>& bestAlig, const Count& weight);
  void calc_lanjm1ip_anji(unsigned int n, const std::vector<WordIndex>& srcSent, const std::vector<WordIndex>& trgSent,
                          PositionIndex slen, const Count& weight, const std::vector<std::vector<double>>& logProbs,
                          const std::vector<std::vector<double>>& alignProbs,
                          const std::vector<std::vector<double>>& alphaMatrix,
                          const std::vector<std::vector<double>>& betaMatrix);
  void calc_lanjm1ip_anji_vit(unsigned int n, const std::vector<WordIndex>& srcSent,
                              const std::vector<WordIndex>& trgSent, PositionIndex slen,
                              const std::vector<PositionIndex>& bestAlig, const Count& weight);
  void gatherLexSuffStats(unsigned int mapped_n, unsigned int mapped_n_aux, const std::vector<WordIndex>& nsrcSent,
                          const std::vector<WordIndex>& trgSent, const Count& weight);
  void incrUpdateCountsLex(unsigned int mapped_n, unsigned int mapped_n_aux, PositionIndex i, PositionIndex j,
                           const std::vector<WordIndex>& nsrcSent, const std::vector<WordIndex>& trgSent,
                           const Count& weight);
  void gatherAligSuffStats(unsigned int mapped_n, unsigned int mapped_n_aux, const std::vector<WordIndex>& srcSent,
                           const std::vector<WordIndex>& trgSent, PositionIndex slen, const Count& weight);
  void incrUpdateCountsAlig(unsigned int mapped_n, unsigned int mapped_n_aux, PositionIndex slen, PositionIndex ip,
                            PositionIndex i, PositionIndex j, const Count& weight);
  void incrMaximizeProbs();
  float obtainLogNewSuffStat(float lcurrSuffStat, float lLocalSuffStatCurr, float lLocalSuffStatNew);

private:
  typedef std::pair<HmmAlignmentKey, PositionIndex> IncrHmmAlignmentCountsKey;
  typedef std::unordered_map<IncrHmmAlignmentCountsKey, std::pair<float, float>, IncrHmmAlignmentCountsKeyHash>
      IncrHmmAlignmentCounts;

  anjiMatrix& lanji;
  anjiMatrix lanji_aux;
  anjm1ip_anjiMatrix& lanjm1ip_anji;
  anjm1ip_anjiMatrix lanjm1ip_anji_aux;

  HmmAlignmentModel& model;
  IncrLexCounts incrLexCounts;
  IncrHmmAlignmentCounts incrHmmAlignmentCounts;
};
