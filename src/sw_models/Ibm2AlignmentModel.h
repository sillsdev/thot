#pragma once

#include "sw_models/AlignmentTable.h"
#include "sw_models/Ibm1AlignmentModel.h"

#include <memory>
#include <unordered_map>

class Ibm2AlignmentModel : public Ibm1AlignmentModel
{
  friend class IncrIbm2AlignmentTrainer;

public:
  Ibm2AlignmentModel();
  Ibm2AlignmentModel(Ibm1AlignmentModel& model);

  // Returns p(i|j,slen,tlen)
  virtual Prob aProb(PositionIndex j, PositionIndex slen, PositionIndex tlen, PositionIndex i);
  // Returns log(p(i|j,slen,tlen))
  LgProb logaProb(PositionIndex j, PositionIndex slen, PositionIndex tlen, PositionIndex i);

  using AlignmentModel::getBestAlignment;
  LgProb getBestAlignment(const std::vector<WordIndex>& srcSentence, const std::vector<WordIndex>& trgSentence,
                          std::vector<PositionIndex>& bestAlignment) override;
  using AlignmentModel::getAlignmentLgProb;
  LgProb getAlignmentLgProb(const std::vector<WordIndex>& srcSentence, const std::vector<WordIndex>& trgSentence,
                            const WordAlignmentMatrix& aligMatrix, int verbose = 0) override;
  using AlignmentModel::getSumLgProb;
  LgProb getSumLgProb(const std::vector<WordIndex>& srcSentence, const std::vector<WordIndex>& trgSentence,
                      int verbose = 0) override;

  bool load(const char* prefFileName, int verbose = 0) override;
  bool print(const char* prefFileName, int verbose = 0) override;

  void clear() override;
  void clearTempVars() override;

  virtual ~Ibm2AlignmentModel()
  {
  }

protected:
  typedef std::vector<double> AlignmentCountsElem;
  typedef OrderedVector<AlignmentKey, AlignmentCountsElem> AlignmentCounts;

  Ibm2AlignmentModel(Ibm2AlignmentModel& model);

  virtual double unsmoothed_aProb(PositionIndex j, PositionIndex slen, PositionIndex tlen, PositionIndex i);
  double unsmoothed_logaProb(PositionIndex j, PositionIndex slen, PositionIndex tlen, PositionIndex i);

  LgProb getIbm2BestAlignment(const std::vector<WordIndex>& nSrcSentIndexVector,
                              const std::vector<WordIndex>& trgSentIndexVector, std::vector<PositionIndex>& bestAlig);
  LgProb getIbm2AlignmentLgProb(const std::vector<WordIndex>& nsSent, const std::vector<WordIndex>& tSent,
                                const std::vector<PositionIndex>& alig, int verbose = 0);
  LgProb getIbm2SumLgProb(const std::vector<WordIndex>& nsSent, const std::vector<WordIndex>& tSent, int verbose = 0);

  void initTargetWord(const std::vector<WordIndex>& nsrc, const std::vector<WordIndex>& trg, PositionIndex j) override;
  double getCountNumerator(const std::vector<WordIndex>& nsrcSent, const std::vector<WordIndex>& trgSent,
                           unsigned int i, unsigned int j) override;
  void incrementWordPairCounts(const std::vector<WordIndex>& nsrc, const std::vector<WordIndex>& trg, PositionIndex i,
                               PositionIndex j, double count) override;
  void batchMaximizeProbs() override;

  // model parameters
  std::shared_ptr<AlignmentTable> alignmentTable;

  // EM counts
  AlignmentCounts alignmentCounts;
};
