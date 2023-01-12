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
  Ibm2AlignmentModel(Ibm2AlignmentModel& model);

  AlignmentModelType getModelType() const override
  {
    return Ibm2;
  }

  bool getCompactAlignmentTable() const;
  void setCompactAlignmentTable(bool value);

  // Returns p(i|j,slen,tlen)
  virtual Prob alignmentProb(PositionIndex j, PositionIndex slen, PositionIndex tlen, PositionIndex i);
  // Returns log(p(i|j,slen,tlen))
  LgProb alignmentLogProb(PositionIndex j, PositionIndex slen, PositionIndex tlen, PositionIndex i);

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

  virtual ~Ibm2AlignmentModel()
  {
  }

protected:
  typedef std::vector<double> AlignmentCountsElem;
  typedef OrderedVector<AlignmentKey, AlignmentCountsElem> AlignmentCounts;

  std::string getModelTypeStr() const override
  {
    return "ibm2";
  }

  double unsmoothedAlignmentLogProb(PositionIndex j, PositionIndex slen, PositionIndex tlen, PositionIndex i);

  LgProb getIbm2BestAlignment(const std::vector<WordIndex>& nSrcSentIndexVector,
                              const std::vector<WordIndex>& trgSentIndexVector, std::vector<PositionIndex>& bestAlig);
  LgProb computeIbm2LogProb(const std::vector<WordIndex>& nsSent, const std::vector<WordIndex>& tSent,
                            const std::vector<PositionIndex>& alig, int verbose = 0);
  LgProb getIbm2SumLogProb(const std::vector<WordIndex>& nsSent, const std::vector<WordIndex>& tSent, int verbose = 0);

  void initTargetWord(const std::vector<WordIndex>& nsrc, const std::vector<WordIndex>& trg, PositionIndex j) override;
  double getCountNumerator(const std::vector<WordIndex>& nsrcSent, const std::vector<WordIndex>& trgSent,
                           unsigned int i, unsigned int j) override;
  void incrementWordPairCounts(const std::vector<WordIndex>& nsrc, const std::vector<WordIndex>& trg, PositionIndex i,
                               PositionIndex j, double count) override;
  void batchMaximizeProbs() override;
  PositionIndex getCompactedSentenceLength(PositionIndex len);

  void loadConfig(const YAML::Node& config) override;
  bool loadOldConfig(const char* prefFileName, int verbose = 0) override;
  void createConfig(YAML::Emitter& out) override;

  bool compactAlignmentTable = true;

  // model parameters
  std::shared_ptr<AlignmentTable> alignmentTable;

  // EM counts
  AlignmentCounts alignmentCounts;
};
