#pragma once

#include "nlp_common/Matrix.h"
#include "sw_models/AlignmentInfo.h"
#include "sw_models/DistortionTable.h"
#include "sw_models/FertilityTable.h"
#include "sw_models/HmmAligModel.h"
#include "sw_models/Ibm2AligModel.h"

#include <functional>
#include <memory>

class Ibm3AligModel : public Ibm2AligModel
{
public:
  Ibm3AligModel();
  Ibm3AligModel(Ibm2AligModel& model);
  Ibm3AligModel(HmmAligModel& model);

  void startTraining(int verbosity = 0) override;

  Prob distortionProb(PositionIndex i, PositionIndex slen, PositionIndex tlen, PositionIndex j);
  LgProb logDistortionProb(PositionIndex i, PositionIndex slen, PositionIndex tlen, PositionIndex j);

  Prob fertilityProb(WordIndex s, PositionIndex phi);
  LgProb logFertilityProb(WordIndex s, PositionIndex phi);

  LgProb obtainBestAlignment(const std::vector<WordIndex>& src, const std::vector<WordIndex>& trg,
                             WordAligMatrix& bestWaMatrix) override;
  LgProb calcLgProbForAlig(const std::vector<WordIndex>& src, const std::vector<WordIndex>& trg,
                           const WordAligMatrix& aligMatrix, int verbose = 0) override;
  LgProb calcLgProb(const std::vector<WordIndex>& src, const std::vector<WordIndex>& trg, int verbose = 0) override;

  bool load(const char* prefFileName, int verbose = 0) override;
  bool print(const char* prefFileName, int verbose = 0) override;

  void clear() override;
  void clearTempVars() override;

  virtual ~Ibm3AligModel()
  {
  }

protected:
  typedef std::vector<double> DistortionCountsElem;
  typedef OrderedVector<DistortionKey, DistortionCountsElem> DistortionCounts;
  typedef std::vector<double> FertilityCountsElem;
  typedef std::vector<FertilityCountsElem> FertilityCounts;
  typedef std::function<Prob(const std::vector<WordIndex>&, const std::vector<WordIndex>&, AlignmentInfo&,
                             Matrix<double>&, Matrix<double>&)>
      SearchForBestAlignmentFunc;

  const PositionIndex MaxFertility = 10;

  Ibm3AligModel(Ibm3AligModel& model);

  double unsmoothedDistortionProb(PositionIndex i, PositionIndex slen, PositionIndex tlen, PositionIndex j);
  double unsmoothedLogDistortionProb(PositionIndex i, PositionIndex slen, PositionIndex tlen, PositionIndex j);

  double unsmoothedFertilityProb(WordIndex s, PositionIndex phi);
  double unsmoothedLogFertilityProb(WordIndex s, PositionIndex phi);

  Prob searchForBestAlignment(const std::vector<WordIndex>& nsrc, const std::vector<WordIndex>& trg,
                              AlignmentInfo& bestAlignment, Matrix<double>* moveScores = nullptr,
                              Matrix<double>* swapScores = nullptr);
  void getInitialAlignmentForSearch(const std::vector<WordIndex>& nsrc, const std::vector<WordIndex>& trg,
                                    AlignmentInfo& alignment);
  virtual Prob calcProbOfAlignment(const std::vector<WordIndex>& nsrc, const std::vector<WordIndex>& trg,
                                   AlignmentInfo& alignment, int verbose = 0);
  virtual double swapScore(const std::vector<WordIndex>& nsrc, const std::vector<WordIndex>& trg, PositionIndex j1,
                           PositionIndex j2, AlignmentInfo& alignment);
  virtual double moveScore(const std::vector<WordIndex>& nsrc, const std::vector<WordIndex>& trg, PositionIndex iNew,
                           PositionIndex j, AlignmentInfo& alignment);

  // batch EM functions
  void ibm2Transfer();
  void ibm2TransferUpdateCounts(const std::vector<std::pair<std::vector<WordIndex>, std::vector<WordIndex>>>& pairs);
  void hmmTransfer();
  double getSumOfPartitions(PositionIndex phi, PositionIndex i, const Matrix<double>& alpha);
  void initSentencePair(const std::vector<WordIndex>& src, const std::vector<WordIndex>& trg) override;
  void initSourceWord(const std::vector<WordIndex>& nsrc, const std::vector<WordIndex>& trg, PositionIndex i) override;
  void addTranslationOptions(std::vector<std::vector<WordIndex>>& insertBuffer) override;
  void batchUpdateCounts(const std::vector<std::pair<std::vector<WordIndex>, std::vector<WordIndex>>>& pairs) override;
  void batchUpdateCounts(const std::vector<std::pair<std::vector<WordIndex>, std::vector<WordIndex>>>& pairs,
                         SearchForBestAlignmentFunc search);
  void incrementWordPairCounts(const std::vector<WordIndex>& nsrc, const std::vector<WordIndex>& trg, PositionIndex i,
                               PositionIndex j, double count) override;
  virtual void incrementTargetWordCounts(const std::vector<WordIndex>& nsrc, const std::vector<WordIndex>& trg,
                                         const AlignmentInfo& alignment, PositionIndex j, double count);
  void batchMaximizeProbs() override;

  // model parameters
  Prob p1 = 0.5;
  std::shared_ptr<DistortionTable> distortionTable;
  std::shared_ptr<FertilityTable> fertilityTable;

  // EM counts
  DistortionCounts distortionCounts;
  FertilityCounts fertilityCounts;
  double p0Count = 0;
  double p1Count = 0;

  bool performIbm2Transfer = false;
  std::unique_ptr<HmmAligModel> hmmModel;
};
