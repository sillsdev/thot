#pragma once

#include "nlp_common/Matrix.h"
#include "sw_models/AlignmentInfo.h"
#include "sw_models/DistortionTable.h"
#include "sw_models/FertilityTable.h"
#include "sw_models/HmmAlignmentModel.h"
#include "sw_models/Ibm2AlignmentModel.h"

#include <functional>
#include <memory>

class Ibm3AlignmentModel : public Ibm2AlignmentModel
{
  friend class Ibm4AlignmentModel;

public:
  Ibm3AlignmentModel();
  Ibm3AlignmentModel(Ibm2AlignmentModel& model);
  Ibm3AlignmentModel(HmmAlignmentModel& model);
  Ibm3AlignmentModel(Ibm3AlignmentModel& model);

  AlignmentModelType getModelType() const override
  {
    return Ibm3;
  }

  double getCountThreshold() const;
  void setCountThreshold(double threshold);

  double getFertilitySmoothFactor() const;
  void setFertilitySmoothFactor(double factor);

  unsigned int startTraining(int verbosity = 0) override;
  void train(int verbosity = 0) override;

  Prob distortionProb(PositionIndex i, PositionIndex slen, PositionIndex tlen, PositionIndex j);
  LgProb distortionLogProb(PositionIndex i, PositionIndex slen, PositionIndex tlen, PositionIndex j);

  Prob fertilityProb(WordIndex s, PositionIndex phi);
  LgProb fertilityLogProb(WordIndex s, PositionIndex phi);

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

  virtual ~Ibm3AlignmentModel()
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
  const PositionIndex MaxSentenceLength = 200;
  const double DefaultCountThreshold = 1e-5;
  const double DefaultP1 = 0.05;
  const double DefaultFertilitySmoothFactor = 64.0;

  std::string getModelTypeStr() const override
  {
    return "ibm3";
  }

  double unsmoothedDistortionLogProb(PositionIndex i, PositionIndex slen, PositionIndex tlen, PositionIndex j);
  double unsmoothedFertilityLogProb(WordIndex s, PositionIndex phi);

  Prob searchForBestAlignment(const std::vector<WordIndex>& nsrc, const std::vector<WordIndex>& trg,
                              AlignmentInfo& bestAlignment, Matrix<double>* moveScores = nullptr,
                              Matrix<double>* swapScores = nullptr);
  void getInitialAlignmentForSearch(const std::vector<WordIndex>& nsrc, const std::vector<WordIndex>& trg,
                                    AlignmentInfo& alignment);
  virtual Prob calcProbOfAlignment(const std::vector<WordIndex>& nsrc, const std::vector<WordIndex>& trg,
                                   AlignmentInfo& alignment, int verbose = 0);
  virtual double swapScore(const std::vector<WordIndex>& nsrc, const std::vector<WordIndex>& trg, PositionIndex j1,
                           PositionIndex j2, AlignmentInfo& alignment, double& cachedAlignmentValue);
  virtual double moveScore(const std::vector<WordIndex>& nsrc, const std::vector<WordIndex>& trg, PositionIndex iNew,
                           PositionIndex j, AlignmentInfo& alignment, double& cachedAlignmentValue);

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
  virtual double updateCounts(const std::vector<WordIndex>& nsrc, const std::vector<WordIndex>& trg,
                              AlignmentInfo& alignment, double aligProb, const Matrix<double>& moveScores,
                              const Matrix<double>& swapScores);
  void batchMaximizeProbs() override;

  bool loadP1(const std::string& filename);
  bool printP1(const std::string& filename);

  void loadConfig(const YAML::Node& config) override;
  void createConfig(YAML::Emitter& out) override;

  double countThreshold = DefaultCountThreshold;
  double fertilitySmoothFactor = DefaultFertilitySmoothFactor;

  // model parameters
  std::shared_ptr<Prob> p1;
  std::shared_ptr<DistortionTable> distortionTable;
  std::shared_ptr<FertilityTable> fertilityTable;

  // EM counts
  DistortionCounts distortionCounts;
  FertilityCounts fertilityCounts;
  double p0Count = 0;
  double p1Count = 0;

  size_t maxSrcWordLen = 0;

  bool performIbm2Transfer = false;
  std::unique_ptr<HmmAlignmentModel> hmmModel;
  CachedHmmAligLgProb cachedHmmAligLogProbs;
};
