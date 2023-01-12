#pragma once

#include "sw_models/HeadDistortionTable.h"
#include "sw_models/HmmAlignmentModel.h"
#include "sw_models/Ibm3AlignmentModel.h"
#include "sw_models/NonheadDistortionTable.h"

#include <memory>

class Ibm4AlignmentModel : public Ibm3AlignmentModel
{
  friend class Ibm4AlignmentModelTest;

public:
  Ibm4AlignmentModel();
  Ibm4AlignmentModel(HmmAlignmentModel& model);
  Ibm4AlignmentModel(Ibm3AlignmentModel& model);
  Ibm4AlignmentModel(Ibm4AlignmentModel& model);

  AlignmentModelType getModelType() const override
  {
    return Ibm4;
  }

  unsigned int startTraining(int verbosity = 0) override;
  void train(int verbosity = 0) override;

  Prob headDistortionProb(WordClassIndex srcWordClass, WordClassIndex trgWordClass, PositionIndex tlen, int dj);
  LgProb headDistortionLogProb(WordClassIndex srcWordClass, WordClassIndex trgWordClass, PositionIndex tlen, int dj);

  Prob nonheadDistortionProb(WordClassIndex trgWordClass, PositionIndex tlen, int dj);
  LgProb nonheadDistortionLogProb(WordClassIndex trgWordClass, PositionIndex tlen, int dj);

  using AlignmentModel::computeSumLogProb;
  LgProb computeSumLogProb(const std::vector<WordIndex>& srcSentence, const std::vector<WordIndex>& trgSentence,
                           int verbose = 0) override;

  double getDistortionSmoothFactor();
  void setDistortionSmoothFactor(double distortionSmoothFactor);

  bool load(const char* prefFileName, int verbose = 0) override;
  bool print(const char* prefFileName, int verbose = 0) override;

  void clear() override;
  void clearTempVars() override;

  virtual ~Ibm4AlignmentModel()
  {
  }

protected:
  typedef OrderedVector<int, double> HeadDistortionCountsElem;
  typedef OrderedVector<HeadDistortionKey, HeadDistortionCountsElem> HeadDistortionCounts;
  typedef OrderedVector<int, double> NonheadDistortionCountsElem;
  typedef std::vector<NonheadDistortionCountsElem> NonheadDistortionCounts;

  const double DefaultDistortionSmoothFactor = 0.2;

  std::string getModelTypeStr() const override
  {
    return "ibm4";
  }

  double unsmoothedHeadDistortionLogProb(WordClassIndex srcWordClass, WordClassIndex trgWordClass, int dj, bool& found);
  double unsmoothedNonheadDistortionLogProb(WordClassIndex trgWordClass, int dj, bool& found);

  Prob calcProbOfAlignment(const std::vector<WordIndex>& nsrc, const std::vector<WordIndex>& trg,
                           AlignmentInfo& alignment, int verbose = 0) override;
  Prob calcDistortionProbOfAlignment(const std::vector<WordIndex>& nsrc, const std::vector<WordIndex>& trg,
                                     AlignmentInfo& alignment);
  double swapScore(const std::vector<WordIndex>& nsrc, const std::vector<WordIndex>& trg, PositionIndex j1,
                   PositionIndex j2, AlignmentInfo& alignment, double& cachedAlignmentValue) override;
  double moveScore(const std::vector<WordIndex>& nsrc, const std::vector<WordIndex>& trg, PositionIndex iNew,
                   PositionIndex j, AlignmentInfo& alignment, double& cachedAlignmentValue) override;

  void ibm3Transfer();

  // batch EM functions
  void initWordPair(const std::vector<WordIndex>& nsrc, const std::vector<WordIndex>& trg, PositionIndex i,
                    PositionIndex j) override;
  double updateCounts(const std::vector<WordIndex>& nsrc, const std::vector<WordIndex>& trg, AlignmentInfo& alignment,
                      double aligProb, const Matrix<double>& moveScores, const Matrix<double>& swapScores) override;
  void incrementDistortionCounts(const std::vector<WordIndex>& nsrc, const std::vector<WordIndex>& trg,
                                 const AlignmentInfo& alignment, double count);
  void batchMaximizeProbs() override;

  void loadConfig(const YAML::Node& config) override;
  void createConfig(YAML::Emitter& out) override;

  double distortionSmoothFactor = DefaultDistortionSmoothFactor;

  // model parameters
  std::shared_ptr<HeadDistortionTable> headDistortionTable;
  std::shared_ptr<NonheadDistortionTable> nonheadDistortionTable;

  // EM counts
  HeadDistortionCounts headDistortionCounts;
  NonheadDistortionCounts nonheadDistortionCounts;

  std::unique_ptr<Ibm3AlignmentModel> ibm3Model;
};
