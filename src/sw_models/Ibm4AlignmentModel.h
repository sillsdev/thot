#pragma once

#include "nlp_common/WordClasses.h"
#include "sw_models/HeadDistortionTable.h"
#include "sw_models/Ibm3AlignmentModel.h"
#include "sw_models/NonheadDistortionTable.h"

#include <memory>

class Ibm4AlignmentModel : public Ibm3AlignmentModel
{
  friend class Ibm4AlignmentModelTest;

public:
  Ibm4AlignmentModel();
  Ibm4AlignmentModel(Ibm3AlignmentModel& model);

  void startTraining(int verbosity = 0) override;

  Prob headDistortionProb(WordClassIndex srcWordClass, WordClassIndex trgWordClass, PositionIndex tlen, int dj);
  LgProb logHeadDistortionProb(WordClassIndex srcWordClass, WordClassIndex trgWordClass, PositionIndex tlen, int dj);

  Prob nonheadDistortionProb(WordClassIndex trgWordClass, PositionIndex tlen, int dj);
  LgProb logNonheadDistortionProb(WordClassIndex trgWordClass, PositionIndex tlen, int dj);

  using AlignmentModel::getSumLgProb;
  LgProb getSumLgProb(const std::vector<WordIndex>& srcSentence, const std::vector<WordIndex>& trgSentence,
                      int verbose = 0) override;

  void setDistortionSmoothFactor(double distortionSmoothFactor, int verbose = 0);

  void addSrcWordClass(WordIndex s, WordClassIndex c);
  void addTrgWordClass(WordIndex t, WordClassIndex c);

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

  Ibm4AlignmentModel(Ibm4AlignmentModel& model);

  bool sentenceLengthIsOk(const std::vector<WordIndex> sentence) override;

  double unsmoothedHeadDistortionProb(WordClassIndex srcWordClass, WordClassIndex trgWordClass, int dj);
  double unsmoothedLogHeadDistortionProb(WordClassIndex srcWordClass, WordClassIndex trgWordClass, int dj);

  double unsmoothedNonheadDistortionProb(WordClassIndex trgWordClass, int dj);
  double unsmoothedLogNonheadDistortionProb(WordClassIndex trgWordClass, int dj);

  Prob calcProbOfAlignment(const std::vector<WordIndex>& nsrc, const std::vector<WordIndex>& trg,
                           AlignmentInfo& alignment, int verbose = 0) override;
  double swapScore(const std::vector<WordIndex>& nsrc, const std::vector<WordIndex>& trg, PositionIndex j1,
                   PositionIndex j2, AlignmentInfo& alignment) override;
  double moveScore(const std::vector<WordIndex>& nsrc, const std::vector<WordIndex>& trg, PositionIndex iNew,
                   PositionIndex j, AlignmentInfo& alignment) override;

  // batch EM functions
  void initWordPair(const std::vector<WordIndex>& nsrc, const std::vector<WordIndex>& trg, PositionIndex i,
                    PositionIndex j) override;
  void incrementTargetWordCounts(const std::vector<WordIndex>& nsrc, const std::vector<WordIndex>& trg,
                                 const AlignmentInfo& alignment, PositionIndex j, double count) override;
  void batchMaximizeProbs() override;

  bool loadDistortionSmoothFactor(const char* distortionSmoothFactorFile, int verbose);
  bool printDistortionSmoothFactor(const char* distortionSmoothFactorFile, int verbose);

  double distortionSmoothFactor;

  std::shared_ptr<WordClasses> wordClasses;

  // model parameters
  std::shared_ptr<HeadDistortionTable> headDistortionTable;
  std::shared_ptr<NonheadDistortionTable> nonheadDistortionTable;

  // EM counts
  HeadDistortionCounts headDistortionCounts;
  NonheadDistortionCounts nonheadDistortionCounts;
};
