#pragma once

#include "nlp_common/WordClasses.h"
#include "sw_models/HeadDistortionTable.h"
#include "sw_models/Ibm3AligModel.h"
#include "sw_models/NonheadDistortionTable.h"

class Ibm4AligModel : public Ibm3AligModel
{
  friend class Ibm4AligModelTest;

public:
  Ibm4AligModel();

  Prob headDistortionProb(WordClassIndex srcWordClass, WordClassIndex trgWordClass, PositionIndex tlen, int dj);
  LgProb logHeadDistortionProb(WordClassIndex srcWordClass, WordClassIndex trgWordClass, PositionIndex tlen, int dj);

  Prob nonheadDistortionProb(WordClassIndex trgWordClass, PositionIndex tlen, int dj);
  LgProb logNonheadDistortionProb(WordClassIndex trgWordClass, PositionIndex tlen, int dj);

  LgProb calcLgProb(const std::vector<WordIndex>& src, const std::vector<WordIndex>& trg, int verbose = 0);

  void setDistortionSmoothFactor(double distortionSmoothFactor, int verbose = 0);

  void addSrcWordClass(WordIndex s, WordClassIndex c);
  void addTrgWordClass(WordIndex t, WordClassIndex c);

  bool load(const char* prefFileName, int verbose = 0);
  bool print(const char* prefFileName, int verbose = 0);

  void clear();
  void clearTempVars();
  void clearInfoAboutSentRange();

  virtual ~Ibm4AligModel()
  {
  }

protected:
  typedef OrderedVector<int, double> HeadDistortionCountsElem;
  typedef OrderedVector<HeadDistortionKey, HeadDistortionCountsElem> HeadDistortionCounts;
  typedef OrderedVector<int, double> NonheadDistortionCountsElem;
  typedef std::vector<NonheadDistortionCountsElem> NonheadDistortionCounts;

  bool sentenceLengthIsOk(const std::vector<WordIndex> sentence);

  double unsmoothedHeadDistortionProb(WordClassIndex srcWordClass, WordClassIndex trgWordClass, int dj);
  double unsmoothedLogHeadDistortionProb(WordClassIndex srcWordClass, WordClassIndex trgWordClass, int dj);

  double unsmoothedNonheadDistortionProb(WordClassIndex trgWordClass, int dj);
  double unsmoothedLogNonheadDistortionProb(WordClassIndex trgWordClass, int dj);

  Prob calcProbOfAlignment(const std::vector<WordIndex>& nsrc, const std::vector<WordIndex>& trg,
                           AlignmentInfo& alignment, int verbose = 0);
  double swapScore(const Sentence& nsrc, const Sentence& trg, PositionIndex j1, PositionIndex j2,
                   AlignmentInfo& alignment);
  double moveScore(const Sentence& nsrc, const Sentence& trg, PositionIndex iNew, PositionIndex j,
                   AlignmentInfo& alignment);

  // batch EM functions
  void initialBatchPass(std::pair<unsigned int, unsigned int> sentPairRange);
  void initWordPair(const Sentence& nsrc, const Sentence& trg, PositionIndex i, PositionIndex j);
  void incrementTargetWordCounts(const Sentence& nsrc, const Sentence& trg, const AlignmentInfo& alignment,
                                 PositionIndex j, double count);
  void batchMaximizeProbs();

  bool loadDistortionSmoothFactor(const char* distortionSmoothFactorFile, int verbose);
  bool printDistortionSmoothFactor(const char* distortionSmoothFactorFile, int verbose);

  double distortionSmoothFactor;

  WordClasses wordClasses;

  HeadDistortionTable headDistortionTable;
  NonheadDistortionTable nonheadDistortionTable;

  HeadDistortionCounts headDistortionCounts;
  NonheadDistortionCounts nonheadDistortionCounts;
};
