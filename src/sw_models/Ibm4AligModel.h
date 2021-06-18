#ifndef _Ibm4AligModel_h
#define _Ibm4AligModel_h

#include "sw_models/HeadDistortionTable.h"
#include "sw_models/Ibm3AligModel.h"
#include "sw_models/NonheadDistortionTable.h"

class Ibm4AligModel : public Ibm3AligModel
{
public:
  Prob headDistortionProb(WordClassIndex srcWordClass, WordClassIndex trgWordClass, int dj);
  LgProb logHeadDistortionProb(WordClassIndex srcWordClass, WordClassIndex trgWordClass, int dj);

  Prob nonheadDistortionProb(WordClassIndex trgWordClass, int dj);
  LgProb logNonheadDistortionProb(WordClassIndex trgWordClass, int dj);

  LgProb calcLgProb(const std::vector<WordIndex>& src, const std::vector<WordIndex>& trg, int verbose = 0);

  bool load(const char* prefFileName, int verbose = 0);
  bool print(const char* prefFileName, int verbose = 0);

  void clear();
  void clearTempVars();
  void clearInfoAboutSentRange();

  ~Ibm4AligModel();

protected:
  typedef OrderedVector<int, double> HeadDistortionCountsElem;
  typedef OrderedVector<HeadDistortionTableKey, HeadDistortionCountsElem> HeadDistortionCounts;
  typedef OrderedVector<int, double> NonheadDistortionCountsElem;
  typedef std::vector<NonheadDistortionCountsElem> NonheadDistortionCounts;

  bool sentenceLengthIsOk(const std::vector<WordIndex> sentence);

  double unsmoothedHeadDistortionProb(WordClassIndex srcWordClass, WordClassIndex trgWordClass, int dj);
  double unsmoothedLogHeadDistortionProb(WordClassIndex srcWordClass, WordClassIndex trgWordClass, int dj);
  double headDistortionProb(WordClassIndex srcWordClass, WordClassIndex trgWordClass, int dj, bool training);

  double unsmoothedNonheadDistortionProb(WordClassIndex trgWordClass, int dj);
  double unsmoothedLogNonheadDistortionProb(WordClassIndex trgWordClass, int dj);
  double nonheadDistortionProb(WordClassIndex trgWordClass, int dj, bool training);

  Prob calcProbOfAlignment(const std::vector<WordIndex>& nsrc, const std::vector<WordIndex>& trg, bool training,
                           AlignmentInfo& alignment, int verbose = 0);
  double swapScore(const Sentence& nsrc, const Sentence& trg, PositionIndex j1, PositionIndex j2, bool training,
                   AlignmentInfo& alignment);
  double moveScore(const Sentence& nsrc, const Sentence& trg, PositionIndex iNew, PositionIndex j, bool training,
                   AlignmentInfo& alignment);

  // batch EM functions
  void initialBatchPass(std::pair<unsigned int, unsigned int> sentPairRange);
  void initWordPair(const Sentence& nsrc, const Sentence& trg, PositionIndex i, PositionIndex j);
  void incrementTargetWordCounts(const Sentence& nsrc, const Sentence& trg, const AlignmentInfo& alignment,
                                 PositionIndex j, double count);
  void batchMaximizeProbs();

  HeadDistortionTable headDistortionTable;
  NonheadDistortionTable nonheadDistortionTable;

  HeadDistortionCounts headDistortionCounts;
  NonheadDistortionCounts nonheadDistortionCounts;

  std::vector<WordClassIndex> srcWordClasses;
  std::vector<WordClassIndex> trgWordClasses;
};

#endif
