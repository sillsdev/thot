#ifndef _Ibm3AligModel_h
#define _Ibm3AligModel_h

#include "AlignmentInfo.h"
#include "Ibm2AligModel.h"
#include "IncrDistortionTable.h"
#include "IncrFertilityTable.h"
#include "Matrix.h"

class Ibm3AligModel : public Ibm2AligModel
{
public:
  // Constructor
  Ibm3AligModel();

  Prob distortionProb(PositionIndex i, PositionIndex slen, PositionIndex tlen, PositionIndex j);
  LgProb logDistortionProb(PositionIndex i, PositionIndex slen, PositionIndex tlen, PositionIndex j);

  Prob fertilityProb(WordIndex s, PositionIndex phi);
  LgProb logFertilityProb(WordIndex s, PositionIndex phi);

  LgProb obtainBestAlignment(const std::vector<WordIndex>& src, const std::vector<WordIndex>& trg,
                             WordAligMatrix& bestWaMatrix);
  LgProb calcLgProbForAlig(const std::vector<WordIndex>& src, const std::vector<WordIndex>& trg,
                           const WordAligMatrix& aligMatrix, int verbose = 0);
  LgProb calcLgProb(const std::vector<WordIndex>& src, const std::vector<WordIndex>& trg, int verbose = 0);

  // load function
  bool load(const char* prefFileName, int verbose = 0);

  // print function
  bool print(const char* prefFileName, int verbose = 0);

  // clear() function
  void clear();

  void clearTempVars();

  void clearInfoAboutSentRange();

  // Destructor
  ~Ibm3AligModel();

protected:
  typedef std::vector<double> DistortionCountsElem;
  typedef OrderedVector<dSource, DistortionCountsElem> DistortionCounts;
  typedef std::vector<double> FertilityCountsElem;
  typedef std::vector<FertilityCountsElem> FertilityCounts;

  const PositionIndex MaxFertility = 10;

  double unsmoothedDistortionProb(PositionIndex i, PositionIndex slen, PositionIndex tlen, PositionIndex j);
  double unsmoothedLogDistortionProb(PositionIndex i, PositionIndex slen, PositionIndex tlen, PositionIndex j);
  double distortionProb(PositionIndex i, PositionIndex slen, PositionIndex tlen, PositionIndex j, bool training);

  double unsmoothedFertilityProb(WordIndex s, PositionIndex phi);
  double unsmoothedLogFertilityProb(WordIndex s, PositionIndex phi);
  double fertilityProb(WordIndex s, PositionIndex phi, bool training);

  Prob searchForBestAlignment(const std::vector<WordIndex>& nsrc, const std::vector<WordIndex>& trg, bool training,
                              AlignmentInfo& bestAlignment, Matrix<double>* moveScores = nullptr,
                              Matrix<double>* swapScores = nullptr);
  void getInitialAlignmentForSearch(const std::vector<WordIndex>& nsrc, const std::vector<WordIndex>& trg,
                                    bool training, AlignmentInfo& alignment);
  Prob calcProbOfAlignment(const std::vector<WordIndex>& nsrc, const std::vector<WordIndex>& trg, bool training,
                           AlignmentInfo& alignment, int verbose = 0);
  double swapScore(const Sentence& nsrc, const Sentence& trg, PositionIndex j1, PositionIndex j2, bool training,
                   AlignmentInfo& alignment);
  double moveScore(const Sentence& nsrc, const Sentence& trg, PositionIndex iNew, PositionIndex j, bool training,
                   AlignmentInfo& alignment);

  // batch EM functions
  void initSourceWord(const Sentence& nsrc, const Sentence& trg, PositionIndex i);
  void addTranslationOptions(std::vector<std::vector<WordIndex>>& insertBuffer);
  void batchUpdateCounts(const SentPairCont& pairs);
  void incrementWordPairCounts(const Sentence& nsrc, const Sentence& trg, PositionIndex i, PositionIndex j,
                               double count);
  virtual void incrementTargetWordCounts(const Sentence& nsrc, const Sentence& trg, const AlignmentInfo& alignment,
                                         PositionIndex j, double count);
  void batchMaximizeProbs();

  // model parameters
  Prob p1;
  IncrDistortionTable distortionTable;
  IncrFertilityTable fertilityTable;

  // EM counts
  DistortionCounts distortionCounts;
  FertilityCounts fertilityCounts;
  double p0Count;
  double p1Count;
};

#endif