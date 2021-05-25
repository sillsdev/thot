#ifndef _Ibm3AligModel_h
#define _Ibm3AligModel_h

#include "IncrIbm2AligModel.h"
#include "IncrDistortionTable.h"
#include "IncrFertilityTable.h"

class Ibm3AligModel : public IncrIbm2AligModel
{
public:
  // Constructor
  Ibm3AligModel();

  Prob distortionProb(PositionIndex i, PositionIndex slen, PositionIndex tlen, PositionIndex j);
  LgProb logDistortionProb(PositionIndex i, PositionIndex slen, PositionIndex tlen, PositionIndex j);

  Prob fertilityProb(WordIndex s, PositionIndex phi);
  LgProb logFertilityProb(WordIndex s, PositionIndex phi);

  // Functions to generate alignments 
  LgProb obtainBestAlignment(const std::vector<WordIndex>& srcSentIndexVector,
    const std::vector<WordIndex>& trgSentIndexVector, WordAligMatrix& bestWaMatrix);
  LgProb lexAligM3LpForBestAlig(const std::vector<WordIndex>& nSrcSentIndexVector,
    const std::vector<WordIndex>& trgSentIndexVector, std::vector<PositionIndex>& bestAlig,
    std::vector<PositionIndex>& bestFertility,
    std::vector<std::pair<std::vector<double>, std::vector<double>>>& neighborProbs, double& totalNeighborProb);

  // Functions to calculate probabilities for alignments
  LgProb calcLgProbForAlig(const std::vector<WordIndex>& sSent, const std::vector<WordIndex>& tSent,
    const WordAligMatrix& aligMatrix, int verbose = 0);
  LgProb calcIbm3LgProbFromAlig(const std::vector<WordIndex>& nsSent, const std::vector<WordIndex>& tSent,
    const std::vector<PositionIndex>& alig, const std::vector<PositionIndex>& fertility, int verbose = 0);

  // Scoring functions without giving an alignment
  LgProb calcLgProb(const std::vector<WordIndex>& sSent, const std::vector<WordIndex>& tSent, int verbose = 0);
  LgProb calcSumIbm3LgProb(const std::vector<WordIndex>& nsSent, const std::vector<WordIndex>& tSent, int verbose = 0);

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
  const PositionIndex MaxFertility = 10;

  // model parameters
  Prob p1;
  IncrDistortionTable distortionTable;
  IncrFertilityTable fertilityTable;

  typedef std::vector<double> DistortionCountsEntry;
  typedef OrderedVector<dSource, DistortionCountsEntry> DistortionCounts;
  typedef std::vector<double> FertilityCountsEntry;
  typedef std::vector<FertilityCountsEntry> FertilityCounts;

  // EM counts
  DistortionCounts distortionCounts;
  FertilityCounts fertilityCounts;
  double p0Count;
  double p1Count;

  double unsmoothedDistortionProb(PositionIndex i, PositionIndex slen, PositionIndex tlen, PositionIndex j);
  double unsmoothedLogDistortionProb(PositionIndex i, PositionIndex slen, PositionIndex tlen, PositionIndex j);

  double unsmoothedFertilityProb(WordIndex s, PositionIndex phi);
  double unsmoothedLogFertilityProb(WordIndex s, PositionIndex phi);

  // batch EM functions
  void initialBatchPass(std::pair<unsigned int, unsigned int> sentPairRange, int verbose);
  void initSourceWord(const Sentence& nsrc, const Sentence& trg, PositionIndex i);
  void initWordPair(const Sentence& nsrc, const Sentence& trg, PositionIndex i, PositionIndex j);
  void addTranslationOptions(std::vector<std::vector<WordIndex>>& insertBuffer);
  void batchUpdateCounts(const SentPairCont& pairs);
  void incrementCounts(const Sentence& nsrc, const Sentence& trg, const std::vector<PositionIndex>& alig,
    const std::vector<PositionIndex>& fertility, double count);
  void incrementWordPairCounts(const Sentence& nsrc, const Sentence& trg, PositionIndex i, PositionIndex j,
    double count);
  void batchMaximizeProbs();
};

#endif