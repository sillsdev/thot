#ifndef _IncrIbm1AligTrainer_h
#define _IncrIbm1AligTrainer_h

#if HAVE_CONFIG_H
#  include <thot_config.h>
#endif /* HAVE_CONFIG_H */

#include "Ibm1AligModel.h"
#include "anjiMatrix.h"
#include "LexCounts.h"

class IncrIbm1AligTrainer
{
public:
  // Constructor
  IncrIbm1AligTrainer(Ibm1AligModel& model, anjiMatrix& anji);

  void incrTrainSentPairRange(std::pair<unsigned int, unsigned int> sentPairRange, int verbosity);
  void clear();

  // Destructor
  ~IncrIbm1AligTrainer();

protected:
  const float SmoothingWeightedAnji = 1e-6f;

  // Incremental EM functions
  void calcNewLocalSuffStats(std::pair<unsigned int, unsigned int> sentPairRange, int verbosity = 0);
  void calc_anji(unsigned int n, const std::vector<WordIndex>& nsrcSent, const std::vector<WordIndex>& trgSent,
    const Count& weight);
  virtual void incrUpdateCounts(unsigned int mapped_n, unsigned int mapped_n_aux, PositionIndex i, PositionIndex j,
    const std::vector<WordIndex>& nsrcSent, const std::vector<WordIndex>& trgSent, const Count& weight);
  virtual void incrMaximizeProbs();
  virtual float obtainLogNewSuffStat(float lcurrSuffStat, float lLocalSuffStatCurr, float lLocalSuffStatNew);

  // Data structures for manipulating expected values
  anjiMatrix& anji;
  anjiMatrix anji_aux;

private:
  Ibm1AligModel& model;
  IncrLexCounts incrLexCounts;
};

#endif
