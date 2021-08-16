#pragma once

#include "sw_models/Ibm1AlignmentModel.h"
#include "sw_models/LexCounts.h"
#include "sw_models/anjiMatrix.h"

class IncrIbm1AlignmentTrainer
{
public:
  IncrIbm1AlignmentTrainer(Ibm1AlignmentModel& model, anjiMatrix& anji);

  void incrTrain(std::pair<unsigned int, unsigned int> sentPairRange, int verbosity);
  virtual void clear();

  virtual ~IncrIbm1AlignmentTrainer()
  {
  }

protected:
  // Incremental EM functions
  void calcNewLocalSuffStats(std::pair<unsigned int, unsigned int> sentPairRange, int verbosity = 0);
  void calc_anji(unsigned int n, const std::vector<WordIndex>& nsrcSent, const std::vector<WordIndex>& trgSent,
                 const Count& weight);
  virtual void incrUpdateCounts(unsigned int mapped_n, unsigned int mapped_n_aux, PositionIndex i, PositionIndex j,
                                const std::vector<WordIndex>& nsrcSent, const std::vector<WordIndex>& trgSent,
                                const Count& weight);
  virtual void incrMaximizeProbs();
  float obtainLogNewSuffStat(float lcurrSuffStat, float lLocalSuffStatCurr, float lLocalSuffStatNew);

  // Data structures for manipulating expected values
  anjiMatrix& anji;
  anjiMatrix anji_aux;

private:
  Ibm1AlignmentModel& model;
  IncrLexCounts incrLexCounts;
};
