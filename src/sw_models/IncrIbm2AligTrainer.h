#ifndef _IncrIbm2AligTrainer_h
#define _IncrIbm2AligTrainer_h

#include "sw_models/Ibm2AligModel.h"
#include "sw_models/IncrIbm1AligTrainer.h"

class IncrIbm2AligTrainer : public IncrIbm1AligTrainer
{
public:
  // Constructor
  IncrIbm2AligTrainer(Ibm2AligModel& model, anjiMatrix& anji);

  void clear();

  // Destructor
  ~IncrIbm2AligTrainer();

protected:
  void incrUpdateCounts(unsigned int mapped_n, unsigned int mapped_n_aux, PositionIndex i, PositionIndex j,
                        const std::vector<WordIndex>& nsrcSent, const std::vector<WordIndex>& trgSent,
                        const Count& weight);
  void incrUpdateCountsAlig(unsigned int mapped_n, unsigned int mapped_n_aux, PositionIndex i, PositionIndex j,
                            PositionIndex slen, PositionIndex tlen, const Count& weight);
  void incrMaximizeProbs();
  void incrMaximizeProbsAlig();

private:
  typedef std::vector<std::pair<float, float>> IncrAlignmentCountsElem;
  typedef OrderedVector<AlignmentKey, IncrAlignmentCountsElem> IncrAlignmentCounts;

  Ibm2AligModel& model;
  IncrAlignmentCounts incrAlignmentCounts;
};

#endif
