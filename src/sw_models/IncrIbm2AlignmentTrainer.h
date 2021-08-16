#pragma once

#include "sw_models/Ibm2AlignmentModel.h"
#include "sw_models/IncrIbm1AlignmentTrainer.h"

class IncrIbm2AlignmentTrainer : public IncrIbm1AlignmentTrainer
{
public:
  IncrIbm2AlignmentTrainer(Ibm2AlignmentModel& model, anjiMatrix& anji);

  void clear() override;

  virtual ~IncrIbm2AlignmentTrainer()
  {
  }

protected:
  void incrUpdateCounts(unsigned int mapped_n, unsigned int mapped_n_aux, PositionIndex i, PositionIndex j,
                        const std::vector<WordIndex>& nsrcSent, const std::vector<WordIndex>& trgSent,
                        const Count& weight) override;
  void incrUpdateCountsAlig(unsigned int mapped_n, unsigned int mapped_n_aux, PositionIndex i, PositionIndex j,
                            PositionIndex slen, PositionIndex tlen, const Count& weight);
  void incrMaximizeProbs() override;
  void incrMaximizeProbsAlig();

private:
  typedef std::vector<std::pair<float, float>> IncrAlignmentCountsElem;
  typedef OrderedVector<AlignmentKey, IncrAlignmentCountsElem> IncrAlignmentCounts;

  Ibm2AlignmentModel& model;
  IncrAlignmentCounts incrAlignmentCounts;
};
