#ifndef _IncrIbm2AligTrainer_h
#define _IncrIbm2AligTrainer_h

#if HAVE_CONFIG_H
#  include <thot_config.h>
#endif /* HAVE_CONFIG_H */

#include "IncrIbm1AligTrainer.h"
#include "Ibm2AligModel.h"

class IncrIbm2AligTrainer : public IncrIbm1AligTrainer
{
public:
  // Constructor
  IncrIbm2AligTrainer(Ibm2AligModel& model, anjiMatrix& anji);

  void clear();

  // Destructor
  ~IncrIbm2AligTrainer();

protected:
  double calc_anji_num(const std::vector<WordIndex>& nsrcSent, const std::vector<WordIndex>& trgSent, unsigned int i,
    unsigned int j);
  double calc_anji_num_alig(PositionIndex i, PositionIndex j, PositionIndex slen, PositionIndex tlen);
  void incrUpdateCounts(unsigned int mapped_n, unsigned int mapped_n_aux, PositionIndex i, PositionIndex j,
    const std::vector<WordIndex>& nsrcSent, const std::vector<WordIndex>& trgSent, const Count& weight);
  void incrUpdateCountsAlig(unsigned int mapped_n, unsigned int mapped_n_aux, PositionIndex i, PositionIndex j,
    PositionIndex slen, PositionIndex tlen, const Count& weight);
  void incrMaximizeProbs();
  void incrMaximizeProbsAlig();

private:
  typedef std::vector<std::pair<float, float>> IncrAligCountsEntry;
  typedef OrderedVector<aSource, IncrAligCountsEntry> IncrAligCounts;

  const double ArbitraryAp = 0.1;

  Ibm2AligModel& model;
  IncrAligCounts incrAligCounts;
};

#endif
