#pragma once

#include "nlp_common/Count.h"
#include "sw_models/AlignmentModel.h"
#include "sw_models/SymmetrizedAligner.h"

#include <memory>

// Both models must be trained on the same pairs in the same order (the inverse
// with source and target swapped) so that index n lines up across the two.
class SymmetrizedAlignmentModel : public SymmetrizedAligner
{
public:
  SymmetrizedAlignmentModel(std::shared_ptr<AlignmentModel> directModel,
                            std::shared_ptr<AlignmentModel> inverseModel);

  unsigned int numSentencePairs();

  // Returns the source/target sentences (and count) for training pair n, in
  // src->trg order (delegates to the direct model).
  int getSentencePair(unsigned int n, std::vector<std::string>& srcSentStr, std::vector<std::string>& trgSentStr,
                      Count& c);

  // bound for getTrainingAlignment: the shorter of the two directions'
  size_t numTrainingAlignments();

  // The symmetrized alignment for training pair n, mirroring getBestAlignment:
  // combines the direct and inverse models' training alignments under the current
  // heuristic. Fills 'alignment' (per target token, 1-based source position,
  // 0 = NULL) and returns the direction-max log-probability.
  LgProb getTrainingAlignment(size_t n, std::vector<PositionIndex>& alignment);
  // As above, but fills a WordAlignmentMatrix instead of a per-target vector.
  LgProb getTrainingAlignment(size_t n, WordAlignmentMatrix& bestWaMatrix);

  virtual ~SymmetrizedAlignmentModel()
  {
  }

private:
  std::shared_ptr<AlignmentModel> directModel;
  std::shared_ptr<AlignmentModel> inverseModel;
};
