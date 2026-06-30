#pragma once

#include "nlp_common/Count.h"
#include "sw_models/AlignmentModel.h"
#include "sw_models/SymmetrizedAligner.h"

#include <memory>

// A symmetrized aligner that also supports transductive alignment, i.e.
// getTrainingAlignment. It combines the per-training-pair alignments inferred by
// a direct model (trained on src->trg) and an inverse model (trained on the same
// pairs with trg/src swapped) using a symmetrization heuristic.
//
// Both models must be trained on the same sentence pairs in the same order (the
// inverse with source and target swapped) and with setEmitTrainingAlignments(true),
// so that index n lines up across the two corpora.
//
// Inherits the symmetrized getBestAlignment overloads and the 'heuristic'
// property from SymmetrizedAligner.
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
