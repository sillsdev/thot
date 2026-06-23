#pragma once

#include "nlp_common/ErrorDefs.h"
#include "nlp_common/PositionIndex.h"

#include <vector>

// Jump (distortion) distribution for the Eflomal model. Unlike the HMM
// alignment table, which is keyed by (prev_i, slen), the jump is a single
// distribution over the signed offset delta = i - prev_i, bucketed into a
// symmetric window [-window, window] and clamped at the edges. This keeps the
// table tiny and source-length independent, matching the eflomal jump model.
class EflomalJumpTable
{
public:
  // Builds the table from raw jump counts. counts is indexed by
  // bucket = clamp(offset, -window, window) + window and must have size
  // 2 * window + 1; sum is the total of those counts. A Dirichlet prior alpha
  // smooths unseen offsets.
  void setFromCounts(const std::vector<double>& counts, double sum, double alpha, int window);

  // Returns log(p(offset)). Offsets beyond the window are clamped to the edge
  // bucket. Returns a small smoothing value when the table is empty.
  double logProb(int offset) const;
  double prob(int offset) const;

  int getWindow() const
  {
    return window;
  }

  bool load(const char* jumpFile, int verbose = 0);
  bool print(const char* jumpFile) const;

  void clear();

private:
  int window = 0;
  std::vector<double> logProbs;
};
