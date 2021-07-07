#pragma once

#include "nlp_common/PositionIndex.h"

#include <vector>

#define CACHED_HMM_ALIG_LGPROB_VIT_INVALID_VAL 99

class CachedHmmAligLgProb
{
public:
  void makeRoomGivenSrcSentLen(PositionIndex slen);
  bool isDefined(PositionIndex prev_i, PositionIndex slen, PositionIndex i);
  void set_boundary_check(PositionIndex prev_i, PositionIndex slen, PositionIndex i, double lp);
  void set(PositionIndex prev_i, PositionIndex slen, PositionIndex i, double lp);
  double get(PositionIndex prev_i, PositionIndex slen, PositionIndex i);
  void clear();

private:
  std::vector<std::vector<std::vector<double>>> cachedLgProbs;
};
