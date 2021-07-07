#pragma once

#include "nlp_common/PositionIndex.h"

#include <iostream>

class aSourceHmm
{
public:
  PositionIndex prev_i;
  PositionIndex slen;

  bool operator==(const aSourceHmm& right) const
  {
    if (right.prev_i != prev_i)
      return 0;
    if (right.slen != slen)
      return 0;
    return 1;
  }

  bool operator<(const aSourceHmm& right) const
  {
    if (right.prev_i < prev_i)
      return 0;
    if (prev_i < right.prev_i)
      return 1;
    if (right.slen < slen)
      return 0;
    if (slen < right.slen)
      return 1;
    return 0;
  }
};

std::ostream& operator<<(std::ostream& outS, const aSourceHmm& aSrcHmm);
