#pragma once

#include "nlp_common/WordAlignmentMatrix.h"

struct AligInfo
{
  std::vector<unsigned int> s;
  WordAlignmentMatrix wordAligMatrix;
  unsigned int count_s_t_;
};
