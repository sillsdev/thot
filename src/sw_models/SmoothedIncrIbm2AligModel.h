/*
thot package for statistical machine translation
Copyright (C) 2013 Daniel Ortiz-Mart\'inez

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public License
as published by the Free Software Foundation; either version 3
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program; If not, see <http://www.gnu.org/licenses/>.
*/

/**
 * @file SmoothedIncrIbm2AligModel.h
 *
 * @brief Defines the SmoothedIncrIbm2AligModel class.
 * SmoothedIncrIbm2AligModel class inherits from the IncrIbm2AligModel
 * class and incorporates a simple smoothing technique.
 *
 */

#pragma once

#include "sw_models/IncrIbm2AligModel.h"

#define IBM2_PROB_THRESHOLD 1e-8

class SmoothedIncrIbm2AligModel : public IncrIbm2AligModel
{
public:
  // lexical model functions
  Prob pts(WordIndex s, WordIndex t);
  // returns p(t|s)

  // alignment model functions
  Prob aProb(PositionIndex j, PositionIndex slen, PositionIndex tlen, PositionIndex i);
};

