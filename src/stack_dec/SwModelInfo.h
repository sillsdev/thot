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

#pragma once

//--------------- Include files --------------------------------------

#include "nlp_common/Prob.h"
#include "stack_dec/SwModelPars.h"
#include "sw_models/BaseSwAligModel.h"

#include <vector>

//--------------- Constants ------------------------------------------

#define DEFAULT_LVALUE_CONF_INTERV 0.01
#define DEFAULT_RVALUE_CONF_INTERV 0.99
#define DEFAULT_MAX_INTERV_SIZE 20
#define DEFAULT_LAMBDA_VALUE 0.9

//--------------- typedefs -------------------------------------------

//--------------- SwModelInfo struct

struct SwModelInfo
{
  // sw model members
  std::vector<BaseSwAligModel*> swAligModelPtrVec;
  SwModelPars swModelPars;

  // Inverse sw model members
  std::vector<BaseSwAligModel*> invSwAligModelPtrVec;
  SwModelPars invSwModelPars;

  // Confidence interval for length model
  std::pair<float, float> lenModelConfInterv;

  // Maximum interval size for length range
  unsigned int maxIntervalSize;

  // Linear interpolation weights
  float lambda_swm;
  float lambda_invswm;

  SwModelInfo(void)
  {
    // Initialize variables related to the generation of length ranges
    lenModelConfInterv.first = DEFAULT_LVALUE_CONF_INTERV;
    lenModelConfInterv.second = DEFAULT_RVALUE_CONF_INTERV;
    maxIntervalSize = DEFAULT_MAX_INTERV_SIZE;
    // Set default linear interpolation weights
    lambda_swm = DEFAULT_LAMBDA_VALUE;
    lambda_invswm = DEFAULT_LAMBDA_VALUE;
  };
};

