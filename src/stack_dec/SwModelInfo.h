#pragma once

#include "nlp_common/Prob.h"
#include "stack_dec/SwModelPars.h"
#include "sw_models/AlignmentModel.h"

#include <memory>
#include <vector>

#define DEFAULT_LVALUE_CONF_INTERV 0.01f
#define DEFAULT_RVALUE_CONF_INTERV 0.99f
#define DEFAULT_MAX_INTERV_SIZE 20
#define DEFAULT_LAMBDA_VALUE 0.9f

struct SwModelInfo
{
  // sw model members
  std::vector<std::shared_ptr<AlignmentModel>> swAligModels;
  SwModelPars swModelPars;

  // Inverse sw model members
  std::vector<std::shared_ptr<AlignmentModel>> invSwAligModels;
  SwModelPars invSwModelPars;

  // Confidence interval for length model
  std::pair<float, float> lenModelConfInterv;

  // Maximum interval size for length range
  unsigned int maxIntervalSize;

  // Linear interpolation weights
  float lambda_swm;
  float lambda_invswm;

  SwModelInfo()
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
