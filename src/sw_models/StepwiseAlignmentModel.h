#pragma once

#include "sw_models/AlignmentModel.h"

/*
 * Interface for alignment models that use stepwise EM.
 */
class StepwiseAlignmentModel : public virtual AlignmentModel
{
public:
  StepwiseAlignmentModel(){};

  // Function to set the value of alpha
  virtual void set_nu_val(float _nu) = 0;

  virtual ~StepwiseAlignmentModel(){};
};
