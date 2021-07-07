#pragma once

#include "sw_models/_swAligModel.h"

class _incrSwAligModel : public virtual BaseSwAligModel
{
public:
  // Function to set a maximum size for the vector of expected
  // values anji (by default the size is not restricted)
  virtual void set_expval_maxnsize(unsigned int _anji_maxnsize) = 0;

  virtual void startIncrTraining(std::pair<unsigned int, unsigned int> sentPairRange, int verbosity = 0) = 0;

  virtual void incrTrain(std::pair<unsigned int, unsigned int> sentPairRange, int verbosity = 0) = 0;

  virtual void endIncrTraining() = 0;

  virtual ~_incrSwAligModel()
  {
  }
};
