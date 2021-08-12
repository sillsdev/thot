#pragma once

#include "sw_models/Ibm1AligModel.h"
#include "sw_models/IncrIbm1AligTrainer.h"
#include "sw_models/_incrSwAligModel.h"
#include "sw_models/anjiMatrix.h"

class IncrIbm1AligModel : public Ibm1AligModel, public virtual _incrSwAligModel
{
public:
  IncrIbm1AligModel();

  // Function to set a maximum size for the vector of expected
  // values anji (by default the size is not restricted)
  void set_expval_maxnsize(unsigned int _anji_maxnsize) override;

  void startIncrTraining(std::pair<unsigned int, unsigned int> sentPairRange, int verbosity = 0) override;
  void incrTrain(std::pair<unsigned int, unsigned int> sentPairRange, int verbosity = 0) override;
  void endIncrTraining() override;

  bool load(const char* prefFileName, int verbose = 0) override;
  bool print(const char* prefFileName, int verbose = 0) override;

  void clear() override;
  void clearTempVars() override;

  virtual ~IncrIbm1AligModel()
  {
  }

protected:
  anjiMatrix anji;
  IncrIbm1AligTrainer trainer;
};
