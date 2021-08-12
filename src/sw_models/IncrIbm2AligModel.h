#pragma once

#include "sw_models/Ibm2AligModel.h"
#include "sw_models/IncrIbm2AligTrainer.h"
#include "sw_models/_incrSwAligModel.h"
#include "sw_models/anjiMatrix.h"

class IncrIbm2AligModel : public Ibm2AligModel, public virtual _incrSwAligModel
{
public:
  // Constructor
  IncrIbm2AligModel();

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

  virtual ~IncrIbm2AligModel()
  {
  }

protected:
  anjiMatrix anji;
  IncrIbm2AligTrainer trainer;
};
