#pragma once

#include "sw_models/HmmAlignmentModel.h"
#include "sw_models/IncrAlignmentModel.h"
#include "sw_models/IncrHmmAlignmentTrainer.h"
#include "sw_models/anjiMatrix.h"

class IncrHmmAlignmentModel : public HmmAlignmentModel, public virtual IncrAlignmentModel
{
public:
  IncrHmmAlignmentModel();

  AlignmentModelType getModelType() const override
  {
    return IncrHmm;
  }

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

  virtual ~IncrHmmAlignmentModel()
  {
  }

protected:
  anjiMatrix lanji;
  anjm1ip_anjiMatrix lanjm1ip_anji;
  IncrHmmAlignmentTrainer trainer;
};
