#include "sw_models/IncrIbm2AligModel.h"

#include "nlp_common/ErrorDefs.h"

using namespace std;

IncrIbm2AligModel::IncrIbm2AligModel() : trainer(*this, anji)
{
}

void IncrIbm2AligModel::set_expval_maxnsize(unsigned int _anji_maxnsize)
{
  anji.set_maxnsize(_anji_maxnsize);
}

void IncrIbm2AligModel::startIncrTraining(std::pair<unsigned int, unsigned int> sentPairRange, int verbosity)
{
  clearTempVars();
  // Train sentence length model
  sentLengthModel->trainSentPairRange(sentPairRange, verbosity);
}

void IncrIbm2AligModel::incrTrain(pair<unsigned int, unsigned int> sentPairRange, int verbosity)
{
  trainer.incrTrain(sentPairRange, verbosity);
}

void IncrIbm2AligModel::endIncrTraining()
{
  clearTempVars();
}

bool IncrIbm2AligModel::load(const char* prefFileName, int verbose)
{
  bool retVal = Ibm2AligModel::load(prefFileName, verbose);
  if (retVal == THOT_ERROR)
    return retVal;

  // Load file with anji values
  return anji.load(prefFileName, verbose);
}

bool IncrIbm2AligModel::print(const char* prefFileName, int verbose)
{
  bool retVal = Ibm2AligModel::print(prefFileName, verbose);
  if (retVal == THOT_ERROR)
    return retVal;

  // Print file anji values
  return anji.print(prefFileName);
}

void IncrIbm2AligModel::clear()
{
  Ibm2AligModel::clear();
  anji.clear();
  trainer.clear();
}

void IncrIbm2AligModel::clearTempVars()
{
  Ibm2AligModel::clearTempVars();
  trainer.clear();
}
