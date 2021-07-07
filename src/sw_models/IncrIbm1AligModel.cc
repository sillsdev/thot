#include "IncrIbm1AligModel.h"

#include "nlp_common/ErrorDefs.h"
#include "sw_models/IncrIbm1AligModel.h"

using namespace std;

IncrIbm1AligModel::IncrIbm1AligModel() : trainer(*this, anji)
{
}

void IncrIbm1AligModel::set_expval_maxnsize(unsigned int _anji_maxnsize)
{
  anji.set_maxnsize(_anji_maxnsize);
}

void IncrIbm1AligModel::startIncrTraining(std::pair<unsigned int, unsigned int> sentPairRange, int verbosity)
{
  clearTempVars();
  // Train sentence length model
  sentLengthModel->trainSentPairRange(sentPairRange, verbosity);
}

void IncrIbm1AligModel::incrTrain(pair<unsigned int, unsigned int> sentPairRange, int verbosity)
{
  trainer.incrTrain(sentPairRange, verbosity);
}

void IncrIbm1AligModel::endIncrTraining()
{
  clearTempVars();
}

bool IncrIbm1AligModel::load(const char* prefFileName, int verbose)
{
  bool retVal = Ibm1AligModel::load(prefFileName, verbose);
  if (retVal == THOT_ERROR)
    return retVal;

  // Load file with anji values
  return anji.load(prefFileName, verbose);
}

bool IncrIbm1AligModel::print(const char* prefFileName, int verbose)
{
  bool retVal = Ibm1AligModel::print(prefFileName, verbose);
  if (retVal == THOT_ERROR)
    return retVal;

  // Print file anji values
  return anji.print(prefFileName);
}

void IncrIbm1AligModel::clear()
{
  Ibm1AligModel::clear();
  anji.clear();
  trainer.clear();
}

void IncrIbm1AligModel::clearTempVars()
{
  Ibm1AligModel::clearTempVars();
  trainer.clear();
}
