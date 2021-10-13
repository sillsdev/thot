#include "IncrIbm1AlignmentModel.h"

#include "nlp_common/ErrorDefs.h"
#include "sw_models/IncrIbm1AlignmentModel.h"

using namespace std;

IncrIbm1AlignmentModel::IncrIbm1AlignmentModel() : trainer(*this, anji)
{
}

void IncrIbm1AlignmentModel::set_expval_maxnsize(unsigned int _anji_maxnsize)
{
  anji.set_maxnsize(_anji_maxnsize);
}

void IncrIbm1AlignmentModel::startIncrTraining(std::pair<unsigned int, unsigned int> sentPairRange, int verbosity)
{
  clearTempVars();
  // Train sentence length model
  sentLengthModel->trainSentencePairRange(sentPairRange, verbosity);
}

void IncrIbm1AlignmentModel::incrTrain(pair<unsigned int, unsigned int> sentPairRange, int verbosity)
{
  trainer.incrTrain(sentPairRange, verbosity);
}

void IncrIbm1AlignmentModel::endIncrTraining()
{
  clearTempVars();
}

bool IncrIbm1AlignmentModel::load(const char* prefFileName, int verbose)
{
  bool retVal = Ibm1AlignmentModel::load(prefFileName, verbose);
  if (retVal == THOT_ERROR)
    return retVal;

  // Load file with anji values
  anji.load(prefFileName, verbose);

  return THOT_OK;
}

bool IncrIbm1AlignmentModel::print(const char* prefFileName, int verbose)
{
  bool retVal = Ibm1AlignmentModel::print(prefFileName, verbose);
  if (retVal == THOT_ERROR)
    return retVal;

  // Print file anji values
  return anji.print(prefFileName);
}

void IncrIbm1AlignmentModel::clear()
{
  Ibm1AlignmentModel::clear();
  anji.clear();
}

void IncrIbm1AlignmentModel::clearTempVars()
{
  Ibm1AlignmentModel::clearTempVars();
  trainer.clear();
}
