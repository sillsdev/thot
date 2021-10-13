#include "sw_models/IncrIbm2AlignmentModel.h"

#include "nlp_common/ErrorDefs.h"

using namespace std;

IncrIbm2AlignmentModel::IncrIbm2AlignmentModel() : trainer(*this, anji)
{
}

void IncrIbm2AlignmentModel::set_expval_maxnsize(unsigned int _anji_maxnsize)
{
  anji.set_maxnsize(_anji_maxnsize);
}

void IncrIbm2AlignmentModel::startIncrTraining(std::pair<unsigned int, unsigned int> sentPairRange, int verbosity)
{
  clearTempVars();
  // Train sentence length model
  sentLengthModel->trainSentencePairRange(sentPairRange, verbosity);
}

void IncrIbm2AlignmentModel::incrTrain(pair<unsigned int, unsigned int> sentPairRange, int verbosity)
{
  trainer.incrTrain(sentPairRange, verbosity);
}

void IncrIbm2AlignmentModel::endIncrTraining()
{
  clearTempVars();
}

bool IncrIbm2AlignmentModel::load(const char* prefFileName, int verbose)
{
  bool retVal = Ibm2AlignmentModel::load(prefFileName, verbose);
  if (retVal == THOT_ERROR)
    return retVal;

  // Load file with anji values
  anji.load(prefFileName, verbose);

  return THOT_OK;
}

bool IncrIbm2AlignmentModel::print(const char* prefFileName, int verbose)
{
  bool retVal = Ibm2AlignmentModel::print(prefFileName, verbose);
  if (retVal == THOT_ERROR)
    return retVal;

  // Print file anji values
  return anji.print(prefFileName);
}

void IncrIbm2AlignmentModel::clear()
{
  Ibm2AlignmentModel::clear();
  anji.clear();
  trainer.clear();
}

void IncrIbm2AlignmentModel::clearTempVars()
{
  Ibm2AlignmentModel::clearTempVars();
  trainer.clear();
}
