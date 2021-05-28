#include "IncrIbm1AligModel.h"

using namespace std;

IncrIbm1AligModel::IncrIbm1AligModel() : trainer(*this, anji)
{
}

void IncrIbm1AligModel::set_expval_maxnsize(unsigned int _anji_maxnsize)
{
  anji.set_maxnsize(_anji_maxnsize);
}

void IncrIbm1AligModel::incrTrainSentPairRange(pair<unsigned int, unsigned int> sentPairRange, int verbosity)
{
  // Train sentence length model
  sentLengthModel.trainSentPairRange(sentPairRange, verbosity);

  trainer.incrTrainSentPairRange(sentPairRange, verbosity);
}

void IncrIbm1AligModel::incrTrainAllSents(int verbosity)
{
  if (numSentPairs() > 0)
    incrTrainSentPairRange(make_pair(0, numSentPairs() - 1), verbosity);
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

void IncrIbm1AligModel::clearInfoAboutSentRange()
{
  // Clear info about sentence range
  Ibm1AligModel::clearInfoAboutSentRange();
  anji.clear();
  trainer.clear();
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

IncrIbm1AligModel::~IncrIbm1AligModel()
{
}
