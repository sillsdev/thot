#include "sw_models/IncrHmmAligModel.h"

#include "nlp_common/ErrorDefs.h"

IncrHmmAligModel::IncrHmmAligModel() : trainer(*this, lanji, lanjm1ip_anji)
{
}

void IncrHmmAligModel::set_expval_maxnsize(unsigned int _expval_maxnsize)
{
  lanji.set_maxnsize(_expval_maxnsize);
  lanjm1ip_anji.set_maxnsize(_expval_maxnsize);
}

void IncrHmmAligModel::startIncrTraining(std::pair<unsigned int, unsigned int> sentPairRange, int verbosity)
{
  clearTempVars();
  // Train sentence length model
  sentLengthModel->trainSentPairRange(sentPairRange, verbosity);
}

void IncrHmmAligModel::incrTrain(std::pair<unsigned int, unsigned int> sentPairRange, int verbosity)
{
  trainer.incrTrain(sentPairRange, verbosity);
}

void IncrHmmAligModel::endIncrTraining()
{
  clearTempVars();
}

bool IncrHmmAligModel::load(const char* prefFileName, int verbose)
{
  bool retVal = HmmAligModel::load(prefFileName, verbose);
  if (retVal == THOT_ERROR)
    return retVal;

  // Load file with lanji values
  retVal = lanji.load(prefFileName, verbose);
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  // Load file with lanjm1ip_anji values
  retVal = lanjm1ip_anji.load(prefFileName, verbose);

  return retVal;
}

bool IncrHmmAligModel::print(const char* prefFileName, int verbose)
{
  bool retVal = HmmAligModel::print(prefFileName, verbose);
  if (retVal == THOT_ERROR)
    return retVal;

  // Print file lanji values
  retVal = lanji.print(prefFileName);
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  // Print file with lanjm1ip_anji values
  retVal = lanjm1ip_anji.print(prefFileName);

  return retVal;
}

void IncrHmmAligModel::clear()
{
  HmmAligModel::clear();
  lanji.clear();
  lanjm1ip_anji.clear();
}

void IncrHmmAligModel::clearTempVars()
{
  HmmAligModel::clearTempVars();
  trainer.clear();
}
