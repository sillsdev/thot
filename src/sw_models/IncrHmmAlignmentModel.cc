#include "sw_models/IncrHmmAlignmentModel.h"

#include "nlp_common/ErrorDefs.h"

IncrHmmAlignmentModel::IncrHmmAlignmentModel() : trainer(*this, lanji, lanjm1ip_anji)
{
}

void IncrHmmAlignmentModel::set_expval_maxnsize(unsigned int _expval_maxnsize)
{
  lanji.set_maxnsize(_expval_maxnsize);
  lanjm1ip_anji.set_maxnsize(_expval_maxnsize);
}

void IncrHmmAlignmentModel::startIncrTraining(std::pair<unsigned int, unsigned int> sentPairRange, int verbosity)
{
  clearTempVars();
  // Train sentence length model
  sentLengthModel->trainSentencePairRange(sentPairRange, verbosity);
}

void IncrHmmAlignmentModel::incrTrain(std::pair<unsigned int, unsigned int> sentPairRange, int verbosity)
{
  trainer.incrTrain(sentPairRange, verbosity);
}

void IncrHmmAlignmentModel::endIncrTraining()
{
  clearTempVars();
}

bool IncrHmmAlignmentModel::load(const char* prefFileName, int verbose)
{
  bool retVal = HmmAlignmentModel::load(prefFileName, verbose);
  if (retVal == THOT_ERROR)
    return retVal;

  // Load file with lanji values
  lanji.load(prefFileName, verbose);

  // Load file with lanjm1ip_anji values
  lanjm1ip_anji.load(prefFileName, verbose);

  return THOT_OK;
}

bool IncrHmmAlignmentModel::print(const char* prefFileName, int verbose)
{
  bool retVal = HmmAlignmentModel::print(prefFileName, verbose);
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

void IncrHmmAlignmentModel::clear()
{
  HmmAlignmentModel::clear();
  lanji.clear();
  lanjm1ip_anji.clear();
}

void IncrHmmAlignmentModel::clearTempVars()
{
  HmmAlignmentModel::clearTempVars();
  trainer.clear();
}
