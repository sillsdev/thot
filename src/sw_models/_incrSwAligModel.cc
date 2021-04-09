#include "_incrSwAligModel.h"

using namespace std;

void _incrSwAligModel::efficientBatchTrainingForAllSents(int verbosity)
{
  if (this->numSentPairs() > 0)
    efficientBatchTrainingForRange(std::make_pair(0, this->numSentPairs() - 1), verbosity);
}