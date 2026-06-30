#include "sw_models/SymmetrizedAlignmentModel.h"

using namespace std;

SymmetrizedAlignmentModel::SymmetrizedAlignmentModel(shared_ptr<AlignmentModel> directModel,
                                                     shared_ptr<AlignmentModel> inverseModel)
    : SymmetrizedAligner{directModel, inverseModel}, directModel{directModel}, inverseModel{inverseModel}
{
}

unsigned int SymmetrizedAlignmentModel::numSentencePairs()
{
  return directModel->numSentencePairs();
}

int SymmetrizedAlignmentModel::getSentencePair(unsigned int n, vector<string>& srcSentStr, vector<string>& trgSentStr,
                                               Count& c)
{
  return directModel->getSentencePair(n, srcSentStr, trgSentStr, c);
}

LgProb SymmetrizedAlignmentModel::getTrainingAlignment(size_t n, WordAlignmentMatrix& bestWaMatrix)
{
  LgProb logProb = directModel->getTrainingAlignment(n, bestWaMatrix);
  if (getHeuristic() == SymmetrizationHeuristic::None)
    return logProb;

  WordAlignmentMatrix invMatrix;
  LgProb invLogProb = inverseModel->getTrainingAlignment(n, invMatrix);
  invMatrix.transpose();

  // Skip the combine when the matrices are degenerate or their dimensions don't
  // line up (e.g. an out-of-range n, or a pair filtered out of training in only
  // one direction): the heuristic operations require matching dimensions.
  if (bestWaMatrix.get_I() == 0 || bestWaMatrix.get_J() == 0 || invMatrix.get_I() != bestWaMatrix.get_I() ||
      invMatrix.get_J() != bestWaMatrix.get_J())
    return logProb;

  applyHeuristic(bestWaMatrix, invMatrix, getHeuristic());
  return max(logProb, invLogProb);
}

LgProb SymmetrizedAlignmentModel::getTrainingAlignment(size_t n, vector<PositionIndex>& alignment)
{
  WordAlignmentMatrix waMatrix;
  LgProb logProb = getTrainingAlignment(n, waMatrix);
  waMatrix.getAligVec(alignment);
  return logProb;
}
