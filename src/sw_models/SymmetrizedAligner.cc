#include "sw_models/SymmetrizedAligner.h"

#include "nlp_common/StrProcUtils.h"

using namespace std;

SymmetrizedAligner::SymmetrizedAligner(shared_ptr<Aligner> directAligner, shared_ptr<Aligner> inverseAligner)
    : directAligner{directAligner}, inverseAligner{inverseAligner}
{
}

void SymmetrizedAligner::setHeuristic(SymmetrizationHeuristic value)
{
  heuristic = value;
}

SymmetrizationHeuristic SymmetrizedAligner::getHeuristic() const
{
  return heuristic;
}

LgProb SymmetrizedAligner::getBestAlignment(const char* srcSentence, const char* trgSentence,
                                            WordAlignmentMatrix& bestWaMatrix)
{
  vector<string> sourceVector = StrProcUtils::charItemsToVector(srcSentence);
  vector<string> targetVector = StrProcUtils::charItemsToVector(trgSentence);
  return getBestAlignment(sourceVector, targetVector, bestWaMatrix);
}

LgProb SymmetrizedAligner::getBestAlignment(const vector<string>& srcSentence, const vector<string>& trgSentence,
                                            WordAlignmentMatrix& bestWaMatrix)
{
  vector<WordIndex> srcWordIndexVector = strVectorToSrcIndexVector(srcSentence);
  vector<WordIndex> trgWordIndexVector = strVectorToTrgIndexVector(trgSentence);
  return getBestAlignment(srcWordIndexVector, trgWordIndexVector, bestWaMatrix);
}

LgProb SymmetrizedAligner::getBestAlignment(const vector<WordIndex>& srcSentence, const vector<WordIndex>& trgSentence,
                                            WordAlignmentMatrix& bestWaMatrix)
{
  LgProb logProb = directAligner->getBestAlignment(srcSentence, trgSentence, bestWaMatrix);
  if (heuristic == SymmetrizationHeuristic::None)
    return logProb;

  WordAlignmentMatrix invMatrix;
  LgProb invLogProb = inverseAligner->getBestAlignment(trgSentence, srcSentence, invMatrix);
  invMatrix.transpose();
  switch (heuristic)
  {
  case SymmetrizationHeuristic::Union:
    bestWaMatrix |= invMatrix;
    break;
  case SymmetrizationHeuristic::Intersection:
    bestWaMatrix &= invMatrix;
    break;
  case SymmetrizationHeuristic::Och:
    bestWaMatrix.symmetr1(invMatrix);
    break;
  case SymmetrizationHeuristic::Grow:
    bestWaMatrix.grow(invMatrix);
    break;
  case SymmetrizationHeuristic::GrowDiag:
    bestWaMatrix.growDiag(invMatrix);
    break;
  case SymmetrizationHeuristic::GrowDiagFinal:
    bestWaMatrix.growDiagFinal(invMatrix);
    break;
  case SymmetrizationHeuristic::GrowDiagFinalAnd:
    bestWaMatrix.growDiagFinalAnd(invMatrix);
    break;
  case SymmetrizationHeuristic::None:
    break;
  }
  return max(logProb, invLogProb);
}

WordIndex SymmetrizedAligner::stringToSrcWordIndex(string s) const
{
  return directAligner->stringToSrcWordIndex(s);
}

vector<WordIndex> SymmetrizedAligner::strVectorToSrcIndexVector(vector<string> s)
{
  return directAligner->strVectorToSrcIndexVector(s);
}

WordIndex SymmetrizedAligner::stringToTrgWordIndex(string t) const
{
  return directAligner->stringToTrgWordIndex(t);
}

vector<WordIndex> SymmetrizedAligner::strVectorToTrgIndexVector(vector<string> t)
{
  return directAligner->strVectorToTrgIndexVector(t);
}
