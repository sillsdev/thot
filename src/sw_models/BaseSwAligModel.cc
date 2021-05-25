#include "sw_models/BaseSwAligModel.h"

using namespace std;

BaseSwAligModel::BaseSwAligModel()
{
}

bool BaseSwAligModel::modelReadsAreProcessSafe()
{
  // By default it will be assumed that model reads are thread safe,
  // those unsafe classes will override this method returning false
  // instead
  return true;
}

void BaseSwAligModel::setVariationalBayes(bool variationalBayes)
{
  this->variationalBayes = variationalBayes;
}

bool BaseSwAligModel::getVariationalBayes()
{
  return variationalBayes;
}

pair<double, double> BaseSwAligModel::loglikelihoodForAllSents(int verbosity)
{
  pair<unsigned int, unsigned int> sentPairRange = make_pair(0, numSentPairs() - 1);
  return loglikelihoodForPairRange(sentPairRange, verbosity);
}

LgProb BaseSwAligModel::calcLgProbForAligChar(const char* sSent, const char* tSent, const WordAligMatrix& aligMatrix,
                                              int verbose)
{
  vector<string> sSentVec, tSentVec;

  sSentVec = StrProcUtils::charItemsToVector(sSent);
  tSentVec = StrProcUtils::charItemsToVector(tSent);
  return calcLgProbForAligVecStr(sSentVec, tSentVec, aligMatrix, verbose);
}

LgProb BaseSwAligModel::calcLgProbForAligVecStr(const vector<string>& sSent, const vector<string>& tSent,
                                                const WordAligMatrix& aligMatrix, int verbose)
{
  vector<WordIndex> sIndexVector, tIndexVector;

  sIndexVector = strVectorToSrcIndexVector(sSent);
  tIndexVector = strVectorToTrgIndexVector(tSent);
  return calcLgProbForAlig(sIndexVector, tIndexVector, aligMatrix, verbose);
}

LgProb BaseSwAligModel::calcLgProbChar(const char* sSent, const char* tSent, int verbose)
{
  vector<string> sSentVec, tSentVec;

  sSentVec = StrProcUtils::charItemsToVector(sSent);
  tSentVec = StrProcUtils::charItemsToVector(tSent);
  return calcLgProbVecStr(sSentVec, tSentVec, verbose);
}

LgProb BaseSwAligModel::calcLgProbVecStr(const vector<string>& sSent, const vector<string>& tSent, int verbose)
{
  vector<WordIndex> sIndexVector, tIndexVector;
  vector<PositionIndex> aligIndexVec;

  sIndexVector = strVectorToSrcIndexVector(sSent);
  tIndexVector = strVectorToTrgIndexVector(tSent);

  return calcLgProb(sIndexVector, tIndexVector, verbose);
}

LgProb BaseSwAligModel::calcLgProbPhr(const vector<WordIndex>& sPhr, const vector<WordIndex>& tPhr, int verbose)
{
  return calcLgProb(sPhr, tPhr, verbose);
}

bool BaseSwAligModel::obtainBestAlignments(const char* sourceTestFileName, const char* targetTestFilename,
                                           const char* outFileName)
{
  AwkInputStream srcTest, trgTest;
  WordAligMatrix waMatrix;
  vector<PositionIndex> bestAlig;
  LgProb bestLgProb;
  ofstream outF;

  outF.open(outFileName, ios::out);
  if (!outF)
  {
    cerr << "Error while opening output file." << endl;
    return 1;
  }

  if (srcTest.open(sourceTestFileName) == THOT_ERROR)
  {
    cerr << "Error in source test file, file " << sourceTestFileName << " does not exist.\n";
    return THOT_ERROR;
  }
  if (trgTest.open(targetTestFilename) == THOT_ERROR)
  {
    cerr << "Error in target test file, file " << targetTestFilename << " does not exist.\n";
    return THOT_ERROR;
  }
  while (srcTest.getln())
  {
    if (trgTest.getln())
    {
      if (srcTest.NF > 0 && trgTest.NF > 0)
      {
        bestLgProb =
            obtainBestAlignmentChar((char*)srcTest.dollar(0).c_str(), (char*)trgTest.dollar(0).c_str(), waMatrix);
        outF << "# Sentence pair " << srcTest.FNR << " ";
        waMatrix.getAligVec(bestAlig);
        printAligInGizaFormat((char*)srcTest.dollar(0).c_str(), (char*)trgTest.dollar(0).c_str(), bestLgProb.get_p(),
                              bestAlig, outF);
      }
    }
    else
    {
      cerr << "Error: Source and target test files have not the same size." << endl;
    }
  }
  outF.close();

  return THOT_OK;
}

LgProb BaseSwAligModel::obtainBestAlignmentChar(const char* sourceSentence, const char* targetSentence,
                                                WordAligMatrix& bestWaMatrix)
{
  vector<string> targetVector, sourceVector;
  LgProb lp;

  // Convert sourceSentence into a vector of strings
  sourceVector = StrProcUtils::charItemsToVector(sourceSentence);

  // Convert targetSentence into a vector of strings
  targetVector = StrProcUtils::charItemsToVector(targetSentence);
  lp = obtainBestAlignmentVecStr(sourceVector, targetVector, bestWaMatrix);

  return lp;
}

LgProb BaseSwAligModel::obtainBestAlignmentVecStr(const vector<string>& srcSentenceVector,
  const vector<string>& trgSentenceVector, WordAligMatrix& bestWaMatrix)
{
  LgProb lp;
  vector<WordIndex> srcSentIndexVector, trgSentIndexVector;

  srcSentIndexVector = strVectorToSrcIndexVector(srcSentenceVector);
  trgSentIndexVector = strVectorToTrgIndexVector(trgSentenceVector);
  lp = obtainBestAlignment(srcSentIndexVector, trgSentIndexVector, bestWaMatrix);

  return lp;
}

ostream& BaseSwAligModel::printAligInGizaFormat(const char* sourceSentence, const char* targetSentence, Prob p,
                                                vector<PositionIndex> alig, ostream& outS)
{
  vector<string> targetVector, sourceVector;
  unsigned int i, j;

  outS << "alignment score : " << p << endl;
  outS << targetSentence << endl;
  sourceVector = StrProcUtils::charItemsToVector(sourceSentence);
  targetVector = StrProcUtils::charItemsToVector(targetSentence);

  outS << "NULL ({ ";
  for (j = 0; j < alig.size(); ++j)
    if (alig[j] == 0)
      outS << j + 1 << " ";
  outS << "}) ";
  for (i = 0; i < sourceVector.size(); ++i)
  {
    outS << sourceVector[i] << " ({ ";
    for (j = 0; j < alig.size(); ++j)
      if (alig[j] == i + 1)
        outS << j + 1 << " ";
    outS << "}) ";
  }
  outS << endl;
  return outS;
}

vector<WordIndex> BaseSwAligModel::addNullWordToWidxVec(const vector<WordIndex>& vw)
{
  vector<WordIndex> result;

  result.push_back(NULL_WORD);
  for (unsigned int i = 0; i < vw.size(); ++i)
    result.push_back(vw[i]);

  return result;
}

vector<string> BaseSwAligModel::addNullWordToStrVec(const vector<string>& vw)
{
  vector<string> result;

  result.push_back(NULL_WORD_STR);
  for (unsigned int i = 0; i < vw.size(); ++i)
    result.push_back(vw[i]);

  return result;
}

BaseSwAligModel::~BaseSwAligModel()
{
}