#include "sw_models/_swAligModel.h"

#include "nlp_common/ErrorDefs.h"
#include "nlp_common/StrProcUtils.h"

using namespace std;

_swAligModel::_swAligModel()
    : alpha{0.01}, variationalBayes{false}, swVocab{make_shared<SingleWordVocab>()},
      sentenceHandler{make_shared<LightSentenceHandler>()}
{
}

_swAligModel::_swAligModel(_swAligModel& model)
    : alpha{model.alpha}, variationalBayes{model.variationalBayes}, swVocab{model.swVocab}, sentenceHandler{
                                                                                                model.sentenceHandler}
{
}

bool _swAligModel::modelReadsAreProcessSafe()
{
  // By default it will be assumed that model reads are thread safe,
  // those unsafe classes will override this method returning false
  // instead
  return true;
}

void _swAligModel::setVariationalBayes(bool variationalBayes)
{
  this->variationalBayes = variationalBayes;
}

bool _swAligModel::getVariationalBayes()
{
  return variationalBayes;
}

bool _swAligModel::readSentencePairs(const char* srcFileName, const char* trgFileName, const char* sentCountsFile,
                                     pair<unsigned int, unsigned int>& sentRange, int verbose)
{
  return sentenceHandler->readSentencePairs(srcFileName, trgFileName, sentCountsFile, sentRange, verbose);
}

void _swAligModel::addSentPair(vector<string> srcSentStr, vector<string> trgSentStr, Count c,
                               pair<unsigned int, unsigned int>& sentRange)
{
  sentenceHandler->addSentPair(srcSentStr, trgSentStr, c, sentRange);
}

unsigned int _swAligModel::numSentPairs()
{
  return sentenceHandler->numSentPairs();
}

int _swAligModel::nthSentPair(unsigned int n, vector<string>& srcSentStr, vector<string>& trgSentStr, Count& c)
{
  return sentenceHandler->nthSentPair(n, srcSentStr, trgSentStr, c);
}

pair<double, double> _swAligModel::loglikelihoodForAllSents(int verbosity)
{
  pair<unsigned int, unsigned int> sentPairRange = make_pair(0, numSentPairs() - 1);
  return loglikelihoodForPairRange(sentPairRange, verbosity);
}

LgProb _swAligModel::calcLgProbForAligChar(const char* sSent, const char* tSent, const WordAligMatrix& aligMatrix,
                                           int verbose)
{
  vector<string> sSentVec, tSentVec;

  sSentVec = StrProcUtils::charItemsToVector(sSent);
  tSentVec = StrProcUtils::charItemsToVector(tSent);
  return calcLgProbForAligVecStr(sSentVec, tSentVec, aligMatrix, verbose);
}

LgProb _swAligModel::calcLgProbForAligVecStr(const vector<string>& sSent, const vector<string>& tSent,
                                             const WordAligMatrix& aligMatrix, int verbose)
{
  vector<WordIndex> sIndexVector, tIndexVector;

  sIndexVector = strVectorToSrcIndexVector(sSent);
  tIndexVector = strVectorToTrgIndexVector(tSent);
  return calcLgProbForAlig(sIndexVector, tIndexVector, aligMatrix, verbose);
}

LgProb _swAligModel::calcLgProbChar(const char* sSent, const char* tSent, int verbose)
{
  vector<string> sSentVec, tSentVec;

  sSentVec = StrProcUtils::charItemsToVector(sSent);
  tSentVec = StrProcUtils::charItemsToVector(tSent);
  return calcLgProbVecStr(sSentVec, tSentVec, verbose);
}

LgProb _swAligModel::calcLgProbVecStr(const vector<string>& sSent, const vector<string>& tSent, int verbose)
{
  vector<WordIndex> sIndexVector, tIndexVector;
  vector<PositionIndex> aligIndexVec;

  sIndexVector = strVectorToSrcIndexVector(sSent);
  tIndexVector = strVectorToTrgIndexVector(tSent);

  return calcLgProb(sIndexVector, tIndexVector, verbose);
}

LgProb _swAligModel::calcLgProbPhr(const vector<WordIndex>& sPhr, const vector<WordIndex>& tPhr, int verbose)
{
  return calcLgProb(sPhr, tPhr, verbose);
}

bool _swAligModel::obtainBestAlignments(const char* sourceTestFileName, const char* targetTestFilename,
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

LgProb _swAligModel::obtainBestAlignmentChar(const char* sourceSentence, const char* targetSentence,
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

LgProb _swAligModel::obtainBestAlignmentVecStr(const vector<string>& srcSentenceVector,
                                               const vector<string>& trgSentenceVector, WordAligMatrix& bestWaMatrix)
{
  LgProb lp;
  vector<WordIndex> srcSentIndexVector, trgSentIndexVector;

  srcSentIndexVector = strVectorToSrcIndexVector(srcSentenceVector);
  trgSentIndexVector = strVectorToTrgIndexVector(trgSentenceVector);
  lp = obtainBestAlignment(srcSentIndexVector, trgSentIndexVector, bestWaMatrix);

  return lp;
}

ostream& _swAligModel::printAligInGizaFormat(const char* sourceSentence, const char* targetSentence, Prob p,
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

bool _swAligModel::loadGIZASrcVocab(const char* srcInputVocabFileName, int verbose)
{
  return swVocab->loadGIZASrcVocab(srcInputVocabFileName, verbose);
}

bool _swAligModel::loadGIZATrgVocab(const char* trgInputVocabFileName, int verbose)
{
  return swVocab->loadGIZATrgVocab(trgInputVocabFileName, verbose);
}

bool _swAligModel::printGIZASrcVocab(const char* srcOutputVocabFileName)
{
  return swVocab->printSrcVocab(srcOutputVocabFileName);
}

bool _swAligModel::printGIZATrgVocab(const char* trgOutputVocabFileName)
{
  return swVocab->printTrgVocab(trgOutputVocabFileName);
}

bool _swAligModel::printSentPairs(const char* srcSentFile, const char* trgSentFile, const char* sentCountsFile)
{
  return sentenceHandler->printSentPairs(srcSentFile, trgSentFile, sentCountsFile);
}

size_t _swAligModel::getSrcVocabSize() const
{
  return swVocab->getSrcVocabSize();
}

WordIndex _swAligModel::stringToSrcWordIndex(string s) const
{
  return swVocab->stringToSrcWordIndex(s);
}

string _swAligModel::wordIndexToSrcString(WordIndex w) const
{
  return swVocab->wordIndexToSrcString(w);
}

bool _swAligModel::existSrcSymbol(string s) const
{
  return swVocab->existSrcSymbol(s);
}

vector<WordIndex> _swAligModel::strVectorToSrcIndexVector(vector<string> s)
{
  return swVocab->strVectorToSrcIndexVector(s);
}

WordIndex _swAligModel::addSrcSymbol(string s)
{
  return swVocab->addSrcSymbol(s);
}

size_t _swAligModel::getTrgVocabSize() const
{
  return swVocab->getTrgVocabSize();
}

WordIndex _swAligModel::stringToTrgWordIndex(string t) const
{
  return swVocab->stringToTrgWordIndex(t);
}

string _swAligModel::wordIndexToTrgString(WordIndex w) const
{
  return swVocab->wordIndexToTrgString(w);
}

bool _swAligModel::existTrgSymbol(string t) const
{
  return swVocab->existTrgSymbol(t);
}

vector<WordIndex> _swAligModel::strVectorToTrgIndexVector(vector<string> t)
{
  return swVocab->strVectorToTrgIndexVector(t);
}

WordIndex _swAligModel::addTrgSymbol(string t)
{
  return swVocab->addTrgSymbol(t);
}

void _swAligModel::clear(void)
{
  swVocab->clear();
  sentenceHandler->clear();
}

bool _swAligModel::loadVariationalBayes(const string& filename)
{
  ifstream in(filename);
  if (!in)
    return THOT_ERROR;
  in >> variationalBayes >> alpha;

  return THOT_OK;
}

bool _swAligModel::printVariationalBayes(const string& filename)
{
  ofstream out(filename);
  if (!out)
    return THOT_ERROR;
  out << variationalBayes << " " << alpha;
  return THOT_OK;
}

vector<WordIndex> _swAligModel::addNullWordToWidxVec(const vector<WordIndex>& vw)
{
  vector<WordIndex> result;

  result.push_back(NULL_WORD);
  for (unsigned int i = 0; i < vw.size(); ++i)
    result.push_back(vw[i]);

  return result;
}

vector<string> _swAligModel::addNullWordToStrVec(const vector<string>& vw)
{
  vector<string> result;

  result.push_back(NULL_WORD_STR);
  for (unsigned int i = 0; i < vw.size(); ++i)
    result.push_back(vw[i]);

  return result;
}
