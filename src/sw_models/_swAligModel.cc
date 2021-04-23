#include "_swAligModel.h"

using namespace std;

_swAligModel::_swAligModel()
{
}

bool _swAligModel::readSentencePairs(const char* srcFileName, const char* trgFileName, const char* sentCountsFile,
  pair<unsigned int, unsigned int>& sentRange, int verbose)
{
  return sentenceHandler.readSentencePairs(srcFileName, trgFileName, sentCountsFile, sentRange, verbose);
}

void _swAligModel::addSentPair(vector<string> srcSentStr, vector<string> trgSentStr, Count c,
  pair<unsigned int, unsigned int>& sentRange)
{
  sentenceHandler.addSentPair(srcSentStr, trgSentStr, c, sentRange);
}

unsigned int _swAligModel::numSentPairs(void)
{
  return sentenceHandler.numSentPairs();
}

int _swAligModel::nthSentPair(unsigned int n, vector<string>& srcSentStr, vector<string>& trgSentStr, Count& c)
{
  return sentenceHandler.nthSentPair(n, srcSentStr, trgSentStr, c);
}

bool _swAligModel::loadGIZASrcVocab(const char* srcInputVocabFileName, int verbose)
{
  return swVocab.loadGIZASrcVocab(srcInputVocabFileName, verbose);
}

bool _swAligModel::loadGIZATrgVocab(const char* trgInputVocabFileName, int verbose)
{
  return swVocab.loadGIZATrgVocab(trgInputVocabFileName, verbose);
}

bool _swAligModel::printGIZASrcVocab(const char* srcOutputVocabFileName)
{
  return swVocab.printSrcVocab(srcOutputVocabFileName);
}

bool _swAligModel::printGIZATrgVocab(const char* trgOutputVocabFileName)
{
  return swVocab.printTrgVocab(trgOutputVocabFileName);
}

bool _swAligModel::printSentPairs(const char* srcSentFile, const char* trgSentFile, const char* sentCountsFile)
{
  return sentenceHandler.printSentPairs(srcSentFile, trgSentFile, sentCountsFile);
}

size_t _swAligModel::getSrcVocabSize() const
{
  return swVocab.getSrcVocabSize();
}

WordIndex _swAligModel::stringToSrcWordIndex(string s) const
{
  return swVocab.stringToSrcWordIndex(s);
}

string _swAligModel::wordIndexToSrcString(WordIndex w) const
{
  return swVocab.wordIndexToSrcString(w);
}

bool _swAligModel::existSrcSymbol(string s) const
{
  return swVocab.existSrcSymbol(s);
}

vector<WordIndex> _swAligModel::strVectorToSrcIndexVector(vector<string> s)
{
  return swVocab.strVectorToSrcIndexVector(s);
}

WordIndex _swAligModel::addSrcSymbol(string s)
{
  return swVocab.addSrcSymbol(s);
}

size_t _swAligModel::getTrgVocabSize() const
{
  return swVocab.getTrgVocabSize();
}

WordIndex _swAligModel::stringToTrgWordIndex(string t) const
{
  return swVocab.stringToTrgWordIndex(t);
}

string _swAligModel::wordIndexToTrgString(WordIndex w) const
{
  return swVocab.wordIndexToTrgString(w);
}

bool _swAligModel::existTrgSymbol(string t) const
{
  return swVocab.existTrgSymbol(t);
}

vector<WordIndex> _swAligModel::strVectorToTrgIndexVector(vector<string> t)
{
  return swVocab.strVectorToTrgIndexVector(t);
}

WordIndex _swAligModel::addTrgSymbol(string t)
{
  return swVocab.addTrgSymbol(t);
}

void _swAligModel::clear(void)
{
  swVocab.clear();
  sentenceHandler.clear();
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

_swAligModel::~_swAligModel()
{
}