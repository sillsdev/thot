#include "sw_models/MemoryLexTable.h"

#include "nlp_common/AwkInputStream.h"
#include "nlp_common/ErrorDefs.h"

#include <fstream>

using namespace std;

void MemoryLexTable::setNumerator(WordIndex s, WordIndex t, float f)
{
  reserveSpace(s);

  // Insert numerator for pair s,t
  numerators[s][t] = f;
}

float MemoryLexTable::getNumerator(WordIndex s, WordIndex t, bool& found) const
{
  if (s >= numerators.size())
  {
    // entry for s in lexNumer does not exist
    found = false;
    return 0;
  }
  else
  {
    // entry for s in lexNumer exists
    auto lexNumerElemIter = numerators[s].find(t);
    if (lexNumerElemIter != numerators[s].end())
    {
      // lexNumer for pair s,t exists
      found = true;
      return lexNumerElemIter->second;
    }
    else
    {
      // lexNumer for pair s,t does not exist
      found = false;
      return 0;
    }
  }
}

void MemoryLexTable::setDenominator(WordIndex s, float d)
{
  reserveSpace(s);
  denominators[s] = make_pair(true, d);
}

float MemoryLexTable::getDenominator(WordIndex s, bool& found) const
{
  if (denominators.size() > s)
  {
    found = denominators[s].first;
    return denominators[s].second;
  }
  else
  {
    found = false;
    return 0;
  }
}

bool MemoryLexTable::getTransForSource(WordIndex s, std::set<WordIndex>& transSet) const
{
  transSet.clear();

  if (s >= numerators.size())
  {
    return false;
  }
  else
  {
    for (auto& numElemIter : numerators[s])
    {
      transSet.insert(numElemIter.first);
    }
    return true;
  }
}

void MemoryLexTable::set(WordIndex s, WordIndex t, float num, float den)
{
  setDenominator(s, den);
  setNumerator(s, t, num);
}

bool MemoryLexTable::load(const char* lexNumDenFile, int verbose)
{
#ifdef THOT_ENABLE_LOAD_PRINT_TEXTPARS
  return loadPlainText(lexNumDenFile, verbose);
#else
  return loadBin(lexNumDenFile, verbose);
#endif
}

bool MemoryLexTable::loadBin(const char* lexNumDenFile, int verbose)
{
  // Clear data structures
  clear();

  if (verbose)
    cerr << "Loading lexnd file in binary format from " << lexNumDenFile << endl;

  // Try to open file
  ifstream inF(lexNumDenFile, ios::in | ios::binary);
  if (!inF)
  {
    if (verbose)
      cerr << "Error in lexical nd file, file " << lexNumDenFile << " does not exist.\n";
    return THOT_ERROR;
  }
  else
  {
    // Read register
    bool end = false;
    while (!end)
    {
      WordIndex s;
      WordIndex t;
      float numer;
      float denom;
      if (inF.read((char*)&s, sizeof(WordIndex)))
      {
        inF.read((char*)&t, sizeof(WordIndex));
        inF.read((char*)&numer, sizeof(float));
        inF.read((char*)&denom, sizeof(float));
        set(s, t, numer, denom);
      }
      else
        end = true;
    }
    return THOT_OK;
  }
}

bool MemoryLexTable::loadPlainText(const char* lexNumDenFile, int verbose)
{
  // Clear data structures
  clear();

  if (verbose)
    cerr << "Loading lexnd file in plain text format from " << lexNumDenFile << endl;

  AwkInputStream awk;
  if (awk.open(lexNumDenFile) == THOT_ERROR)
  {
    if (verbose)
      cerr << "Error in file with lexical parameters, file " << lexNumDenFile << " does not exist.\n";
    return THOT_ERROR;
  }
  else
  {
    // Read gaussian parameters
    while (awk.getln())
    {
      if (awk.NF == 4)
      {
        WordIndex s = atoi(awk.dollar(1).c_str());
        WordIndex t = atoi(awk.dollar(2).c_str());
        float numer = (float)atof(awk.dollar(3).c_str());
        float denom = (float)atof(awk.dollar(4).c_str());
        set(s, t, numer, denom);
      }
    }
    return THOT_OK;
  }
}

bool MemoryLexTable::print(const char* lexNumDenFile, int verbose) const
{
#ifdef THOT_ENABLE_LOAD_PRINT_TEXTPARS
  return printPlainText(lexNumDenFile, verbose);
#else
  return printBin(lexNumDenFile, verbose);
#endif
}

bool MemoryLexTable::printBin(const char* lexNumDenFile, int verbose) const
{
  ofstream outF;
  outF.open(lexNumDenFile, ios::out | ios::binary);
  if (!outF)
  {
    if (verbose)
      cerr << "Error while printing lexical nd file." << endl;
    return THOT_ERROR;
  }
  else
  {
    // print file with lexical nd values
    for (WordIndex s = 0; s < numerators.size(); ++s)
    {
      NumeratorsElem::const_iterator numElemIter;
      for (numElemIter = numerators[s].begin(); numElemIter != numerators[s].end(); ++numElemIter)
      {
        bool found;
        outF.write((char*)&s, sizeof(WordIndex));
        outF.write((char*)&numElemIter->first, sizeof(WordIndex));
        outF.write((char*)&numElemIter->second, sizeof(float));
        float denom = getDenominator(s, found);
        outF.write((char*)&denom, sizeof(float));
      }
    }
    return THOT_OK;
  }
}

bool MemoryLexTable::printPlainText(const char* lexNumDenFile, int verbose) const
{
  ofstream outF;
  outF.open(lexNumDenFile, ios::out);
  if (!outF)
  {
    if (verbose)
      std::cerr << "Error while printing lexical nd file." << std::endl;
    return THOT_ERROR;
  }
  else
  {
    // print file with lexical nd values
    for (WordIndex s = 0; s < numerators.size(); ++s)
    {
      NumeratorsElem::const_iterator numElemIter;
      for (numElemIter = numerators[s].begin(); numElemIter != numerators[s].end(); ++numElemIter)
      {
        outF << s << " ";
        outF << numElemIter->first << " ";
        outF << numElemIter->second << " ";
        bool found;
        float denom = getDenominator(s, found);
        outF << denom << std::endl;
        ;
      }
    }
    return THOT_OK;
  }
}

void MemoryLexTable::reserveSpace(WordIndex s)
{
  if (numerators.size() <= s)
    numerators.resize(s + 1);

  if (denominators.size() <= s)
  {
    pair<bool, float> pair(false, 0.0f);
    denominators.resize(s + 1, pair);
  }
}

void MemoryLexTable::clear()
{
  numerators.clear();
  denominators.clear();
}
