#include "nlp_common/WordClasses.h"

#include "nlp_common/ErrorDefs.h"

#include <fstream>
#include <iostream>

using namespace std;

WordClasses::WordClasses() : srcWordClassCount{0}, trgWordClassCount{0}
{
}

void WordClasses::addSrcWordClass(WordIndex s, WordClassIndex c)
{
  if (srcWordClasses.size() <= s)
    srcWordClasses.resize(s + 1);
  srcWordClasses[s] = c;
  if (c >= srcWordClassCount)
    srcWordClassCount = c + 1;
}

void WordClasses::addTrgWordClass(WordIndex t, WordClassIndex c)
{
  if (trgWordClasses.size() <= t)
    trgWordClasses.resize(t + 1);
  trgWordClasses[t] = c;
  if (c >= trgWordClassCount)
    trgWordClassCount = c + 1;
}

WordClassIndex WordClasses::getSrcWordClass(WordIndex s) const
{
  if (s >= srcWordClasses.size())
    return NULL_WORD_CLASS;
  return srcWordClasses[s];
}

WordClassIndex WordClasses::getTrgWordClass(WordIndex t) const
{
  if (t >= trgWordClasses.size())
    return NULL_WORD_CLASS;
  return trgWordClasses[t];
}

WordClassIndex WordClasses::getSrcWordClassCount() const
{
  return srcWordClassCount;
}

WordClassIndex WordClasses::getTrgWordClassCount() const
{
  return trgWordClassCount;
}

bool WordClasses::loadSrcWordClasses(const char* srcWordClassesFile, int verbose)
{
  return loadBin(srcWordClassesFile, srcWordClasses, srcWordClassCount, verbose);
}

bool WordClasses::loadTrgWordClasses(const char* trgWordClassesFile, int verbose)
{
  return loadBin(trgWordClassesFile, trgWordClasses, trgWordClassCount, verbose);
}

bool WordClasses::printSrcWordClasses(const char* srcWordClassesFile, int verbose) const
{
  return printBin(srcWordClassesFile, srcWordClasses, verbose);
}

bool WordClasses::printTrgWordClasses(const char* trgWordClassesFile, int verbose) const
{
  return printBin(trgWordClassesFile, trgWordClasses, verbose);
}

void WordClasses::clear()
{
  srcWordClasses.clear();
  trgWordClasses.clear();
  srcWordClassCount = 0;
  trgWordClassCount = 0;
}

bool WordClasses::loadBin(const char* wordClassesFile, vector<WordClassIndex>& wordClasses,
                          WordClassIndex& wordClassCount, int verbose)
{
  wordClasses.clear();
  wordClassCount = 0;

  if (verbose)
    cerr << "Loading word classes file in binary format from " << wordClassesFile << endl;

  // Try to open file
  ifstream inF(wordClassesFile, ios::in | ios::binary);
  if (!inF)
  {
    if (verbose)
      cerr << "Error in word classes file, file " << wordClassesFile << " does not exist.\n";
    return THOT_ERROR;
  }
  else
  {
    // Read register
    bool end = false;
    while (!end)
    {
      WordIndex w;
      WordClassIndex c;
      if (inF.read((char*)&w, sizeof(WordIndex)))
      {
        inF.read((char*)&c, sizeof(WordClassIndex));

        if (wordClasses.size() <= w)
          wordClasses.resize(w + 1);
        wordClasses[w] = c;
        if (c >= wordClassCount)
          wordClassCount = c + 1;
      }
      else
      {
        end = true;
      }
    }
    return THOT_OK;
  }
}

bool WordClasses::printBin(const char* wordClassesFile, const vector<WordClassIndex>& wordClasses, int verbose) const
{
  ofstream outF;
  outF.open(wordClassesFile, ios::out | ios::binary);
  if (!outF)
  {
    if (verbose)
      cerr << "Error while printing word classes file." << endl;
    return THOT_ERROR;
  }
  else
  {
    for (WordIndex w = 0; w < wordClasses.size(); ++w)
    {
      outF.write((char*)&w, sizeof(WordIndex));
      outF.write((char*)&wordClasses[w], sizeof(WordClassIndex));
    }
    return THOT_OK;
  }
}
