#include "sw_models/HeadDistortionTable.h"

#include "nlp_common/AwkInputStream.h"
#include "nlp_common/ErrorDefs.h"

#include <fstream>

void HeadDistortionTable::setNumerator(WordClassIndex srcWordClass, WordClassIndex trgWordClass, int dj, float f)
{
  HeadDistortionKey key{srcWordClass, trgWordClass};
  numerators[key][dj] = f;
}

float HeadDistortionTable::getNumerator(WordClassIndex srcWordClass, WordClassIndex trgWordClass, int dj,
                                        bool& found) const
{
  HeadDistortionKey key{srcWordClass, trgWordClass};
  auto iter = numerators.find(key);
  if (iter != numerators.end())
  {
    auto elemIter = iter->second.find(dj);
    if (elemIter != iter->second.end())
    {
      found = true;
      return elemIter->second;
    }
  }

  found = false;
  return 0;
}

void HeadDistortionTable::setDenominator(WordClassIndex srcWordClass, WordClassIndex trgWordClass, float f)
{
  HeadDistortionKey key{srcWordClass, trgWordClass};
  denominators[key] = f;
}

float HeadDistortionTable::getDenominator(WordClassIndex srcWordClass, WordClassIndex trgWordClass, bool& found) const
{
  HeadDistortionKey key{srcWordClass, trgWordClass};
  auto iter = denominators.find(key);
  if (iter != denominators.end())
  {
    found = true;
    return iter->second;
  }

  found = false;
  return 0;
}

void HeadDistortionTable::set(WordClassIndex srcWordClass, WordClassIndex trgWordClass, int dj, float num, float den)
{
  setNumerator(srcWordClass, trgWordClass, dj, num);
  setDenominator(srcWordClass, trgWordClass, den);
}

void HeadDistortionTable::reserveSpace(WordClassIndex srcWordClass, WordClassIndex trgWordClass)
{
  HeadDistortionKey key{srcWordClass, trgWordClass};
  numerators[key];
  denominators[key];
}

bool HeadDistortionTable::load(const char* tableFile, int verbose)
{
#ifdef THOT_ENABLE_LOAD_PRINT_TEXTPARS
  return loadPlainText(tableFile, verbose);
#else
  return loadBin(tableFile, verbose);
#endif
}

bool HeadDistortionTable::print(const char* tableFile) const
{
#ifdef THOT_ENABLE_LOAD_PRINT_TEXTPARS
  return printPlainText(tableFile);
#else
  return printBin(tableFile);
#endif
}

void HeadDistortionTable::clear()
{
  numerators.clear();
  denominators.clear();
}

bool HeadDistortionTable::loadPlainText(const char* tableFile, int verbose)
{
  clear();

  if (verbose)
    std::cerr << "Loading head distortion nd file in plain text format from " << tableFile << std::endl;

  AwkInputStream awk;
  if (awk.open(tableFile) == THOT_ERROR)
  {
    if (verbose)
      std::cerr << "Error in head distortion nd file, file " << tableFile << " does not exist.\n";
    return THOT_ERROR;
  }

  while (awk.getln())
  {
    if (awk.NF == 6)
    {
      WordClassIndex sourceWordClass = atoi(awk.dollar(1).c_str());
      WordClassIndex targetWordClass = atoi(awk.dollar(2).c_str());
      int dj = atoi(awk.dollar(3).c_str());
      float numer = (float)atof(awk.dollar(4).c_str());
      float denom = (float)atof(awk.dollar(5).c_str());
      set(sourceWordClass, targetWordClass, dj, numer, denom);
    }
  }
  return THOT_OK;
}

bool HeadDistortionTable::loadBin(const char* tableFile, int verbose)
{
  clear();

  if (verbose)
    std::cerr << "Loading head distortion nd file in binary format from " << tableFile << std::endl;

  std::ifstream inF(tableFile, std::ios::in | std::ios::binary);
  if (!inF)
  {
    if (verbose)
      std::cerr << "Error in head distortion nd file, file " << tableFile << " does not exist.\n";
    return THOT_ERROR;
  }

  bool end = false;
  while (!end)
  {
    WordClassIndex sourceWordClass;
    WordClassIndex targetWordClass;
    int dj;
    float numer;
    float denom;
    if (inF.read((char*)&sourceWordClass, sizeof(WordClassIndex)))
    {
      inF.read((char*)&targetWordClass, sizeof(WordClassIndex));
      inF.read((char*)&dj, sizeof(int));
      inF.read((char*)&numer, sizeof(float));
      inF.read((char*)&denom, sizeof(float));
      set(sourceWordClass, targetWordClass, dj, numer, denom);
    }
    else
      end = true;
  }
  return THOT_OK;
}

bool HeadDistortionTable::printPlainText(const char* tableFile) const
{
  std::ofstream outF;
  outF.open(tableFile, std::ios::out);
  if (!outF)
  {
    std::cerr << "Error while printing head distortion nd file." << std::endl;
    return THOT_ERROR;
  }

  for (auto& numElemPair : numerators)
  {
    for (auto& numPair : numElemPair.second)
    {
      outF << numElemPair.first.srcWordClass << " ";
      outF << numElemPair.first.trgWordClass << " ";
      outF << numPair.first << " ";
      outF << numPair.second << " ";
      bool found;
      float denom = getDenominator(numElemPair.first.srcWordClass, numElemPair.first.trgWordClass, found);
      outF << denom << std::endl;
    }
  }
  return THOT_OK;
}

bool HeadDistortionTable::printBin(const char* tableFile) const
{
  std::ofstream outF;
  outF.open(tableFile, std::ios::out | std::ios::binary);
  if (!outF)
  {
    std::cerr << "Error while printing head distortion nd file." << std::endl;
    return THOT_ERROR;
  }

  for (auto& numElemPair : numerators)
  {
    for (auto& numPair : numElemPair.second)
    {
      outF.write((char*)&numElemPair.first.srcWordClass, sizeof(WordClassIndex));
      outF.write((char*)&numElemPair.first.trgWordClass, sizeof(WordClassIndex));
      outF.write((char*)&numPair.first, sizeof(int));
      outF.write((char*)&numPair.second, sizeof(float));
      bool found;
      float denom = getDenominator(numElemPair.first.srcWordClass, numElemPair.first.trgWordClass, found);
      outF.write((char*)&denom, sizeof(float));
    }
  }
  return THOT_OK;
}
