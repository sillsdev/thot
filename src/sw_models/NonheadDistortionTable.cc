#include "sw_models/NonheadDistortionTable.h"

#include "nlp_common/AwkInputStream.h"
#include "nlp_common/ErrorDefs.h"

#include <fstream>

void NonheadDistortionTable::setNumerator(WordClassIndex trgWordClass, int dj, float f)
{
  if (numerators.size() <= trgWordClass)
    numerators.resize(trgWordClass + 1);

  numerators[trgWordClass][dj] = f;
}

float NonheadDistortionTable::getNumerator(WordClassIndex trgWordClass, int dj, bool& found) const
{
  if (numerators.size() > trgWordClass)
  {
    auto iter = numerators[trgWordClass].find(dj);
    if (iter != numerators[trgWordClass].end())
    {
      found = true;
      return iter->second;
    }
  }

  found = false;
  return 0;
}

void NonheadDistortionTable::setDenominator(WordClassIndex trgWordClass, float f)
{
  if (denominators.size() <= trgWordClass)
    denominators.resize(trgWordClass + 1);

  denominators[trgWordClass] = std::make_pair(true, f);
}

float NonheadDistortionTable::getDenominator(WordClassIndex trgWordClass, bool& found) const
{
  if (denominators.size() > trgWordClass)
  {
    found = denominators[trgWordClass].first;
    return denominators[trgWordClass].second;
  }

  found = false;
  return 0;
}

void NonheadDistortionTable::set(WordClassIndex trgWordClass, int dj, float num, float den)
{
  setNumerator(trgWordClass, dj, num);
  setDenominator(trgWordClass, den);
}

void NonheadDistortionTable::reserveSpace(WordClassIndex trgWordClass)
{
  if (numerators.size() <= trgWordClass)
    numerators.resize(trgWordClass + 1);

  if (denominators.size() <= trgWordClass)
    denominators.resize(trgWordClass + 1);
}

bool NonheadDistortionTable::load(const char* tableFile, int verbose)
{
#ifdef THOT_ENABLE_LOAD_PRINT_TEXTPARS
  return loadPlainText(tableFile, verbose);
#else
  return loadBin(tableFile, verbose);
#endif
}

bool NonheadDistortionTable::print(const char* tableFile) const
{
#ifdef THOT_ENABLE_LOAD_PRINT_TEXTPARS
  return printPlainText(tableFile);
#else
  return printBin(tableFile);
#endif
}

void NonheadDistortionTable::clear()
{
  numerators.clear();
  denominators.clear();
}

bool NonheadDistortionTable::loadBin(const char* tableFile, int verbose)
{
  clear();

  if (verbose)
    std::cerr << "Loading nonhead distortion nd file in binary format from " << tableFile << std::endl;

  std::ifstream inF(tableFile, std::ios::in | std::ios::binary);
  if (!inF)
  {
    if (verbose)
      std::cerr << "Error in nonhead distortion nd file, file " << tableFile << " does not exist.\n";
    return THOT_ERROR;
  }

  bool end = false;
  while (!end)
  {
    WordClassIndex targetWordClass;
    int dj;
    float numer;
    float denom;
    if (inF.read((char*)&targetWordClass, sizeof(WordClassIndex)))
    {
      inF.read((char*)&dj, sizeof(int));
      inF.read((char*)&numer, sizeof(float));
      inF.read((char*)&denom, sizeof(float));
      set(targetWordClass, dj, numer, denom);
    }
    else
      end = true;
  }
  return THOT_OK;
}

bool NonheadDistortionTable::loadPlainText(const char* tableFile, int verbose)
{
  clear();

  if (verbose)
    std::cerr << "Loading nonhead distortion nd file in plain text format from " << tableFile << std::endl;

  AwkInputStream awk;
  if (awk.open(tableFile) == THOT_ERROR)
  {
    if (verbose)
      std::cerr << "Error in nonhead distortion nd file, file " << tableFile << " does not exist.\n";
    return THOT_ERROR;
  }

  while (awk.getln())
  {
    if (awk.NF == 6)
    {
      WordClassIndex targetWordClass = atoi(awk.dollar(1).c_str());
      int dj = atoi(awk.dollar(2).c_str());
      float numer = (float)atof(awk.dollar(3).c_str());
      float denom = (float)atof(awk.dollar(4).c_str());
      set(targetWordClass, dj, numer, denom);
    }
  }
  return THOT_OK;
}

bool NonheadDistortionTable::printBin(const char* tableFile) const
{
  std::ofstream outF;
  outF.open(tableFile, std::ios::out | std::ios::binary);
  if (!outF)
  {
    std::cerr << "Error while printing nonhead distortion nd file." << std::endl;
    return THOT_ERROR;
  }

  for (WordClassIndex targetWordClass = 0; targetWordClass < numerators.size(); ++targetWordClass)
  {
    for (auto& numPair : numerators[targetWordClass])
    {
      outF.write((char*)&targetWordClass, sizeof(WordClassIndex));
      outF.write((char*)&numPair.first, sizeof(int));
      outF.write((char*)&numPair.second, sizeof(float));
      bool found;
      float denom = getDenominator(targetWordClass, found);
      outF.write((char*)&denom, sizeof(float));
    }
  }
  return THOT_OK;
}

bool NonheadDistortionTable::printPlainText(const char* tableFile) const
{
  std::ofstream outF;
  outF.open(tableFile, std::ios::out);
  if (!outF)
  {
    std::cerr << "Error while printing nonhead distortion nd file." << std::endl;
    return THOT_ERROR;
  }

  for (WordClassIndex targetWordClass = 0; targetWordClass < numerators.size(); ++targetWordClass)
  {
    for (auto& numPair : numerators[targetWordClass])
    {
      outF << targetWordClass << " ";
      outF << numPair.first << " ";
      outF << numPair.second << " ";
      bool found;
      float denom = getDenominator(targetWordClass, found);
      outF << denom << std::endl;
    }
  }
  return THOT_OK;
}
