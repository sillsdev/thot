#include "sw_models/DistortionTable.h"

#include <fstream>
#include <iostream>

void DistortionTable::setNumerator(PositionIndex i, PositionIndex slen, PositionIndex tlen, PositionIndex j, float f)
{
  DistortionKey key{i, slen, tlen};
  NumeratorsElem& distortionNumerElem = numerators[key];
  if (distortionNumerElem.size() != tlen)
    distortionNumerElem.resize(tlen);
  distortionNumerElem[j - 1] = f;
}

float DistortionTable::getNumerator(PositionIndex i, PositionIndex slen, PositionIndex tlen, PositionIndex j,
                                    bool& found) const
{
  DistortionKey key{i, slen, tlen};
  auto iter = numerators.find(key);
  if (iter != numerators.end())
  {
    if (iter->second.size() == tlen)
    {
      found = true;
      return iter->second[j - 1];
    }
  }

  found = false;
  return 0;
}

void DistortionTable::setDenominator(PositionIndex i, PositionIndex slen, PositionIndex tlen, float f)
{
  DistortionKey key{i, slen, tlen};
  denominators[key] = f;
}

float DistortionTable::getDenominator(PositionIndex i, PositionIndex slen, PositionIndex tlen, bool& found) const
{
  DistortionKey key{i, slen, tlen};
  auto iter = denominators.find(key);
  if (iter != denominators.end())
  {
    // ds is stored in distortionDenom
    found = true;
    return iter->second;
  }
  else
  {
    // ds is not stored in distortionDenom
    found = false;
    return 0;
  }
}

void DistortionTable::set(PositionIndex i, PositionIndex slen, PositionIndex tlen, PositionIndex j, float num,
                          float den)
{
  setNumerator(i, slen, tlen, j, num);
  setDenominator(i, slen, tlen, den);
}

void DistortionTable::reserveSpace(PositionIndex i, PositionIndex slen, PositionIndex tlen)
{
  DistortionKey key{i, slen, tlen};
  numerators[key];
  denominators[key];
}

bool DistortionTable::load(const char* distortionNumDenFile, int verbose)
{
#ifdef THOT_ENABLE_LOAD_PRINT_TEXTPARS
  return loadPlainText(distortionNumDenFile, verbose);
#else
  return loadBin(distortionNumDenFile, verbose);
#endif
}

bool DistortionTable::loadPlainText(const char* distortionNumDenFile, int verbose)
{
  // Clear data structures
  clear();

  if (verbose)
    std::cerr << "Loading distortion nd file in plain text format from " << distortionNumDenFile << std::endl;

  AwkInputStream awk;
  if (awk.open(distortionNumDenFile) == THOT_ERROR)
  {
    if (verbose)
      std::cerr << "Error in distortion nd file, file " << distortionNumDenFile << " does not exist.\n";
    return THOT_ERROR;
  }
  else
  {
    // Read entries
    while (awk.getln())
    {
      if (awk.NF == 6)
      {
        PositionIndex i = atoi(awk.dollar(1).c_str());
        PositionIndex slen = atoi(awk.dollar(2).c_str());
        PositionIndex tlen = atoi(awk.dollar(3).c_str());
        PositionIndex j = atoi(awk.dollar(4).c_str());
        float numer = (float)atof(awk.dollar(5).c_str());
        float denom = (float)atof(awk.dollar(6).c_str());
        set(i, slen, tlen, j, numer, denom);
      }
    }
    return THOT_OK;
  }
}

bool DistortionTable::loadBin(const char* distortionNumDenFile, int verbose)
{
  // Clear data structures
  clear();

  if (verbose)
    std::cerr << "Loading distortion nd file in binary format from " << distortionNumDenFile << std::endl;

  // Try to open file
  std::ifstream inF(distortionNumDenFile, std::ios::in | std::ios::binary);
  if (!inF)
  {
    if (verbose)
      std::cerr << "Error in distortion nd file, file " << distortionNumDenFile << " does not exist.\n";
    return THOT_ERROR;
  }
  else
  {
    // Read register
    bool end = false;
    while (!end)
    {
      PositionIndex i;
      PositionIndex slen;
      PositionIndex tlen;
      PositionIndex j;
      float numer;
      float denom;
      if (inF.read((char*)&i, sizeof(PositionIndex)))
      {
        inF.read((char*)&slen, sizeof(PositionIndex));
        inF.read((char*)&tlen, sizeof(PositionIndex));
        inF.read((char*)&j, sizeof(PositionIndex));
        inF.read((char*)&numer, sizeof(float));
        inF.read((char*)&denom, sizeof(float));
        set(i, slen, tlen, j, numer, denom);
      }
      else
        end = true;
    }
    return THOT_OK;
  }
}

bool DistortionTable::print(const char* distortionNumDenFile) const
{
#ifdef THOT_ENABLE_LOAD_PRINT_TEXTPARS
  return printPlainText(distortionNumDenFile);
#else
  return printBin(distortionNumDenFile);
#endif
}

bool DistortionTable::printPlainText(const char* distortionNumDenFile) const
{
  std::ofstream outF;
  outF.open(distortionNumDenFile, std::ios::out);
  if (!outF)
  {
    std::cerr << "Error while printing distortion nd file." << std::endl;
    return THOT_ERROR;
  }
  else
  {
    // print file with distortion nd values
    for (auto& elem : numerators)
    {
      for (PositionIndex j = 1; j <= elem.second.size(); ++j)
      {
        outF << elem.first.i << " ";
        outF << elem.first.slen << " ";
        outF << elem.first.tlen << " ";
        outF << j << " ";
        outF << elem.second[j - 1] << " ";
        bool found;
        float denom = getDenominator(elem.first.i, elem.first.slen, elem.first.tlen, found);
        outF << denom << std::endl;
      }
    }
    return THOT_OK;
  }
}

bool DistortionTable::printBin(const char* distortionNumDenFile) const
{
  std::ofstream outF;
  outF.open(distortionNumDenFile, std::ios::out | std::ios::binary);
  if (!outF)
  {
    std::cerr << "Error while printing distortion nd file." << std::endl;
    return THOT_ERROR;
  }
  else
  {
    // print file with alignment nd values
    for (auto& elem : numerators)
    {
      for (PositionIndex j = 1; j <= elem.second.size(); ++j)
      {
        outF.write((char*)&elem.first.i, sizeof(PositionIndex));
        outF.write((char*)&elem.first.slen, sizeof(PositionIndex));
        outF.write((char*)&elem.first.tlen, sizeof(PositionIndex));
        outF.write((char*)&j, sizeof(PositionIndex));
        outF.write((char*)&elem.second[j - 1], sizeof(float));
        bool found;
        float denom = getDenominator(elem.first.i, elem.first.slen, elem.first.tlen, found);
        outF.write((char*)&denom, sizeof(float));
      }
    }
    return THOT_OK;
  }
}

void DistortionTable::clear()
{
  numerators.clear();
  denominators.clear();
}
