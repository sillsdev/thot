#include "sw_models/AlignmentTable.h"

#include "nlp_common/AwkInputStream.h"
#include "nlp_common/ErrorDefs.h"

#include <fstream>
#include <iostream>

void AlignmentTable::setNumerator(PositionIndex j, PositionIndex slen, PositionIndex tlen, PositionIndex i, float f)
{
  AlignmentKey key{j, slen, tlen};
  NumeratorsElem& aligNumerElem = numerators[key];
  if (aligNumerElem.size() != slen + 1)
    aligNumerElem.resize(slen + 1);
  aligNumerElem[i] = f;
}

float AlignmentTable::getNumerator(PositionIndex j, PositionIndex slen, PositionIndex tlen, PositionIndex i,
                                   bool& found) const
{
  AlignmentKey key{j, slen, tlen};
  auto iter = numerators.find(key);
  if (iter != numerators.end())
  {
    if (iter->second.size() == slen + 1)
    {
      found = true;
      return iter->second[i];
    }
  }

  found = false;
  return 0;
}

void AlignmentTable::setDenominator(PositionIndex j, PositionIndex slen, PositionIndex tlen, float f)
{
  AlignmentKey key{j, slen, tlen};
  denominators[key] = f;
}

float AlignmentTable::getDenominator(PositionIndex j, PositionIndex slen, PositionIndex tlen, bool& found) const
{
  AlignmentKey key{j, slen, tlen};
  auto iter = denominators.find(key);
  if (iter != denominators.end())
  {
    // s is stored in aligDenom
    found = true;
    return iter->second;
  }
  else
  {
    // s is not stored in aligDenom
    found = false;
    return 0;
  }
}

void AlignmentTable::set(PositionIndex j, PositionIndex slen, PositionIndex tlen, PositionIndex i, float num, float den)
{
  setNumerator(j, slen, tlen, i, num);
  setDenominator(j, slen, tlen, den);
}

void AlignmentTable::reserveSpace(PositionIndex j, PositionIndex slen, PositionIndex tlen)
{
  AlignmentKey key{j, slen, tlen};
  numerators[key];
  denominators[key];
}

bool AlignmentTable::load(const char* aligNumDenFile, int verbose)
{
#ifdef THOT_ENABLE_LOAD_PRINT_TEXTPARS
  return loadPlainText(aligNumDenFile, verbose);
#else
  return loadBin(aligNumDenFile, verbose);
#endif
}

bool AlignmentTable::loadPlainText(const char* aligNumDenFile, int verbose)
{
  // Clear data structures
  clear();

  if (verbose)
    std::cerr << "Loading alignd file in plain text format from " << aligNumDenFile << std::endl;

  AwkInputStream awk;
  if (awk.open(aligNumDenFile) == THOT_ERROR)
  {
    if (verbose)
      std::cerr << "Error in alignment nd file, file " << aligNumDenFile << " does not exist.\n";
    return THOT_ERROR;
  }
  else
  {
    // Read entries
    while (awk.getln())
    {
      if (awk.NF == 6)
      {
        PositionIndex j = atoi(awk.dollar(1).c_str());
        PositionIndex slen = atoi(awk.dollar(2).c_str());
        PositionIndex tlen = atoi(awk.dollar(3).c_str());
        PositionIndex i = atoi(awk.dollar(4).c_str());
        float numer = (float)atof(awk.dollar(5).c_str());
        float denom = (float)atof(awk.dollar(6).c_str());
        set(j, slen, tlen, i, numer, denom);
      }
    }
    return THOT_OK;
  }
}

bool AlignmentTable::loadBin(const char* aligNumDenFile, int verbose)
{
  // Clear data structures
  clear();

  if (verbose)
    std::cerr << "Loading alignd file in binary format from " << aligNumDenFile << std::endl;

  // Try to open file
  std::ifstream inF(aligNumDenFile, std::ios::in | std::ios::binary);
  if (!inF)
  {
    if (verbose)
      std::cerr << "Error in alignment nd file, file " << aligNumDenFile << " does not exist.\n";
    return THOT_ERROR;
  }
  else
  {
    // Read register
    bool end = false;
    while (!end)
    {
      PositionIndex j;
      PositionIndex slen;
      PositionIndex tlen;
      PositionIndex i;
      float numer;
      float denom;
      if (inF.read((char*)&j, sizeof(PositionIndex)))
      {
        inF.read((char*)&slen, sizeof(PositionIndex));
        inF.read((char*)&tlen, sizeof(PositionIndex));
        inF.read((char*)&i, sizeof(PositionIndex));
        inF.read((char*)&numer, sizeof(float));
        inF.read((char*)&denom, sizeof(float));
        set(j, slen, tlen, i, numer, denom);
      }
      else
        end = true;
    }
    return THOT_OK;
  }
}

bool AlignmentTable::print(const char* aligNumDenFile) const
{
#ifdef THOT_ENABLE_LOAD_PRINT_TEXTPARS
  return printPlainText(aligNumDenFile);
#else
  return printBin(aligNumDenFile);
#endif
}

bool AlignmentTable::printPlainText(const char* aligNumDenFile) const
{
  std::ofstream outF;
  outF.open(aligNumDenFile, std::ios::out);
  if (!outF)
  {
    std::cerr << "Error while printing alignment nd file." << std::endl;
    return THOT_ERROR;
  }
  else
  {
    // print file with alignment nd values
    for (auto& elem : numerators)
    {
      for (PositionIndex i = 0; i < elem.second.size(); ++i)
      {
        outF << elem.first.j << " ";
        outF << elem.first.slen << " ";
        outF << elem.first.tlen << " ";
        outF << i << " ";
        outF << elem.second[i] << " ";
        bool found;
        float denom = getDenominator(elem.first.j, elem.first.slen, elem.first.tlen, found);
        outF << denom << std::endl;
      }
    }
    return THOT_OK;
  }
}

bool AlignmentTable::printBin(const char* aligNumDenFile) const
{
  std::ofstream outF;
  outF.open(aligNumDenFile, std::ios::out | std::ios::binary);
  if (!outF)
  {
    std::cerr << "Error while printing alignment nd file." << std::endl;
    return THOT_ERROR;
  }
  else
  {
    // print file with alignment nd values
    for (auto& elem : numerators)
    {
      for (PositionIndex i = 0; i < elem.second.size(); ++i)
      {
        outF.write((char*)&elem.first.j, sizeof(PositionIndex));
        outF.write((char*)&elem.first.slen, sizeof(PositionIndex));
        outF.write((char*)&elem.first.tlen, sizeof(PositionIndex));
        outF.write((char*)&i, sizeof(PositionIndex));
        outF.write((char*)&elem.second[i], sizeof(float));
        bool found;
        float denom = getDenominator(elem.first.j, elem.first.slen, elem.first.tlen, found);
        outF.write((char*)&denom, sizeof(float));
      }
    }
    return THOT_OK;
  }
}

void AlignmentTable::clear()
{
  numerators.clear();
  denominators.clear();
}
