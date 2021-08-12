#include "sw_models/HmmAlignmentTable.h"

#include "nlp_common/AwkInputStream.h"
#include "nlp_common/ErrorDefs.h"

#include <fstream>
#include <iostream>

using namespace std;

void HmmAlignmentTable::setNumerator(PositionIndex prev_i, PositionIndex slen, PositionIndex i, float f)
{
  reserveSpace(prev_i, slen);

  if (numerators[prev_i][slen].size() <= i)
    numerators[prev_i][slen].resize((size_t)i + 1);

  numerators[prev_i][slen][i] = make_pair(true, f);
}

float HmmAlignmentTable::getNumerator(PositionIndex prev_i, PositionIndex slen, PositionIndex i, bool& found)
{
  if (prev_i < numerators.size() && slen < numerators[prev_i].size() && i < numerators[prev_i][slen].size()
      && numerators[prev_i][slen][i].first)
  {
    found = true;
    return numerators[prev_i][slen][i].second;
  }

  found = false;
  return 0;
}

void HmmAlignmentTable::setDenominator(PositionIndex prev_i, PositionIndex slen, float f)
{
  reserveSpace(prev_i, slen);
  denominators[prev_i][slen] = std::make_pair(true, f);
}

float HmmAlignmentTable::getDenominator(PositionIndex prev_i, PositionIndex slen, bool& found)
{
  if (prev_i < denominators.size() && slen < denominators[prev_i].size() && denominators[prev_i][slen].first)
  {
    found = true;
    return denominators[prev_i][slen].second;
  }

  found = false;
  return 0;
}

void HmmAlignmentTable::set(PositionIndex prev_i, PositionIndex slen, PositionIndex i, float num, float den)
{
  setNumerator(prev_i, slen, i, num);
  setDenominator(prev_i, slen, den);
}

void HmmAlignmentTable::reserveSpace(PositionIndex prev_i, PositionIndex slen)
{
  if (numerators.size() <= prev_i)
    numerators.resize((size_t)prev_i + 1);

  if (numerators[prev_i].size() <= slen)
    numerators[prev_i].resize((size_t)slen + 1);

  if (denominators.size() <= prev_i)
    denominators.resize((size_t)prev_i + 1);

  if (denominators[prev_i].size() <= slen)
    denominators[prev_i].resize((size_t)slen + 1);
}

bool HmmAlignmentTable::load(const char* aligNumDenFile, int verbose /*=0*/)
{
#ifdef THOT_ENABLE_LOAD_PRINT_TEXTPARS
  return loadPlainText(aligNumDenFile, verbose);
#else
  return loadBin(aligNumDenFile, verbose);
#endif
}

bool HmmAlignmentTable::loadPlainText(const char* aligNumDenFile, int verbose)
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
    while (awk.getln())
    {
      if (awk.NF == 5)
      {
        PositionIndex prev_i = atoi(awk.dollar(1).c_str());
        PositionIndex slen = atoi(awk.dollar(2).c_str());
        PositionIndex i = atoi(awk.dollar(3).c_str());
        float numer = (float)atof(awk.dollar(4).c_str());
        float denom = (float)atof(awk.dollar(5).c_str());
        set(prev_i, slen, i, numer, denom);
      }
    }
    return THOT_OK;
  }
}

bool HmmAlignmentTable::loadBin(const char* aligNumDenFile, int verbose)
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
      PositionIndex prev_i;
      PositionIndex slen;
      PositionIndex i;
      float numer;
      float denom;
      if (inF.read((char*)&prev_i, sizeof(PositionIndex)))
      {
        inF.read((char*)&slen, sizeof(PositionIndex));
        inF.read((char*)&i, sizeof(PositionIndex));
        inF.read((char*)&numer, sizeof(float));
        inF.read((char*)&denom, sizeof(float));
        set(prev_i, slen, i, numer, denom);
      }
      else
        end = true;
    }
    return THOT_OK;
  }
}

bool HmmAlignmentTable::print(const char* aligNumDenFile)
{
#ifdef THOT_ENABLE_LOAD_PRINT_TEXTPARS
  return printPlainText(aligNumDenFile);
#else
  return printBin(aligNumDenFile);
#endif
}

bool HmmAlignmentTable::printBin(const char* aligNumDenFile)
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
    for (PositionIndex prev_i = 0; prev_i < numerators.size(); ++prev_i)
    {
      for (PositionIndex slen = 0; slen < numerators[prev_i].size(); ++slen)
      {
        for (PositionIndex i = 0; i < numerators[prev_i][slen].size(); ++i)
        {
          if (numerators[prev_i][slen][i].first)
          {
            bool found;
            outF.write((char*)&prev_i, sizeof(PositionIndex));
            outF.write((char*)&slen, sizeof(PositionIndex));
            outF.write((char*)&i, sizeof(PositionIndex));
            outF.write((char*)&numerators[prev_i][slen][i].second, sizeof(float));
            float denom = getDenominator(prev_i, slen, found);
            outF.write((char*)&denom, sizeof(float));
          }
        }
      }
    }
    return THOT_OK;
  }
}

bool HmmAlignmentTable::printPlainText(const char* aligNumDenFile)
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
    for (PositionIndex prev_i = 0; prev_i < numerators.size(); ++prev_i)
    {
      for (PositionIndex slen = 0; slen < numerators[prev_i].size(); ++slen)
      {
        for (PositionIndex i = 0; i < numerators[prev_i][slen].size(); ++i)
        {
          if (numerators[prev_i][slen][i].first)
          {
            bool found;
            outF << prev_i << " ";
            outF << slen << " ";
            outF << i << " ";
            outF << numerators[prev_i][slen][i].second << " ";
            float denom = getDenominator(prev_i, slen, found);
            outF << denom << std::endl;
          }
        }
      }
    }
    return THOT_OK;
  }
}

void HmmAlignmentTable::clear()
{
  numerators.clear();
  denominators.clear();
}
