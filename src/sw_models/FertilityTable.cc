#include "sw_models/FertilityTable.h"

#include <iostream>

using namespace std;

void FertilityTable::setNumerator(WordIndex s, PositionIndex phi, float f)
{
  if (numerators.size() <= s)
    numerators.resize(s + 1);
  if (numerators[s].size() <= phi)
    numerators[s].resize(phi + 1);

  // Insert numerator for pair s,phi
  numerators[s][phi] = f;
}

float FertilityTable::getNumerator(WordIndex s, PositionIndex phi, bool& found) const
{
  if (s >= numerators.size())
  {
    // entry for s in fertilityNumer does not exist
    found = false;
    return 0;
  }
  else
  {
    // entry for s in fertilityNumer exists

    if (phi >= numerators[s].size())
    {
      // entry for s,phi in fertilityNumer does not exist
      found = false;
      return 0;
    }
    else
    {
      // entry for s,phi in fertilityNumer exists
      found = true;
      return numerators[s][phi];
    }
  }
}

void FertilityTable::setDenominator(WordIndex s, float d)
{
  if (denominators.size() <= s)
    denominators.resize(s + 1, 0.0f);
  denominators[s] = d;
}

float FertilityTable::getDenominator(WordIndex s, bool& found) const
{
  if (denominators.size() > s)
  {
    found = true;
    return denominators[s];
  }
  else
  {
    found = false;
    return 0;
  }
}

void FertilityTable::set(WordIndex s, PositionIndex phi, float num, float den)
{
  setDenominator(s, den);
  setNumerator(s, phi, num);
}

bool FertilityTable::load(const char* fertilityNumDenFile, int verbose)
{
#ifdef THOT_ENABLE_LOAD_PRINT_TEXTPARS
  return loadPlainText(fertilityNumDenFile, verbose);
#else
  return loadBin(fertilityNumDenFile, verbose);
#endif
}

bool FertilityTable::loadBin(const char* fertilityNumDenFile, int verbose)
{
  // Clear data structures
  clear();

  if (verbose)
    cerr << "Loading fertility nd file in binary format from " << fertilityNumDenFile << endl;

  // Try to open file
  ifstream inF(fertilityNumDenFile, ios::in | ios::binary);
  if (!inF)
  {
    if (verbose)
      cerr << "Error in fertility nd file, file " << fertilityNumDenFile << " does not exist.\n";
    return THOT_ERROR;
  }
  else
  {
    // Read register
    bool end = false;
    while (!end)
    {
      WordIndex s;
      PositionIndex phi;
      float numer;
      float denom;
      if (inF.read((char*)&s, sizeof(WordIndex)))
      {
        inF.read((char*)&phi, sizeof(PositionIndex));
        inF.read((char*)&numer, sizeof(float));
        inF.read((char*)&denom, sizeof(float));
        set(s, phi, numer, denom);
      }
      else
        end = true;
    }
    return THOT_OK;
  }
}

bool FertilityTable::loadPlainText(const char* fertilityNumDenFile, int verbose)
{
  // Clear data structures
  clear();

  if (verbose)
    cerr << "Loading fertility nd file in plain text format from " << fertilityNumDenFile << endl;

  AwkInputStream awk;
  if (awk.open(fertilityNumDenFile) == THOT_ERROR)
  {
    if (verbose)
      cerr << "Error in fertility nd file, file " << fertilityNumDenFile << " does not exist.\n";
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
        PositionIndex phi = atoi(awk.dollar(2).c_str());
        float numer = (float)atof(awk.dollar(3).c_str());
        float denom = (float)atof(awk.dollar(4).c_str());
        set(s, phi, numer, denom);
      }
    }
    return THOT_OK;
  }
}

bool FertilityTable::print(const char* fertilityNumDenFile) const
{
#ifdef THOT_ENABLE_LOAD_PRINT_TEXTPARS
  return printPlainText(lexNumDenFile);
#else
  return printBin(fertilityNumDenFile);
#endif
}

bool FertilityTable::printBin(const char* fertilityNumDenFile) const
{
  ofstream outF;
  outF.open(fertilityNumDenFile, ios::out | ios::binary);
  if (!outF)
  {
    cerr << "Error while printing fertility nd file." << endl;
    return THOT_ERROR;
  }
  else
  {
    // print file with fertility nd values
    for (WordIndex s = 0; s < numerators.size(); ++s)
    {
      for (PositionIndex phi = 0; phi < numerators[s].size(); ++phi)
      {
        bool found;
        outF.write((char*)&s, sizeof(WordIndex));
        outF.write((char*)&phi, sizeof(PositionIndex));
        outF.write((char*)&numerators[s][phi], sizeof(float));
        float denom = getDenominator(s, found);
        outF.write((char*)&denom, sizeof(float));
      }
    }
    return THOT_OK;
  }
}

bool FertilityTable::printPlainText(const char* fertilityNumDenFile) const
{
  ofstream outF;
  outF.open(fertilityNumDenFile, ios::out);
  if (!outF)
  {
    std::cerr << "Error while printing lexical nd file." << std::endl;
    return THOT_ERROR;
  }
  else
  {
    // print file with lexical nd values
    for (WordIndex s = 0; s < numerators.size(); ++s)
    {
      for (PositionIndex phi = 0; phi < numerators[s].size(); ++phi)
      {
        bool found;
        outF << s << " ";
        outF << phi << " ";
        outF << numerators[s][phi] << " ";
        float denom = getDenominator(s, found);
        outF << denom << std::endl;
        ;
      }
    }
    return THOT_OK;
  }
}

void FertilityTable::reserveSpace(WordIndex s)
{
  if (numerators.size() <= s)
    numerators.resize(s + 1);

  if (denominators.size() <= s)
    denominators.resize(s + 1, 0.0f);
}

void FertilityTable::clear()
{
  numerators.clear();
  denominators.clear();
}
