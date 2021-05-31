/*
thot package for statistical machine translation
Copyright (C) 2013 Daniel Ortiz-Mart\'inez

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public License
as published by the Free Software Foundation; either version 3
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program; If not, see <http://www.gnu.org/licenses/>.
*/

/**
 * @file IncrFertilityTable.cc
 *
 * @brief Definitions file for IncrFertilityTable.h
 */

 //--------------- Include files --------------------------------------

#include "IncrFertilityTable.h"

using namespace std;

IncrFertilityTable::IncrFertilityTable()
{
}

void IncrFertilityTable::setFertilityNumer(WordIndex s, PositionIndex phi, float f)
{
  if (fertilityNumer.size() <= s)
    fertilityNumer.resize(s + 1);
  if (fertilityNumer[s].size() <= phi)
    fertilityNumer[s].resize(phi + 1);

  // Insert numerator for pair s,phi
  fertilityNumer[s][phi] = f;
}

float IncrFertilityTable::getFertilityNumer(WordIndex s, PositionIndex phi, bool& found) const
{
  if (s >= fertilityNumer.size())
  {
    // entry for s in fertilityNumer does not exist
    found = false;
    return 0;
  }
  else
  {
    // entry for s in fertilityNumer exists

    if (phi >= fertilityNumer[s].size())
    {
      // entry for s,phi in fertilityNumer does not exist
      found = false;
      return 0;
    }
    else
    {
      // entry for s,phi in fertilityNumer exists
      found = true;
      return fertilityNumer[s][phi];
    }
  }
}

void IncrFertilityTable::setFertilityDenom(WordIndex s, float d)
{
  if (fertilityDenom.size() <= s)
    fertilityDenom.resize(s + 1, 0.0f);
  fertilityDenom[s] = d;
}

float IncrFertilityTable::getFertilityDenom(WordIndex s, bool& found) const
{
  if (fertilityDenom.size() > s)
  {
    found = true;
    return fertilityDenom[s];
  }
  else
  {
    found = false;
    return 0;
  }
}

void IncrFertilityTable::setFertilityNumDen(WordIndex s, PositionIndex phi, float num, float den)
{
  setFertilityDenom(s, den);
  setFertilityNumer(s, phi, num);
}

bool IncrFertilityTable::load(const char* fertilityNumDenFile, int verbose)
{
#ifdef THOT_ENABLE_LOAD_PRINT_TEXTPARS 
  return loadPlainText(fertilityNumDenFile, verbose);
#else
  return loadBin(fertilityNumDenFile, verbose);
#endif
}

bool IncrFertilityTable::loadBin(const char* fertilityNumDenFile, int verbose)
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
        setFertilityNumDen(s, phi, numer, denom);
      }
      else end = true;
    }
    return THOT_OK;
  }
}

bool IncrFertilityTable::loadPlainText(const char* fertilityNumDenFile, int verbose)
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
        setFertilityNumDen(s, phi, numer, denom);
      }
    }
    return THOT_OK;
  }
}

bool IncrFertilityTable::print(const char* fertilityNumDenFile) const
{
#ifdef THOT_ENABLE_LOAD_PRINT_TEXTPARS 
  return printPlainText(lexNumDenFile);
#else
  return printBin(fertilityNumDenFile);
#endif
}

bool IncrFertilityTable::printBin(const char* fertilityNumDenFile) const
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
    for (WordIndex s = 0; s < fertilityNumer.size(); ++s)
    {
      for (PositionIndex phi = 0; phi < fertilityNumer[s].size(); ++phi)
      {
        bool found;
        outF.write((char*)&s, sizeof(WordIndex));
        outF.write((char*)&phi, sizeof(PositionIndex));
        outF.write((char*)&fertilityNumer[s][phi], sizeof(float));
        float denom = getFertilityDenom(s, found);
        outF.write((char*)&denom, sizeof(float));
      }
    }
    return THOT_OK;
  }
}

bool IncrFertilityTable::printPlainText(const char* fertilityNumDenFile) const
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
    for (WordIndex s = 0; s < fertilityNumer.size(); ++s)
    {
      for (PositionIndex phi = 0; phi < fertilityNumer[s].size(); ++phi)
      {
        bool found;
        outF << s << " ";
        outF << phi << " ";
        outF << fertilityNumer[s][phi] << " ";
        float denom = getFertilityDenom(s, found);
        outF << denom << std::endl;;
      }
    }
    return THOT_OK;
  }
}

void IncrFertilityTable::reserveSpace(WordIndex s)
{
  if (fertilityNumer.size() <= s)
    fertilityNumer.resize(s + 1);

  if (fertilityDenom.size() <= s)
    fertilityDenom.resize(s + 1, 0.0f);
}

void IncrFertilityTable::clear()
{
  fertilityNumer.clear();
  fertilityDenom.clear();
}

IncrFertilityTable::~IncrFertilityTable()
{
  // Nothing to do
}
