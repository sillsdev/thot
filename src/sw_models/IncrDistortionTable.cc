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
 * @file IncrDistortionTable.cc
 *
 * @brief Definitions file for IncrDistortionTable.h
 */

#include "sw_models/IncrDistortionTable.h"

IncrDistortionTable::IncrDistortionTable()
{
}

void IncrDistortionTable::setDistortionNumer(dSource ds, PositionIndex j, float f)
{
  DistortionNumerElem& distortionNumerElem = distortionNumer[ds];
  if (distortionNumerElem.size() != ds.tlen)
    distortionNumerElem.resize(ds.tlen);
  distortionNumerElem[j - 1] = f;
}

float IncrDistortionTable::getDistortionNumer(dSource ds, PositionIndex j, bool& found) const
{
  auto iter = distortionNumer.find(ds);
  if (iter != distortionNumer.end())
  {
    if (iter->second.size() == ds.tlen)
    {
      found = true;
      return iter->second[j - 1];
    }
  }

  found = false;
  return 0;
}

void IncrDistortionTable::setDistortionDenom(dSource ds, float f)
{
  distortionDenom[ds] = f;
}

float IncrDistortionTable::getDistortionDenom(dSource ds, bool& found) const
{
  auto iter = distortionDenom.find(ds);
  if (iter != distortionDenom.end())
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

void IncrDistortionTable::setDistortionNumDen(dSource ds, PositionIndex j, float num, float den)
{
  setDistortionNumer(ds, j, num);
  setDistortionDenom(ds, den);
}

void IncrDistortionTable::reserveSpace(dSource ds)
{
  distortionNumer[ds];
  distortionDenom[ds];
}

bool IncrDistortionTable::load(const char* distortionNumDenFile, int verbose)
{
#ifdef THOT_ENABLE_LOAD_PRINT_TEXTPARS
  return loadPlainText(distortionNumDenFile, verbose);
#else
  return loadBin(distortionNumDenFile, verbose);
#endif
}

bool IncrDistortionTable::loadPlainText(const char* distortionNumDenFile, int verbose)
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
        dSource ds;
        ds.i = atoi(awk.dollar(1).c_str());
        ds.slen = atoi(awk.dollar(2).c_str());
        ds.tlen = atoi(awk.dollar(3).c_str());
        PositionIndex j = atoi(awk.dollar(4).c_str());
        float numer = (float)atof(awk.dollar(5).c_str());
        float denom = (float)atof(awk.dollar(6).c_str());
        setDistortionNumDen(ds, j, numer, denom);
      }
    }
    return THOT_OK;
  }
}

bool IncrDistortionTable::loadBin(const char* distortionNumDenFile, int verbose)
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
      dSource ds;
      PositionIndex j;
      float numer;
      float denom;
      if (inF.read((char*)&ds.i, sizeof(PositionIndex)))
      {
        inF.read((char*)&ds.slen, sizeof(PositionIndex));
        inF.read((char*)&ds.tlen, sizeof(PositionIndex));
        inF.read((char*)&j, sizeof(PositionIndex));
        inF.read((char*)&numer, sizeof(float));
        inF.read((char*)&denom, sizeof(float));
        setDistortionNumDen(ds, j, numer, denom);
      }
      else
        end = true;
    }
    return THOT_OK;
  }
}

bool IncrDistortionTable::print(const char* distortionNumDenFile) const
{
#ifdef THOT_ENABLE_LOAD_PRINT_TEXTPARS
  return printPlainText(distortionNumDenFile);
#else
  return printBin(distortionNumDenFile);
#endif
}

bool IncrDistortionTable::printPlainText(const char* distortionNumDenFile) const
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
    for (auto& elem : distortionNumer)
    {
      for (PositionIndex j = 1; j <= distortionNumer.size(); ++j)
      {
        outF << elem.first.i << " ";
        outF << elem.first.slen << " ";
        outF << elem.first.tlen << " ";
        outF << j << " ";
        outF << elem.second[j - 1] << " ";
        bool found;
        float denom = getDistortionDenom(elem.first, found);
        outF << denom << std::endl;
      }
    }
    return THOT_OK;
  }
}

bool IncrDistortionTable::printBin(const char* distortionNumDenFile) const
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
    for (auto& elem : distortionNumer)
    {
      for (PositionIndex j = 1; j <= distortionNumer.size(); ++j)
      {
        outF.write((char*)&elem.first.i, sizeof(PositionIndex));
        outF.write((char*)&elem.first.slen, sizeof(PositionIndex));
        outF.write((char*)&elem.first.tlen, sizeof(PositionIndex));
        outF.write((char*)&j, sizeof(PositionIndex));
        outF.write((char*)&elem.second[j - 1], sizeof(float));
        bool found;
        float denom = getDistortionDenom(elem.first, found);
        outF.write((char*)&denom, sizeof(float));
      }
    }
    return THOT_OK;
  }
}

void IncrDistortionTable::clear()
{
  distortionNumer.clear();
  distortionDenom.clear();
}
