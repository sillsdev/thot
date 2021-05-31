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
 * @file IncrLexTable.cc
 *
 * @brief Definitions file for IncrLexTable.h
 */

//--------------- Include files --------------------------------------

#include "sw_models/IncrLexTable.h"

using namespace std;

IncrLexTable::IncrLexTable()
{
}

void IncrLexTable::setLexNumer(WordIndex s, WordIndex t, float f)
{
  // Grow lexNumer
  if (lexNumer.size() <= s)
    lexNumer.resize(s + 1);

  // Insert lexNumer for pair s,t
  lexNumer[s][t] = f;
}

float IncrLexTable::getLexNumer(WordIndex s, WordIndex t, bool& found) const
{
  if (s >= lexNumer.size())
  {
    // entry for s in lexNumer does not exist
    found = false;
    return 0;
  }
  else
  {
    // entry for s in lexNumer exists
    auto lexNumerElemIter = lexNumer[s].find(t);
    if (lexNumerElemIter != lexNumer[s].end())
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

void IncrLexTable::setLexDenom(WordIndex s, float d)
{
  reserveSpace(s);
  lexDenom[s] = make_pair(true, d);
}

float IncrLexTable::getLexDenom(WordIndex s, bool& found) const
{
  if (lexDenom.size() > s)
  {
    found = lexDenom[s].first;
    return lexDenom[s].second;
  }
  else
  {
    found = false;
    return 0;
  }
}

bool IncrLexTable::getTransForSource(WordIndex s, set<WordIndex>& transSet) const
{
  transSet.clear();

  if (s >= lexNumer.size())
  {
    return false;
  }
  else
  {
    for (auto &numElemIter : lexNumer[s])
    {
      transSet.insert(numElemIter.first);
    }
    return true;
  }
}

void IncrLexTable::setLexNumDen(WordIndex s, WordIndex t, float num, float den)
{
  setLexDenom(s, den);
  setLexNumer(s, t, num);
}

bool IncrLexTable::load(const char* lexNumDenFile, int verbose)
{
#ifdef THOT_ENABLE_LOAD_PRINT_TEXTPARS
  return loadPlainText(lexNumDenFile, verbose);
#else
  return loadBin(lexNumDenFile, verbose);
#endif
}

bool IncrLexTable::loadBin(const char* lexNumDenFile, int verbose)
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
        setLexNumDen(s, t, numer, denom);
      }
      else
        end = true;
    }
    return THOT_OK;
  }
}

bool IncrLexTable::loadPlainText(const char* lexNumDenFile, int verbose)
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
        setLexNumDen(s, t, numer, denom);
      }
    }
    return THOT_OK;
  }
}

bool IncrLexTable::print(const char* lexNumDenFile) const
{
#ifdef THOT_ENABLE_LOAD_PRINT_TEXTPARS
  return printPlainText(lexNumDenFile);
#else
  return printBin(lexNumDenFile);
#endif
}

bool IncrLexTable::printBin(const char* lexNumDenFile) const
{
  ofstream outF;
  outF.open(lexNumDenFile, ios::out | ios::binary);
  if (!outF)
  {
    cerr << "Error while printing lexical nd file." << endl;
    return THOT_ERROR;
  }
  else
  {
    // print file with lexical nd values
    for (WordIndex s = 0; s < lexNumer.size(); ++s)
    {
      LexNumerElem::const_iterator numElemIter;
      for (numElemIter = lexNumer[s].begin(); numElemIter != lexNumer[s].end(); ++numElemIter)
      {
        bool found;
        outF.write((char*)&s, sizeof(WordIndex));
        outF.write((char*)&numElemIter->first, sizeof(WordIndex));
        outF.write((char*)&numElemIter->second, sizeof(float));
        float denom = getLexDenom(s, found);
        outF.write((char*)&denom, sizeof(float));
      }
    }
    return THOT_OK;
  }
}

bool IncrLexTable::printPlainText(const char* lexNumDenFile) const
{
  ofstream outF;
  outF.open(lexNumDenFile, ios::out);
  if (!outF)
  {
    std::cerr << "Error while printing lexical nd file." << std::endl;
    return THOT_ERROR;
  }
  else
  {
    // print file with lexical nd values
    for (WordIndex s = 0; s < lexNumer.size(); ++s)
    {
      LexNumerElem::const_iterator numElemIter;
      for (numElemIter = lexNumer[s].begin(); numElemIter != lexNumer[s].end(); ++numElemIter)
      {
        bool found;
        outF << s << " ";
        outF << numElemIter->first << " ";
        outF << numElemIter->second << " ";
        float denom = getLexDenom(s, found);
        outF << denom << std::endl;
        ;
      }
    }
    return THOT_OK;
  }
}

void IncrLexTable::reserveSpace(WordIndex s)
{
  if (lexNumer.size() <= s)
    lexNumer.resize(s + 1);

  if (lexDenom.size() <= s)
  {
    pair<bool, float> pair(false, 0.0f);
    lexDenom.resize(s + 1, pair);
  }
}

void IncrLexTable::clear()
{
  lexNumer.clear();
  lexDenom.clear();
}

IncrLexTable::~IncrLexTable()
{
  // Nothing to do
}
