/*
thot package for statistical machine translation
Copyright (C) 2013-2017 Daniel Ortiz-Mart\'inez, Adam Harasimowicz

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
 * @file IncrLexTable.h
 *
 * @brief Defines the IncrLexTable class. IncrLexTable class stores an
 * incremental lexical table.
 *
 */

#pragma once

#include "nlp_common/AwkInputStream.h"
#include "nlp_common/ErrorDefs.h"
#include "nlp_common/StatModelDefs.h"
#include "sw_models/_incrLexTable.h"

#include <fstream>
#include <set>
#include <vector>

#ifdef THOT_DISABLE_SPACE_EFFICIENT_LEXDATA_STRUCTURES
#include <unordered_map>
#else
#include "nlp_common/OrderedVector.h"
#endif

class IncrLexTable : public _incrLexTable
{
public:
  // Constructor and destructor
  IncrLexTable();
  ~IncrLexTable();

  // Functions to handle lexNumer
  void setLexNumer(WordIndex s, WordIndex t, float f);
  float getLexNumer(WordIndex s, WordIndex t, bool& found);

  // Functions to handle lexDenom
  void setLexDenom(WordIndex s, float f);
  float getLexDenom(WordIndex s, bool& found);

  // Function to set lexical numerator and denominator
  void setLexNumDen(WordIndex s, WordIndex t, float num, float den);

  // Functions to get translations for word
  bool getTransForSource(WordIndex t, std::set<WordIndex>& transSet);

  // load function
  bool load(const char* lexNumDenFile, int verbose = 0);

  // print function
  bool print(const char* lexNumDenFile);

  void reserveSpace(WordIndex s);

  // clear() function
  void clear();

protected:
  // Lexical model types
#ifdef THOT_DISABLE_SPACE_EFFICIENT_LEXDATA_STRUCTURES
  typedef std::unordered_map<WordIndex, float> LexNumerElem;
#else
  typedef OrderedVector<WordIndex, float> LexNumerElem;
#endif

  typedef std::vector<LexNumerElem> LexNumer;
  typedef std::vector<std::pair<bool, float>> LexDenom;

  LexNumer lexNumer;
  LexDenom lexDenom;

  // load and print auxiliary functions
  bool loadBin(const char* lexNumDenFile, int verbose);
  bool loadPlainText(const char* lexNumDenFile, int verbose);
  bool printBin(const char* lexNumDenFile);
  bool printPlainText(const char* lexNumDenFile);
};

