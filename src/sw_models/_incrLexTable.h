/*
thot package for statistical machine translation
Copyright (C) 2017 Adam Harasimowicz

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
 * @file _incrLexTable.h
 *
 * @brief Defines interface for incremental lexical table.
 *
 */

#pragma once

#include "nlp_common/StatModelDefs.h"

#include <set>
#include <vector>

class _incrLexTable
{
public:
  // Functions to handle lexNumer
  virtual void setLexNumer(WordIndex s, WordIndex t, float f) = 0;
  virtual float getLexNumer(WordIndex s, WordIndex t, bool& found) const = 0;

  // Functions to handle lexDenom
  virtual void setLexDenom(WordIndex s, float f) = 0;
  virtual float getLexDenom(WordIndex s, bool& found) const = 0;

  // Function to set lexical numerator and denominator
  virtual void setLexNumDen(WordIndex s, WordIndex t, float num, float den) = 0;

  // Functions to get translations for word
  virtual bool getTransForSource(WordIndex s, std::set<WordIndex>& transSet) const = 0;

  // load function
  virtual bool load(const char* lexNumDenFile, int verbose = 0) = 0;

  // print function
  virtual bool print(const char* lexNumDenFile) const = 0;

  virtual void reserveSpace(WordIndex s) = 0;

  // clear() function
  virtual void clear() = 0;

  // Destructor
  virtual ~_incrLexTable(){};
};

