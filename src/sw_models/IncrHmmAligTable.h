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
 * @file IncrHmmAligTable.h
 *
 * @brief Defines the IncrHmmAligTable class. IncrHmmAligTable class
 * stores an incremental HMM alignment table.
 *
 */

#pragma once

//--------------- Include files --------------------------------------

#include "nlp_common/AwkInputStream.h"
#include "nlp_common/ErrorDefs.h"
#include "sw_models/aSourceHmm.h"

#include <fstream>
#include <vector>

//--------------- Constants ------------------------------------------

//--------------- typedefs -------------------------------------------

//--------------- function declarations ------------------------------

//--------------- Classes --------------------------------------------

//--------------- IncrHmmAligTable class

class IncrHmmAligTable
{
public:
  // Constructor
  IncrHmmAligTable(void);

  // Functions to handle aligNumer
  void setAligNumer(aSourceHmm asHmm, PositionIndex i, float f);
  float getAligNumer(aSourceHmm asHmm, PositionIndex i, bool& found);

  // Functions to handle aligDenom
  void setAligDenom(aSourceHmm asHmm, float f);
  float getAligDenom(aSourceHmm asHmm, bool& found);

  // Function to set lexical numerator and denominator
  void setAligNumDen(aSourceHmm asHmm, PositionIndex i, float num, float den);

  // load function
  bool load(const char* lexNumDenFile, int verbose = 0);

  // print function
  bool print(const char* lexNumDenFile);

  // clear() function
  void clear(void);

protected:
  // Alignment model types
  typedef std::vector<std::vector<std::pair<bool, float>>> AligNumerElem;
  typedef std::vector<AligNumerElem> AligNumer;
  typedef std::vector<std::vector<std::pair<bool, float>>> AligDenom;

  AligNumer aligNumer;
  AligDenom aligDenom;

  // load and print auxiliary functions
  bool loadBin(const char* lexNumDenFile, int verbose);
  bool loadPlainText(const char* lexNumDenFile, int verbose);
  bool printBin(const char* lexNumDenFile);
  bool printPlainText(const char* lexNumDenFile);
};

