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
 * @file IncrIbm2AligTable.h
 *
 * @brief Defines the IncrIbm2AligTable class.  IncrIbm2AligTable class
 * stores an incremental IBM 2 alignment table.
 */

#pragma once

#include "nlp_common/AwkInputStream.h"
#include "nlp_common/ErrorDefs.h"
#include "nlp_common/StatModelDefs.h"
#include "sw_models/aSource.h"
#include "sw_models/aSourceHashF.h"

#include <fstream>
#include <unordered_map>
#include <vector>
class IncrIbm2AligTable
{
public:
  // Constructor
  IncrIbm2AligTable();

  // Functions to handle aligNumer
  void setAligNumer(aSource as, PositionIndex i, float f);
  float getAligNumer(aSource as, PositionIndex i, bool& found) const;

  // Functions to handle aligDenom
  void setAligDenom(aSource as, float f);
  float getAligDenom(aSource as, bool& found) const;

  // Function to set numerator and denominator
  void setAligNumDen(aSource as, PositionIndex i, float num, float den);

  void reserveSpace(aSource as);

  // load function
  bool load(const char* aligNumDenFile, int verbose = 0);

  // print function
  bool print(const char* aligNumDenFile) const;

  // clear() function
  void clear();

protected:
  // Alignment model types
  typedef std::vector<float> AligNumerElem;
  typedef std::unordered_map<aSource, AligNumerElem, aSourceHashF> AligNumer;
  typedef std::unordered_map<aSource, float, aSourceHashF> AligDenom;

  AligNumer aligNumer;
  AligDenom aligDenom;

  // load and print auxiliary functions
  bool loadBin(const char* aligNumDenFile, int verbose);
  bool loadPlainText(const char* aligNumDenFile, int verbose);
  bool printBin(const char* aligNumDenFile) const;
  bool printPlainText(const char* aligNumDenFile) const;
};

