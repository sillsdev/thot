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
 * @file IncrFertilityTable.h
 *
 * @brief Defines the IncrFertilityTable class.  IncrFertilityTable class
 * stores an incremental fertility table.
 */

#ifndef _IncrFertilityTable_h
#define _IncrFertilityTable_h

#if HAVE_CONFIG_H
#  include <thot_config.h>
#endif /* HAVE_CONFIG_H */

#include <ErrorDefs.h>
#include <fstream>
#include <AwkInputStream.h>
#include <StatModelDefs.h>
#include <vector>

class IncrFertilityTable
{
public:

  // Constructor and destructor
  IncrFertilityTable();
  ~IncrFertilityTable();

  // Functions to handle numerator
  void setFertilityNumer(WordIndex s, PositionIndex phi, float f);
  float getFertilityNumer(WordIndex s, PositionIndex phi, bool& found) const;

  // Functions to handle denominator
  void setFertilityDenom(WordIndex s, float f);
  float getFertilityDenom(WordIndex s, bool& found) const;

  // Function to set numerator and denominator
  void setFertilityNumDen(WordIndex s, PositionIndex phi, float num, float den);

  // load function
  bool load(const char* fertilityNumDenFile, int verbose = 0);

  // print function
  bool print(const char* fertilityNumDenFile) const;

  void reserveSpace(WordIndex s);

  // clear() function
  void clear();

protected:
 
  typedef std::vector<float> FertilityNumerElem;
  typedef std::vector<FertilityNumerElem> FertilityNumer;
  typedef std::vector<float> FertilityDenom;

  FertilityNumer fertilityNumer;
  FertilityDenom fertilityDenom;

  // load and print auxiliary functions
  bool loadBin(const char* fertilityNumDenFile, int verbose);
  bool loadPlainText(const char* fertilityNumDenFile, int verbose);
  bool printBin(const char* fertilityNumDenFile) const;
  bool printPlainText(const char* fertilityNumDenFile) const;
};

#endif

