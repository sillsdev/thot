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
 * @file IncrDistortionTable.h
 *
 * @brief Defines the IncrDistortionTable class.  IncrDistortionTable class
 * stores an incremental distortion table.
 */

#ifndef _IncrDistortionTable_h
#define _IncrDistortionTable_h

#if HAVE_CONFIG_H
#  include <thot_config.h>
#endif /* HAVE_CONFIG_H */

#include <ErrorDefs.h>
#include <fstream>
#include <AwkInputStream.h>
#include <StatModelDefs.h>
#include "dSource.h"
#include "dSourceHashF.h"
#include <vector>

#if __GNUC__>2
#include <ext/hash_map>
using __gnu_cxx::hash_map;
#else
#include <hash_map>
using stdext::hash_map;
#endif

class IncrDistortionTable
{
public:
  // Constructor
  IncrDistortionTable();

  // Functions to handle numerator
  void setDistortionNumer(dSource ds, PositionIndex j, float f);
  float getDistortionNumer(dSource ds, PositionIndex j, bool& found);

  // Functions to handle denominator
  void setDistortionDenom(dSource ds, float f);
  float getDistortionDenom(dSource ds, bool& found);

  // Function to set lexical numerator and denominator
  void setDistortionNumDen(dSource ds, PositionIndex j, float num, float den);

  // load function
  bool load(const char* distortionNumDenFile, int verbose = 0);

  // print function
  bool print(const char* distortionNumDenFile);

  // clear() function
  void clear();

protected:

  // Alignment model types
  typedef hash_map<dSource, float, dSourceHashF> DistortionNumerElem;
  typedef std::vector<DistortionNumerElem> DistortionNumer;
  typedef hash_map<dSource, float, dSourceHashF> DistortionDenom;

  DistortionNumer distortionNumer;
  DistortionDenom distortionDenom;

  // load and print auxiliary functions
  bool loadBin(const char* distortionNumDenFile, int verbose);
  bool loadPlainText(const char* distortionNumDenFile, int verbose);
  bool printBin(const char* distortionNumDenFile);
  bool printPlainText(const char* distortionNumDenFile);
};

#endif
