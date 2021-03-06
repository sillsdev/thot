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
 * @file NbestTransCacheData.h
 *
 * @brief Class for caching information related to translation model.
 */

#pragma once

//--------------- Include files --------------------------------------

#include "stack_dec/PhrNbestTransTable.h"
#include "stack_dec/PhrNbestTransTablePref.h"
#include "stack_dec/PhrNbestTransTableRef.h"
#include "stack_dec/PhraseCacheTable.h"
#include "stack_dec/PhrasePairCacheTable.h"

//--------------- Classes --------------------------------------------

class NbestTransCacheData
{
public:
  // Cached n-best lm scores
  PhraseCacheTable cnbLmScores;

  // Cached translation table to store phrase N-best translations
  PhrNbestTransTable cPhrNbestTransTable;
  // The same as cPhrNbestTransTable but to be used in assisted
  // translation
  PhrNbestTransTableRef cPhrNbestTransTableRef;
  PhrNbestTransTablePref cPhrNbestTransTablePref;

  // Cached n-best translations scores (these cached scores are
  // those generated by the nbestTransScore() and
  // nbestTransScoreLast() functions)
  PhrasePairCacheTable cnbestTransScore;
  PhrasePairCacheTable cnbestTransScoreLast;

  // Function to clear cached data
  void clear(void)
  {
    cnbLmScores.clear();
    cPhrNbestTransTable.clear();
    cPhrNbestTransTableRef.clear();
    cPhrNbestTransTablePref.clear();
    cnbestTransScore.clear();
    cnbestTransScoreLast.clear();
  };
};

