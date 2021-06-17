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
 * @file CachedHmmAligLgProb.h
 *
 * @brief Manages an IBM translation table.
 *
 */

#ifndef _CachedHmmAligLgProb
#define _CachedHmmAligLgProb

#include "SwDefs.h"

#define CACHED_HMM_ALIG_LGPROB_VIT_INVALID_VAL 99

class CachedHmmAligLgProb
{
public:
  void makeRoomGivenSrcSentLen(PositionIndex slen);
  bool isDefined(PositionIndex prev_i, PositionIndex slen, PositionIndex i);
  void set_boundary_check(PositionIndex prev_i, PositionIndex slen, PositionIndex i, double lp);
  void set(PositionIndex prev_i, PositionIndex slen, PositionIndex i, double lp);
  double get(PositionIndex prev_i, PositionIndex slen, PositionIndex i);
  void clear();

private:
  std::vector<std::vector<std::vector<double>>> cachedLgProbs;
};

#endif
