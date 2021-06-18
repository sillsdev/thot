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
 * @file CachedHmmAligLgProb.cc
 *
 * @brief Definitions file for CachedHmmAligLgProb.h
 */

#include "sw_models/CachedHmmAligLgProb.h"

using namespace std;

void CachedHmmAligLgProb::makeRoomGivenSrcSentLen(PositionIndex slen)
{
  PositionIndex nslen = slen * 2;
  if (cachedLgProbs.size() <= nslen)
  {
    vector<vector<double>> lpVecVec;
    cachedLgProbs.resize(nslen + 1, lpVecVec);

    for (unsigned int i = 0; i < cachedLgProbs.size(); ++i)
    {
      if (cachedLgProbs[i].size() <= slen)
      {
        vector<double> lpVec;
        cachedLgProbs[i].resize(slen + 1, lpVec);
      }

      for (unsigned int j = 0; j < cachedLgProbs[i].size(); ++j)
      {
        if (cachedLgProbs[i][j].size() <= j * 2)
          cachedLgProbs[i][j].resize((j * 2) + 1, (double)CACHED_HMM_ALIG_LGPROB_VIT_INVALID_VAL);
      }
    }
  }
}

bool CachedHmmAligLgProb::isDefined(PositionIndex prev_i, PositionIndex slen, PositionIndex i)
{
  if (cachedLgProbs.size() > prev_i && cachedLgProbs[prev_i].size() > slen && cachedLgProbs[prev_i][slen].size() > i)
  {
    if (cachedLgProbs[prev_i][slen][i] >= (double)CACHED_HMM_ALIG_LGPROB_VIT_INVALID_VAL)
      return false;
    else
      return true;
  }
  else
  {
    return false;
  }
}

void CachedHmmAligLgProb::set_boundary_check(PositionIndex prev_i, PositionIndex slen, PositionIndex i, double lp)
{
  // Make room in cachedLgProbs if necessary
  if (cachedLgProbs.size() <= prev_i)
  {
    vector<vector<double>> lpVecVec;
    cachedLgProbs.resize(prev_i + 1, lpVecVec);
  }

  if (cachedLgProbs[prev_i].size() <= slen)
  {
    vector<double> lpVec;
    cachedLgProbs[prev_i].resize(slen + 1, lpVec);
  }

  if (cachedLgProbs[prev_i][slen].size() <= i)
    cachedLgProbs[prev_i][slen].resize(i + 1, (double)CACHED_HMM_ALIG_LGPROB_VIT_INVALID_VAL);

  // Set value
  cachedLgProbs[prev_i][slen][i] = lp;
}

void CachedHmmAligLgProb::set(PositionIndex prev_i, PositionIndex slen, PositionIndex i, double lp)
{
  cachedLgProbs[prev_i][slen][i] = lp;
}

double CachedHmmAligLgProb::get(PositionIndex prev_i, PositionIndex slen, PositionIndex i)
{
  return cachedLgProbs[prev_i][slen][i];
}

void CachedHmmAligLgProb::clear()
{
  cachedLgProbs.clear();
}
