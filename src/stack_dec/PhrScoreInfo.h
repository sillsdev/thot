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
 * @file PhrScoreInfo.h
 *
 * @brief Class to store score information about phrase-based model
 * hypotheses.
 */

#pragma once

//--------------- Include files --------------------------------------

#include "nlp_common/Score.h"
#include "stack_dec/LM_State.h"

//--------------- Classes --------------------------------------------

class PhrScoreInfo
{
public:
  Score score{};

  // Language model info
  LM_State lmHist;

  Score getScore(void) const;
  void addHeuristic(Score h);
  void subtractHeuristic(Score h);
};
