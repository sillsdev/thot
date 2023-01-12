/*
thot package for statistical machine translation
Copyright (C) 2013 Daniel Ortiz-Mart\'inez and SIL International

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

#pragma once

#include "error_correction/HypStateIndex.h"
#include "nlp_common/PositionIndex.h"
#include "nlp_common/Score.h"

#include <string>
#include <vector>

class WordGraphArc
{
public:
  HypStateIndex predStateIndex;
  HypStateIndex succStateIndex;
  Score arcScore;
  std::vector<std::string> words;
  PositionIndex srcStartIndex;
  PositionIndex srcEndIndex;
  bool unknown;
};
