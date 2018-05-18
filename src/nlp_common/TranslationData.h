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

#ifndef _TranslationData_h
#define _TranslationData_h

//--------------- Include files --------------------------------------

#if HAVE_CONFIG_H
#  include <thot_config.h>
#endif /* HAVE_CONFIG_H */

#include "PositionIndex.h"
#include <vector>
#include "Score.h"
#include <set>

//--------------- Classes --------------------------------------------

class TranslationData
{
public:
  std::vector<std::string> target;

  std::vector<std::pair<PositionIndex, PositionIndex> > sourceSegmentation;
  std::vector<PositionIndex> targetSegmentCuts;
  std::set<PositionIndex> targetUnknownWords;

  Score score;
  std::vector<Score> scoreComponents;
};

#endif