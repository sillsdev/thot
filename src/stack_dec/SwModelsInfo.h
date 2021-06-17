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

#ifndef _SwModelsInfo_h
#define _SwModelsInfo_h

//--------------- Include files --------------------------------------

#include "BaseSwAligModel.h"
#include "SimpleDynClassLoader.h"

#include <vector>

//--------------- Constants ------------------------------------------

//--------------- typedefs -------------------------------------------

//--------------- SwModelsInfo struct

struct SwModelsInfo
{
  // sw model members
  std::vector<BaseSwAligModel*> swAligModelPtrVec;
  std::vector<std::string> featNameVec;

  // Inverse sw model members
  std::vector<BaseSwAligModel*> invSwAligModelPtrVec;
  std::vector<std::string> invFeatNameVec;

  SimpleDynClassLoader<BaseSwAligModel> defaultClassLoader;
};

#endif
