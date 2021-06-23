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

#pragma once

//--------------- Include files --------------------------------------

#include "phrase_models/BpSet.h"

#include <float.h>

//--------------- Constants ------------------------------------------
#define UNINIT_NUM_LEAFS FLT_MAX

//--------------- typedefs and structs  ------------------------------

//--------------- Classes

class SrfNodeInfo
{
public:
  SrfNodeInfo(void)
  {
    estLogNumLeafs = UNINIT_NUM_LEAFS;
  }

  double estLogNumLeafs;
  BpSet C;

private:
};

