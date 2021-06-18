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

#include "sw_models/aSource.h"

struct aSourceHashF
{
public:
  std::size_t operator()(const aSource& a1) const
  {
    return (size_t)(16384 * a1.j) + ((size_t)256 * a1.slen) + a1.tlen;
  }
};

