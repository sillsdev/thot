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

#ifndef _dSourceHashF_h
#define _dSourceHashF_h

#include "sw_models/dSource.h"

struct dSourceHashF
{
public:
  std::size_t operator()(const dSource& a1) const
  {
    return (size_t)(16384 * a1.i) + ((size_t)256 * a1.slen) + a1.tlen;
  }
};

#endif
