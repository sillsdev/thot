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

#ifndef _LexAuxVar_h
#define _LexAuxVar_h

#include "SwDefs.h"

#ifdef THOT_DISABLE_SPACE_EFFICIENT_LEXDATA_STRUCTURES
#include <unordered_map>
#else
#include <OrderedVector.h>
#endif

#ifdef THOT_DISABLE_SPACE_EFFICIENT_LEXDATA_STRUCTURES
typedef std::unordered_map<WordIndex, std::pair<float, float>> IncrLexAuxVarElem;
typedef std::vector<IncrLexAuxVarElem> IncrLexAuxVar;
typedef std::unordered_map<WordIndex, double> LexAuxVarElem;
typedef std::vector<LexAuxVarElem> LexAuxVar;
#else
typedef OrderedVector<WordIndex, std::pair<float, float>> IncrLexAuxVarElem;
typedef std::vector<IncrLexAuxVarElem> IncrLexAuxVar;
typedef OrderedVector<WordIndex, double> LexAuxVarElem;
typedef std::vector<LexAuxVarElem> LexAuxVar;
#endif

#endif
