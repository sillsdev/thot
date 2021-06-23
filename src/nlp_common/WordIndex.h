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

#include <limits.h>
#include <math.h>

#define NULL_WORD 0
#define NULL_WORD_STR "NULL"
#define UNK_WORD 1
#define UNK_WORD_STR "UNKNOWN_WORD"
#define UNUSED_WORD 2
#define UNUSED_WORD_STR "<UNUSED_WORD>"

#ifdef USHORT_WORDINDEX
typedef unsigned short WordIndex;
const unsigned int MAX_VOCAB_SIZE = USHRT_MAX;
#else
typedef unsigned int WordIndex;
const unsigned int MAX_VOCAB_SIZE = UINT_MAX;
#endif

