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
 * @file StatModelDefs.h
 *
 * @brief Constants, typedefs and basic classes for statistical models.
 */

#ifndef _StatModelDefs_h
#define _StatModelDefs_h

//--------------- Include files --------------------------------------

#include "Count.h"
#include "LogCount.h"
#include "PositionIndex.h"
#include "Prob.h"
#include "Score.h"
#include "WordIndex.h"

//--------------- Constants ------------------------------------------

#define NULL_WORD 0
#define NULL_WORD_STR "NULL"
#define UNK_WORD 1
#define UNK_WORD_STR "UNKNOWN_WORD"
#define UNUSED_WORD 2
#define UNUSED_WORD_STR "<UNUSED_WORD>"

//--------------- typedefs -------------------------------------------

typedef int ClassIndex;

//---------------

#endif
