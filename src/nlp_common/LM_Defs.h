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
 * @file LM_Defs.h
 *
 * @brief Definitions related to n-gram language models.
 */

#pragma once

//--------------- Include files --------------------------------------

#include "nlp_common/Count.h"
#include "nlp_common/Prob.h"
#include "nlp_common/WordIndex.h"
#include "nlp_common/lt_op_vec.h" // provides an ordering relationship for vectors

#include <vector>

//--------------- Constants ------------------------------------------

#define UNK_SYMBOL 0
#define S_BEGIN 1
#define S_END 2
#define SP_SYM1_LM 3
#define UNK_SYMBOL_STR "<unk>"
#define BOS_STR "<s>"
#define EOS_STR "</s>"
#define SP_SYM1_LM_STR "<sp_sym1>"
#define LM_PROB_SMOOTH 1e-10

//--------------- User defined types ---------------------------------

typedef WordIndex ngramWordIndex;
typedef unsigned int NgIdx;

