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
 * @file PhraseDefs.h
 *
 * @brief Constants, typedefs and basic classes used in the phrase-model
 * classes.
 */

#pragma once

//--------------- Include files --------------------------------------

#include "phrase_models/PhraseId.h"
#include "phrase_models/PhraseSortCriterion.h"
#include "phrase_models/SentSegmentation.h"
#include "phrase_models/VecUnsignedIntSortCriterion.h"

//--------------- Constants ------------------------------------------

#define PHRASE_PROB_SMOOTH 1e-10
#define LOG_PHRASE_PROB_SMOOTH log(PHRASE_PROB_SMOOTH)
#define SEGM_SIZE_PROB_SMOOTH 1e-7
#define MAX_SENTENCE_LENGTH 201

//--------------- typedefs -------------------------------------------

typedef off_t PhrIndex;

