/*
thot package for statistical machine translation
Copyright (C) 2013 Daniel Ortiz-Mart\'inez and SIL International

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

#include "nlp_common/Bitset.h"
#include "nlp_common/PositionIndex.h"
#include "nlp_common/SmtDefs.h"
#include "stack_dec/BaseHypState.h"
#include "stack_dec/LM_State.h"

/**
 * @brief The PhrHypState class represents the state of a hypothesis for
 * phrase-based translation.
 */
class PhrHypState : public BaseHypState
{
public:
  // Language model info
  LM_State lmHist;

  // Target length
  unsigned int trglen{};

  // End position of the last covered source phrase
  PositionIndex endLastSrcPhrase{};

  // Coverage info
  Bitset<MAX_SENTENCE_LENGTH_ALLOWED> sourceWordsAligned;

  // Ordering
  bool operator<(const PhrHypState& right) const;
};
