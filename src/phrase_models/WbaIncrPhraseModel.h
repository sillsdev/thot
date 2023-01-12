/*
thot package for statistical machine translation
Copyright (C) 2013-2017 Daniel Ortiz-Mart\'inez, Adam Harasimowicz, and SIL International

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

#include "phrase_models/PhraseExtractionTable.h"
#include "phrase_models/_wbaIncrPhraseModel.h"

#ifdef THOT_USE_HAT_TRIE_PHRASE_TABLE
#include "phrase_models/HatTriePhraseTable.h"
#else
#include "phrase_models/StlPhraseTable.h"
#endif

#define VERBOSE_AACHEN -1

/**
 * Defines the WbaIncrPhraseModel class.  WbaIncrPhraseModel
 * implements a phrase model which use word-based alignments (as those
 * obtained with the GIZA++ tool).
 */
class WbaIncrPhraseModel : public _wbaIncrPhraseModel
{
public:
  typedef _wbaIncrPhraseModel::SrcTableNode SrcTableNode;
  typedef _wbaIncrPhraseModel::TrgTableNode TrgTableNode;

  // Constructor
  WbaIncrPhraseModel() : _wbaIncrPhraseModel()
  {
#ifdef THOT_USE_HAT_TRIE_PHRASE_TABLE
    basePhraseTablePtr = new HatTriePhraseTable;
#else
    basePhraseTablePtr = new StlPhraseTable;
#endif
  }

  bool printPhraseTable(const char* outputFileName, int n = -1) override;

  // Destructor
  ~WbaIncrPhraseModel();
};
