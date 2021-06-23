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
 * @file PhraseExtractUtils.h
 * @brief Defines string processing utilities
 */

#pragma once

#include "nlp_common/WordAligMatrix.h"
#include "phrase_models/BaseIncrPhraseModel.h"
#include "phrase_models/PhraseDefs.h"
#include "phrase_models/PhraseExtractionTable.h"
#include "phrase_models/PhrasePair.h"
#include "phrase_models/StrictCategPhrasePairFilter.h"
#include "sw_models/BaseSwAligModel.h"

#include <stdio.h>
#include <string>
#include <vector>

namespace PhraseExtractUtils
{
int extractPhrPairsFromCorpusFiles(BaseSwAligModel* swAligModelPtr, BaseSwAligModel* invSwAligModelPtr,
                                   std::string srcCorpusFileName, std::string trgCorpusFileName,
                                   std::vector<std::vector<PhrasePair>>& phrPairs, int verbose = 0);
int extractConsistentPhrasePairs(BaseSwAligModel* swAligModelPtr, BaseSwAligModel* invSwAligModelPtr,
                                 const std::vector<std::string>& srcSentStrVec,
                                 const std::vector<std::string>& refSentStrVec, std::vector<PhrasePair>& vecPhrPair,
                                 bool verbose = 0);
void extractPhrasesFromPairPlusAlig(PhraseExtractParameters phePars, std::vector<std::string> ns,
                                    std::vector<std::string> t, WordAligMatrix waMatrix,
                                    std::vector<PhrasePair>& vecPhrPair, int verbose = 0);
void extractPhrasesFromPairPlusAligBrf(PhraseExtractParameters phePars, std::vector<std::string> ns,
                                       std::vector<std::string> t, WordAligMatrix waMatrix,
                                       std::vector<PhrasePair>& vecPhrPair, int verbose = 0);
void filterPhrasePairs(const std::vector<PhrasePair>& vecUnfiltPhrPair, std::vector<PhrasePair>& vecPhrPair);
} // namespace PhraseExtractUtils

