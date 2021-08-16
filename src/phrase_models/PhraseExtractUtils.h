
#pragma once

#include "nlp_common/WordAlignmentMatrix.h"
#include "phrase_models/BaseIncrPhraseModel.h"
#include "phrase_models/PhraseDefs.h"
#include "phrase_models/PhraseExtractionTable.h"
#include "phrase_models/PhrasePair.h"
#include "phrase_models/StrictCategPhrasePairFilter.h"
#include "sw_models/AlignmentModel.h"

#include <stdio.h>
#include <string>
#include <vector>

namespace PhraseExtractUtils
{
int extractPhrPairsFromCorpusFiles(AlignmentModel* swAligModelPtr, AlignmentModel* invSwAligModelPtr,
                                   std::string srcCorpusFileName, std::string trgCorpusFileName,
                                   std::vector<std::vector<PhrasePair>>& phrPairs, int verbose = 0);
int extractConsistentPhrasePairs(AlignmentModel* swAligModelPtr, AlignmentModel* invSwAligModelPtr,
                                 const std::vector<std::string>& srcSentStrVec,
                                 const std::vector<std::string>& refSentStrVec, std::vector<PhrasePair>& vecPhrPair,
                                 bool verbose = 0);
void extractPhrasesFromPairPlusAlig(PhraseExtractParameters phePars, std::vector<std::string> ns,
                                    std::vector<std::string> t, WordAlignmentMatrix waMatrix,
                                    std::vector<PhrasePair>& vecPhrPair, int verbose = 0);
void extractPhrasesFromPairPlusAligBrf(PhraseExtractParameters phePars, std::vector<std::string> ns,
                                       std::vector<std::string> t, WordAlignmentMatrix waMatrix,
                                       std::vector<PhrasePair>& vecPhrPair, int verbose = 0);
void filterPhrasePairs(const std::vector<PhrasePair>& vecUnfiltPhrPair, std::vector<PhrasePair>& vecPhrPair);
} // namespace PhraseExtractUtils
