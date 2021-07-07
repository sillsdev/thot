#pragma once

#include "nlp_common/WordIndex.h"

#ifdef THOT_DISABLE_SPACE_EFFICIENT_LEXDATA_STRUCTURES
#include <unordered_map>
#else
#include "nlp_common/OrderedVector.h"
#endif

#ifdef THOT_DISABLE_SPACE_EFFICIENT_LEXDATA_STRUCTURES
typedef std::unordered_map<WordIndex, std::pair<float, float>> IncrLexCountsElem;
typedef std::vector<IncrLexAuxVarElem> IncrLexCounts;
typedef std::unordered_map<WordIndex, double> LexCountsElem;
typedef std::vector<LexAuxVarElem> LexCounts;
#else
typedef OrderedVector<WordIndex, std::pair<float, float>> IncrLexCountsElem;
typedef std::vector<IncrLexCountsElem> IncrLexCounts;
typedef OrderedVector<WordIndex, double> LexCountsElem;
typedef std::vector<LexCountsElem> LexCounts;
#endif
