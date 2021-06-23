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
 * @file WeightUpdateUtils.h
 * @brief Defines string processing utilities
 */

#pragma once

extern "C"
{
#include "downhill_simplex/step_by_step_dhs.h"
}

#include "error_correction/WordGraph.h"
#include "nlp_common/BaseNgramLM.h"
#include "phrase_models/BasePhraseModel.h"
#include "phrase_models/PhraseExtractUtils.h"
#include "phrase_models/PhrasePair.h"
#include "stack_dec/BaseLogLinWeightUpdater.h"
#include "stack_dec/DirectPhraseModelFeat.h"
#include "stack_dec/InversePhraseModelFeat.h"
#include "stack_dec/LM_State.h"
#include "sw_models/BaseSwAligModel.h"

#include <stdio.h>
#include <string>
#include <vector>

//--------------- Constants ------------------------------------------

#define NBLIST_SIZE_FOR_LLWEIGHT_UPDATE 1000
#define PHRSWLITM_DHS_FTOL 0.001
#define PHRSWLITM_DHS_SCALE_PAR 1

namespace WeightUpdateUtils
{
void updateLogLinearWeights(std::string refSent, WordGraph* wgPtr, BaseLogLinWeightUpdater* llWeightUpdaterPtr,
                            const std::vector<std::pair<std::string, float>>& compWeights,
                            std::vector<float>& newWeights, int verbose = 0);
template <class THypScoreInfo>
int updatePmLinInterpWeights(std::string srcCorpusFileName, std::string trgCorpusFileName,
                             DirectPhraseModelFeat<THypScoreInfo>* dirPhrModelFeatPtr,
                             InversePhraseModelFeat<THypScoreInfo>* invPhrModelFeatPtr, int verbose = 0);
} // namespace WeightUpdateUtils

