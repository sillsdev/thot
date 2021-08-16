#pragma once

#include "nlp_common/BaseNgramLM.h"
#include "phrase_models/BasePhraseModel.h"
#include "stack_dec/LM_State.h"
#include "sw_models/AlignmentModel.h"

#include <stdio.h>
#include <string>
#include <vector>

namespace SmtModelUtils
{
int loadPhrModel(BasePhraseModel* basePhraseModelPtr, std::string modelFileName);
int printPhrModel(BasePhraseModel* basePhraseModelPtr, std::string modelFileName);
int loadDirectSwModel(AlignmentModel* baseSwAligModelPtr, std::string modelFileName);
int printDirectSwModel(AlignmentModel* baseSwAligModelPtr, std::string modelFileName);
int loadInverseSwModel(AlignmentModel* baseSwAligModelPtr, std::string modelFileName);
int printInverseSwModel(AlignmentModel* baseSwAligModelPtr, std::string modelFileName);
int loadLangModel(BaseNgramLM<LM_State>* baseNgLmPtr, std::string modelFileName);
int printLangModel(BaseNgramLM<LM_State>* baseNgLmPtr, std::string modelFileName);
bool loadSwmLambdas(std::string lambdaFileName, float& lambda_swm, float& lambda_invswm);
bool printSwmLambdas(const char* lambdaFileName, float lambda_swm, float lambda_invswm);
} // namespace SmtModelUtils
