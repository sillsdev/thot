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
 * @file SmtModelUtils.h
 * @brief Defines string processing utilities
 */

#ifndef _SmtModelUtils_h
#define _SmtModelUtils_h

#include "BaseNgramLM.h"
#include "BasePhraseModel.h"
#include "BaseSwAligModel.h"
#include "LM_State.h"

#include <stdio.h>
#include <string>
#include <vector>

namespace SmtModelUtils
{
int loadPhrModel(BasePhraseModel* basePhraseModelPtr, std::string modelFileName);
int printPhrModel(BasePhraseModel* basePhraseModelPtr, std::string modelFileName);
int loadDirectSwModel(BaseSwAligModel* baseSwAligModelPtr, std::string modelFileName);
int printDirectSwModel(BaseSwAligModel* baseSwAligModelPtr, std::string modelFileName);
int loadInverseSwModel(BaseSwAligModel* baseSwAligModelPtr, std::string modelFileName);
int printInverseSwModel(BaseSwAligModel* baseSwAligModelPtr, std::string modelFileName);
int loadLangModel(BaseNgramLM<LM_State>* baseNgLmPtr, std::string modelFileName);
int printLangModel(BaseNgramLM<LM_State>* baseNgLmPtr, std::string modelFileName);
bool loadSwmLambdas(std::string lambdaFileName, float& lambda_swm, float& lambda_invswm);
bool printSwmLambdas(const char* lambdaFileName, float lambda_swm, float lambda_invswm);
} // namespace SmtModelUtils

#endif
