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
 * @file LevelDbPhraseModel.cc
 * 
 * @brief Definitions file for LevelDbPhraseModel.h
 */

//--------------- Include files --------------------------------------

#include "LevelDbPhraseModel.h"

//--------------- Global variables -----------------------------------


//--------------- LevelDbPhraseModel class function definitions

//------------------------------
bool LevelDbPhraseModel::modelReadsAreProcessSafe(void)
{
      // Reads are not process safe in LevelDB based models since the
      // first process opening LevelDB database locks it (LevelDB reads
      // are thread-safe but not process-safe)
  return false;
}

//-------------------------
bool LevelDbPhraseModel::load_ttable(const char* _incrPhraseModelFileName,
                                     int verbose/*=0*/)
{
  LevelDbPhraseTable* phraseTablePtr = static_cast<LevelDbPhraseTable*>(this->basePhraseTablePtr);
  return phraseTablePtr->load(_incrPhraseModelFileName);
}

//-------------------------
bool LevelDbPhraseModel::printTTable(const char* outputFileName)
{
  return THOT_OK;
}

void LevelDbPhraseModel::printTTable(FILE* file)
{
}

//-------------------------
LevelDbPhraseModel::~LevelDbPhraseModel()
{
  delete basePhraseTablePtr;
}

//-------------------------
