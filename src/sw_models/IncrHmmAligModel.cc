/*
thot package for statistical machine translation
Copyright (C) 2013-2017 Daniel Ortiz-Mart\'inez, Adam Harasimowicz

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
 * @file IncrHmmAligModel.cc
 *
 * @brief Definitions file for IncrHmmAligModel.h
 */

//--------------- Include files --------------------------------------

#include "sw_models/IncrHmmAligModel.h"

#include "sw_models/MemoryLexTable.h"

//--------------- IncrHmmAligModel class function definitions

//-------------------------
IncrHmmAligModel::IncrHmmAligModel() : _incrHmmAligModel()
{
  // Create table with lexical parameters
  lexTable = new MemoryLexTable();
  lexNumDenFileExtension = ".hmm_lexnd";
}

void IncrHmmAligModel::clearSentLengthModel(void)
{
  sentLengthModel.clear();
}
