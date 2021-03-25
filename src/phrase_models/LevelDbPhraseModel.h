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
 * @brief Defines the LevelDbPhraseModel base class.  LevelDbPhraseModel
 * is derived from the abstract class BasePhraseModel and implements a
 * phrase model stored and accessed using Berkeley databases.
 */

#ifndef _LevelDbPhraseModel_h
#define _LevelDbPhraseModel_h

//--------------- Include files --------------------------------------

#if HAVE_CONFIG_H
#  include <thot_config.h>
#endif /* HAVE_CONFIG_H */

#include "LevelDbPhraseTable.h"
#include "_wbaIncrPhraseModel.h"

//--------------- Constants ------------------------------------------


//--------------- typedefs -------------------------------------------


//--------------- Classes --------------------------------------------


//--------------- LevelDbPhraseModel class

class LevelDbPhraseModel: public _wbaIncrPhraseModel
{
 public:

    typedef _wbaIncrPhraseModel::SrcTableNode SrcTableNode;
    typedef _wbaIncrPhraseModel::TrgTableNode TrgTableNode;

    // Constructor
    LevelDbPhraseModel(void):_wbaIncrPhraseModel()
    {
      basePhraseTablePtr=new LevelDbPhraseTable;
    }

        // Thread/Process safety related functions
    bool modelReadsAreProcessSafe(void);

    
    bool load_ttable(const char* phraseTTableFileName,
                     int verbose=0);
    bool printTTable(const char* outputFileName);

        // destructor
    ~LevelDbPhraseModel();
	
 protected:
    void printTTable(FILE* file);
};

#endif
