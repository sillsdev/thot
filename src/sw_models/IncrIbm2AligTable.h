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
 * @file IncrIbm2AligTable.h
 * 
 * @brief Defines the IncrIbm2AligTable class.  IncrIbm2AligTable class
 * stores an incremental IBM 2 alignment table.
 */

#ifndef _IncrIbm2AligTable_h
#define _IncrIbm2AligTable_h

//--------------- Include files --------------------------------------

#include <ErrorDefs.h>
#include <fstream>
#include <AwkInputStream.h>
#include <StatModelDefs.h>
#include "aSource.h"
#include "aSourceHashF.h"
#include <vector>
#include <unordered_map>

//--------------- Constants ------------------------------------------


//--------------- typedefs -------------------------------------------


//--------------- function declarations ------------------------------

//--------------- Classes --------------------------------------------

//--------------- IncrIbm2AligTable class

class IncrIbm2AligTable
{
  public:

       // Constructor
   IncrIbm2AligTable(void);   

       // Functions to handle aligNumer
   void setAligNumer(aSource as,
                     PositionIndex i,
                     float f);
   float getAligNumer(aSource as,
                      PositionIndex i,
                      bool& found);
   
   // Functions to handle aligDenom
   void setAligDenom(aSource as,
                     float f);
   float getAligDenom(aSource as,
                      bool& found);

   // Function to set lexical numerator and denominator
   void setAligNumDen(aSource as,
                      PositionIndex i,
                      float num,
                      float den);

       // load function
   bool load(const char* lexNumDenFile,
             int verbose=0);
   
       // print function
   bool print(const char* lexNumDenFile);

       // clear() function
   void clear(void);
   
  protected:

       // Alignment model types
   typedef std::unordered_map<aSource,float,aSourceHashF> AligNumerElem;
   typedef std::vector<AligNumerElem> AligNumer;
   typedef std::unordered_map<aSource,float,aSourceHashF> AligDenom;

   AligNumer aligNumer;
   AligDenom aligDenom;

       // load and print auxiliary functions
   bool loadBin(const char* lexNumDenFile,
                int verbose);
   bool loadPlainText(const char* lexNumDenFile,
                      int verbose);
   bool printBin(const char* lexNumDenFile);
   bool printPlainText(const char* lexNumDenFile);
};

#endif
