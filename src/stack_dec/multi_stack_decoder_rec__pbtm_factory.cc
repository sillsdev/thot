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
 
/********************************************************************/
/*                                                                  */
/* Module: multi_stack_decoder_rec__pbtm_factory                    */
/*                                                                  */
/* Definitions file: multi_stack_decoder_rec__pbtm_factory.cc       */
/*                                                                  */
/********************************************************************/


//--------------- Include files --------------------------------------

#include "PhrHypNumcovJumps01EqClassF.h"
#include "PbTransModel.h"
#include "multi_stack_decoder_rec.h"
#include <string>

//--------------- Function definitions

extern "C" BaseStackDecoder<PbTransModel<PhrHypNumcovJumps01EqClassF> >* create(const char* /*str*/)
{
  return new multi_stack_decoder_rec<PbTransModel<PhrHypNumcovJumps01EqClassF> >;
}

//---------------
extern "C" const char* type_id(void)
{
  return "multi_stack_decoder_rec<PbTransModel>";
}
