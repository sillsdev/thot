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
 * @file BaseStepwiseAligModel.h
 *
 * @brief Defines the BaseStepwiseAligModel class.
 * BaseStepwiseAligModel is a base class for derivating single-word
 * statistical alignment models using stepwise EM.
 */

#pragma once

//--------------- Include files --------------------------------------

#include "nlp_common/ErrorDefs.h"
#include "sw_models/SwDefs.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <string>

//--------------- Constants ------------------------------------------

//--------------- typedefs -------------------------------------------

//--------------- Function declarations ------------------------------

//--------------- Classes --------------------------------------------

//--------------- BaseStepwiseAligModel class

class BaseStepwiseAligModel
{
public:
  // Constructor
  BaseStepwiseAligModel(void){};

  virtual void set_nu_val(float _nu) = 0;
  // Function to set the value of alpha

  // Destructor
  virtual ~BaseStepwiseAligModel(){};
};

