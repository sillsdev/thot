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
 * @file _incrSwAligModel.h
 *
 * @brief Defines the _incrSwAligModel class.  _incrSwAligModel is a
 * predecessor class for derivating single-word incremental statistical
 * alignment models.
 *
 */

#pragma once

#include "sw_models/_swAligModel.h"

class _incrSwAligModel : public _swAligModel
{
public:
  virtual void set_expval_maxnsize(unsigned int _anji_maxnsize) = 0;
  // Function to set a maximum size for the vector of expected
  // values anji (by default the size is not restricted)

  virtual void efficientBatchTrainingForRange(std::pair<unsigned int, unsigned int> sentPairRange,
                                              int verbosity = 0) = 0;
  virtual void efficientBatchTrainingForAllSents(int verbosity = 0);
};

