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
 * @file IncrIbm2AligModel.h
 *
 * @brief Defines the IncrIbm2AligModel class. IncrIbm2AligModel class
 * allows to generate and access to the data of an IBM 2 statistical
 * alignment model.
 */

#pragma once

#if HAVE_CONFIG_H
#  include <thot_config.h>
#endif /* HAVE_CONFIG_H */

#include "_incrSwAligModel.h"
#include "Ibm2AligModel.h"
#include "IncrIbm2AligTrainer.h"
#include "anjiMatrix.h"

class IncrIbm2AligModel : public Ibm2AligModel, public virtual _incrSwAligModel
{
public:
  // Constructor
  IncrIbm2AligModel();

  void set_expval_maxnsize(unsigned int _anji_maxnsize);
  // Function to set a maximum size for the vector of expected
  // values anji (by default the size is not restricted)

// Functions to train model
  void incrTrainSentPairRange(std::pair<unsigned int, unsigned int> sentPairRange, int verbosity = 0);
  void incrTrainAllSents(int verbosity = 0);

  bool load(const char* prefFileName, int verbose = 0);
  bool print(const char* prefFileName, int verbose = 0);

  void clearInfoAboutSentRange();
  void clear();
  void clearTempVars();

  // Destructor
  ~IncrIbm2AligModel();

protected:
  anjiMatrix anji;
  IncrIbm2AligTrainer trainer;
};

