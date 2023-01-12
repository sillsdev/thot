/*
thot package for statistical machine translation
Copyright (C) 2013 Daniel Ortiz-Mart\'inez and SIL International

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

#pragma once

#include "stack_dec/BaseScorer.h"

#include <string>
#include <utility>
#include <vector>

/**
 * @brief Base abstract class that defines the interface that a
 * log-linear weight updater algorithm should offer to a statistical
 * machine translation model.
 */

class BaseLogLinWeightUpdater
{
public:
  // Function to link scorer
  virtual bool setScorer(BaseScorer* baseScorerPtr) = 0;

  // Function to compute new weights
  virtual void update(const std::string& reference, const std::vector<std::string>& nblist,
                      const std::vector<std::vector<double>>& scoreCompsVec, const std::vector<double>& currWeightsVec,
                      std::vector<double>& newWeightsVec) = 0;

  // Compute new weights for a closed corpus
  virtual void updateClosedCorpus(const std::vector<std::string>& reference,
                                  const std::vector<std::vector<std::string>>& nblist,
                                  const std::vector<std::vector<std::vector<double>>>& scoreCompsVec,
                                  const std::vector<double>& currWeightsVec, std::vector<double>& newWeightsVec) = 0;

  // Destructor
  virtual ~BaseLogLinWeightUpdater(){};
};
