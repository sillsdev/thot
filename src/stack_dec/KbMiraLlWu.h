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

#include "stack_dec/BaseLogLinWeightUpdater.h"
#include "stack_dec/BaseMiraScorer.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <float.h>
#include <iostream>
#include <iterator>
#include <memory>

#define KBMIRA_RANDOM_SEED 31415

struct HopeFearData
{
  std::vector<double> hopeFeatures, fearFeatures;
  std::vector<unsigned int> hopeQualityStats;
  double hopeScore, hopeQuality;
  double fearScore, fearQuality;
};

/**
 * @brief Class implementing the K-best MIRA algorithm.
 */
class KbMiraLlWu : public BaseLogLinWeightUpdater
{
public:
  KbMiraLlWu(double C = 0.01, double gamma = 0.999, unsigned int J = 60, unsigned int epochs_to_restart = 20,
             unsigned int max_restarts = 1);
  ~KbMiraLlWu();

  // Function to link scorer
  bool setScorer(BaseScorer* baseScorerPtr);

  // Compute new weights for an individual sentence
  void update(const std::string& reference, const std::vector<std::string>& nblist,
              const std::vector<std::vector<double>>& scoreCompsVec, const std::vector<double>& currWeightsVec,
              std::vector<double>& newWeightsVec);

  // Compute new weights for a closed corpus
  void updateClosedCorpus(const std::vector<std::string>& references,
                          const std::vector<std::vector<std::string>>& nblists,
                          const std::vector<std::vector<std::vector<double>>>& scoreCompsVecs,
                          const std::vector<double>& currWeightsVec, std::vector<double>& newWeightsVec);

private:
  double c;                     // Step-size cap C
  double decay;                 // Pseudo-corpus decay \gamma
  unsigned int nIters;          // Max epochs J
  unsigned int epochsToRestart; // epochs without improvement before re-start
  unsigned int maxRestarts;     // max number of re-starts
  std::unique_ptr<BaseMiraScorer> scorer{};

  // Compute max scoring translaiton according to w
  void MaxTranslation(const std::vector<double>& w, const std::vector<std::string>& nBest,
                      const std::vector<std::vector<double>>& nScores, std::string& maxTranslation);

  // Compute hope/fear translations and stores info in hopeFear
  void HopeFear(const std::string& reference, const std::vector<std::string>& nBest,
                const std::vector<std::vector<double>>& nScores, const std::vector<double>& wv, HopeFearData* hopeFear);

  // get permutation indices
  void sampleWoReplacement(unsigned int nSamples, std::vector<unsigned int>& indices);
};
