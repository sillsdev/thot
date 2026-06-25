#pragma once

#include "nlp_common/PositionIndex.h"

#include <random>
#include <vector>

// Sampling helpers for the Eflomal Gibbs sampler. thot has no shared random
// number facilities in sw_models, so these are kept small, header-only and
// dependency-free. A fixed seed fed to std::mt19937_64 yields identical
// results across platforms and compilers, which is what the deterministic
// test mode relies on.
namespace SamplingUtils
{
// Draws an index in [0, weights.size()) with probability proportional to the
// (non-negative, unnormalized) entries of weights, whose sum is total. Walking
// the cumulative distribution keeps the draw a pure function of the engine
// state and the weights, so it is reproducible for a given seed. Cross-platform
// bit-identical results are not required: the deterministic test mode pins the
// seed and runs single-threaded, giving identical results within each platform
// independently. mt19937_64 itself is standardized; uniform_real_distribution's
// mapping is an implementation detail but is stable within each toolchain.
inline PositionIndex sampleCategorical(const std::vector<double>& weights, double total, std::mt19937_64& engine)
{
  if (total <= 0.0)
    return 0;
  std::uniform_real_distribution<double> dist(0.0, total);
  double threshold = dist(engine);
  double cumulative = 0.0;
  for (PositionIndex i = 0; i < (PositionIndex)weights.size(); ++i)
  {
    cumulative += weights[i];
    if (threshold < cumulative)
      return i;
  }
  return (PositionIndex)weights.size() - 1;
}

// Draws an index in [0, n) from an unnormalized CUMULATIVE distribution, where
// cumulative[i] holds the running sum of weights up to and including i and total
// is cumulative[n-1]. This is the form the Eflomal sampler builds its conditional
// in, so sampling needs no separate normalization pass. Same portability note as
// sampleCategorical applies: cross-platform bit-identity is not required.
inline PositionIndex sampleCumulative(const std::vector<double>& cumulative, double total, std::mt19937_64& engine,
                                      PositionIndex n)
{
  if (total <= 0.0)
    return 0;
  std::uniform_real_distribution<double> dist(0.0, total);
  double threshold = dist(engine);
  for (PositionIndex i = 0; i < n; ++i)
  {
    if (threshold < cumulative[i])
      return i;
  }
  return n - 1;
}

// Returns the index of the largest weight, breaking ties towards the lowest
// index. Used when a deterministic argmax is preferred over a draw.
inline PositionIndex argmax(const std::vector<double>& weights)
{
  PositionIndex best = 0;
  double bestValue = weights.empty() ? 0.0 : weights[0];
  for (PositionIndex i = 1; i < (PositionIndex)weights.size(); ++i)
  {
    if (weights[i] > bestValue)
    {
      bestValue = weights[i];
      best = i;
    }
  }
  return best;
}
} // namespace SamplingUtils
