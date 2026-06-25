#include "sw_models/EflomalJumpTable.h"

#include "nlp_common/MathDefs.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>

void EflomalJumpTable::setFromCounts(const std::vector<double>& counts, double sum, double alpha, int w)
{
  window = w;
  size_t buckets = size_t{2} * window + 1;
  logProbs.assign(buckets, 0.0);
  double denom = sum + alpha * (double)buckets;
  double logDenom = denom > 0 ? std::log(denom) : 0.0;
  for (size_t b = 0; b < buckets && b < counts.size(); ++b)
    logProbs[b] = std::log(counts[b] + alpha) - logDenom;
}

double EflomalJumpTable::logProb(int offset) const
{
  if (logProbs.empty())
    return SMALL_LG_NUM;
  int clamped = std::max(-window, std::min(window, offset));
  return logProbs[(size_t)(clamped + window)];
}

double EflomalJumpTable::prob(int offset) const
{
  return std::exp(logProb(offset));
}

bool EflomalJumpTable::load(const char* jumpFile, int /*verbose*/)
{
  std::ifstream in(jumpFile);
  if (!in)
    return THOT_ERROR;

  in >> window;
  size_t buckets = size_t{2} * window + 1;
  logProbs.assign(buckets, 0.0);
  for (size_t b = 0; b < buckets; ++b)
    in >> logProbs[b];

  return THOT_OK;
}

bool EflomalJumpTable::print(const char* jumpFile) const
{
  std::ofstream out(jumpFile);
  if (!out)
    return THOT_ERROR;

  out << std::setprecision(std::numeric_limits<double>::max_digits10);
  out << window << "\n";
  for (double lp : logProbs)
    out << lp << "\n";

  return THOT_OK;
}

void EflomalJumpTable::clear()
{
  window = 0;
  logProbs.clear();
}
