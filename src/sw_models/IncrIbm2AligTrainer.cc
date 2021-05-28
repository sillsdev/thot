#include "IncrIbm2AligTrainer.h"

using namespace std;

IncrIbm2AligTrainer::IncrIbm2AligTrainer(Ibm2AligModel& model, anjiMatrix& anji)
  : IncrIbm1AligTrainer(model, anji), model(model)
{
}

double IncrIbm2AligTrainer::calc_anji_num(const vector<WordIndex>& nsrcSent, const vector<WordIndex>& trgSent,
  unsigned int i, unsigned int j)
{
  double d = IncrIbm1AligTrainer::calc_anji_num(nsrcSent, trgSent, i, j);
  d = d * calc_anji_num_alig(i, j, (PositionIndex)nsrcSent.size() - 1, (PositionIndex)trgSent.size());
  return d;
}

double IncrIbm2AligTrainer::calc_anji_num_alig(PositionIndex i, PositionIndex j, PositionIndex slen, PositionIndex tlen)
{
  bool found;
  aSource as;
  as.j = j;
  as.slen = slen;
  as.tlen = tlen;
  model.aSourceMask(as);

  model.aligTable.getAligNumer(as, i, found);
  if (found)
  {
    // alig. parameter has previously been seen
    return model.unsmoothed_aProb(as.j, as.slen, as.tlen, i);
  }
  else
  {
    // alig. parameter has never been seen
    return ArbitraryAp;
  }
}

void IncrIbm2AligTrainer::incrUpdateCounts(unsigned int mapped_n, unsigned int mapped_n_aux, PositionIndex i,
  PositionIndex j, const vector<WordIndex>& nsrcSent, const vector<WordIndex>& trgSent, const Count& weight)
{
  IncrIbm1AligTrainer::incrUpdateCounts(mapped_n, mapped_n_aux, i, j, nsrcSent, trgSent, weight);
  incrUpdateCountsAlig(mapped_n, mapped_n_aux, i, j, (PositionIndex)nsrcSent.size() - 1, (PositionIndex)trgSent.size(),
    weight);
}

void IncrIbm2AligTrainer::incrUpdateCountsAlig(unsigned int mapped_n, unsigned int mapped_n_aux, PositionIndex i,
  PositionIndex j, PositionIndex slen, PositionIndex tlen, const Count& weight)
{
  // Init vars
  float curr_anji = anji.get_fast(mapped_n, j, i);
  float weighted_curr_anji = 0;
  if (curr_anji != INVALID_ANJI_VAL)
  {
    weighted_curr_anji = (float)weight * curr_anji;
    if (weighted_curr_anji < SmoothingWeightedAnji)
      weighted_curr_anji = SmoothingWeightedAnji;
  }

  float weighted_new_anji = (float)weight * anji_aux.get_invp_fast(mapped_n_aux, j, i);
  if (weighted_new_anji < SmoothingWeightedAnji)
    weighted_new_anji = SmoothingWeightedAnji;

  // Init aSource data structure
  aSource as;
  as.j = j;
  as.slen = slen;
  as.tlen = tlen;
  model.aSourceMask(as);

  // Obtain logarithms
  float weighted_curr_lanji;
  if (weighted_curr_anji == 0)
    weighted_curr_lanji = SMALL_LG_NUM;
  else
    weighted_curr_lanji = log(weighted_curr_anji);

  float weighted_new_lanji = log(weighted_new_anji);

  // Store contributions
  IncrAligCountsEntry& elem = incrAligCounts[as];
  while (elem.size() < slen + 1)
    elem.push_back(make_pair(SMALL_LG_NUM, SMALL_LG_NUM));
  pair<float, float>& p = elem[i];
  if (p.first != SMALL_LG_NUM || p.second != SMALL_LG_NUM)
  {
    if (weighted_curr_lanji != SMALL_LG_NUM)
      p.first = MathFuncs::lns_sumlog_float(p.first, weighted_curr_lanji);
    p.second = MathFuncs::lns_sumlog_float(p.second, weighted_new_lanji);
  }
  else
  {
    p.first = weighted_curr_lanji;
    p.second = weighted_new_lanji;
  }
}

void IncrIbm2AligTrainer::incrMaximizeProbs()
{
  IncrIbm1AligTrainer::incrMaximizeProbs();
  incrMaximizeProbsAlig();
}

void IncrIbm2AligTrainer::incrMaximizeProbsAlig()
{
  // Update parameters
  for (IncrAligCounts::iterator aligAuxVarIter = incrAligCounts.begin(); aligAuxVarIter != incrAligCounts.end();
    ++aligAuxVarIter)
  {
    aSource as = aligAuxVarIter->first;
    IncrAligCountsEntry& elem = aligAuxVarIter->second;
    for (PositionIndex i = 0; i < elem.size(); ++i)
    {
      float log_suff_stat_curr = elem[i].first;
      float log_suff_stat_new = elem[i].second;

      // Update parameters only if current and new sufficient statistics
      // are different
      if (log_suff_stat_curr != log_suff_stat_new)
      {
        // Obtain aligNumer
        bool found;
        float numer = model.aligTable.getAligNumer(as, i, found);
        if (!found) numer = SMALL_LG_NUM;

        // Obtain aligDenom
        float denom = model.aligTable.getAligDenom(as, found);
        if (!found) denom = SMALL_LG_NUM;

        // Obtain new sufficient statistics
        float new_numer = obtainLogNewSuffStat(numer, log_suff_stat_curr, log_suff_stat_new);
        float new_denom = denom;
        if (numer != SMALL_LG_NUM)
          new_denom = MathFuncs::lns_sublog_float(denom, numer);
        new_denom = MathFuncs::lns_sumlog_float(new_denom, new_numer);

        // Set lexical numerator and denominator
        model.aligTable.setAligNumDen(as, i, new_numer, new_denom);
      }
    }
  }
  // Clear auxiliary variables
  incrAligCounts.clear();
}

void IncrIbm2AligTrainer::clear()
{
  IncrIbm1AligTrainer::clear();
  incrAligCounts.clear();
}

IncrIbm2AligTrainer::~IncrIbm2AligTrainer()
{
}
