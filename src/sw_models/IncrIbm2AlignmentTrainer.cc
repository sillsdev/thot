#include "sw_models/IncrIbm2AlignmentTrainer.h"

#include "sw_models/SwDefs.h"

using namespace std;

IncrIbm2AlignmentTrainer::IncrIbm2AlignmentTrainer(Ibm2AlignmentModel& model, anjiMatrix& anji)
    : IncrIbm1AlignmentTrainer(model, anji), model(model)
{
}

void IncrIbm2AlignmentTrainer::incrUpdateCounts(unsigned int mapped_n, unsigned int mapped_n_aux, PositionIndex i,
                                                PositionIndex j, const vector<WordIndex>& nsrcSent,
                                                const vector<WordIndex>& trgSent, const Count& weight)
{
  IncrIbm1AlignmentTrainer::incrUpdateCounts(mapped_n, mapped_n_aux, i, j, nsrcSent, trgSent, weight);
  incrUpdateCountsAlig(mapped_n, mapped_n_aux, i, j, (PositionIndex)nsrcSent.size() - 1, (PositionIndex)trgSent.size(),
                       weight);
}

void IncrIbm2AlignmentTrainer::incrUpdateCountsAlig(unsigned int mapped_n, unsigned int mapped_n_aux, PositionIndex i,
                                                    PositionIndex j, PositionIndex slen, PositionIndex tlen,
                                                    const Count& weight)
{
  // Init vars
  float curr_anji = anji.get_fast(mapped_n, j, i);
  float weighted_curr_anji = 0;
  if (curr_anji != INVALID_ANJI_VAL)
  {
    weighted_curr_anji = (float)weight * curr_anji;
    weighted_curr_anji = max(weighted_curr_anji, float{SW_PROB_SMOOTH});
  }

  float weighted_new_anji = (float)weight * anji_aux.get_invp_fast(mapped_n_aux, j, i);
  weighted_new_anji = max(weighted_new_anji, float{SW_PROB_SMOOTH});

  AlignmentKey key{j, slen, model.getCompactedSentenceLength(tlen)};

  // Obtain logarithms
  float weighted_curr_lanji;
  if (weighted_curr_anji == 0)
    weighted_curr_lanji = SMALL_LG_NUM;
  else
    weighted_curr_lanji = log(weighted_curr_anji);

  float weighted_new_lanji = log(weighted_new_anji);

  // Store contributions
  IncrAlignmentCountsElem& elem = incrAlignmentCounts[key];
  while (elem.size() < size_t{slen} + 1)
    elem.push_back(make_pair((float)SMALL_LG_NUM, (float)SMALL_LG_NUM));
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

void IncrIbm2AlignmentTrainer::incrMaximizeProbs()
{
  IncrIbm1AlignmentTrainer::incrMaximizeProbs();
  incrMaximizeProbsAlig();
}

void IncrIbm2AlignmentTrainer::incrMaximizeProbsAlig()
{
  // Update parameters
  for (IncrAlignmentCounts::iterator iter = incrAlignmentCounts.begin(); iter != incrAlignmentCounts.end(); ++iter)
  {
    AlignmentKey key = iter->first;
    IncrAlignmentCountsElem& elem = iter->second;
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
        float numer = model.alignmentTable->getNumerator(key.j, key.slen, key.tlen, i, found);
        if (!found)
          numer = SMALL_LG_NUM;

        // Obtain aligDenom
        float denom = model.alignmentTable->getDenominator(key.j, key.slen, key.tlen, found);
        if (!found)
          denom = SMALL_LG_NUM;

        // Obtain new sufficient statistics
        float new_numer = obtainLogNewSuffStat(numer, log_suff_stat_curr, log_suff_stat_new);
        float new_denom = denom;
        if (numer != SMALL_LG_NUM)
          new_denom = MathFuncs::lns_sublog_float(denom, numer);
        new_denom = MathFuncs::lns_sumlog_float(new_denom, new_numer);

        // Set lexical numerator and denominator
        model.alignmentTable->set(key.j, key.slen, key.tlen, i, new_numer, new_denom);
      }
    }
  }
  // Clear auxiliary variables
  incrAlignmentCounts.clear();
}

void IncrIbm2AlignmentTrainer::clear()
{
  IncrIbm1AlignmentTrainer::clear();
  incrAlignmentCounts.clear();
}
