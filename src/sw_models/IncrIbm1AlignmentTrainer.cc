#include "sw_models/IncrIbm1AlignmentTrainer.h"

#include "sw_models/SwDefs.h"

using namespace std;

IncrIbm1AlignmentTrainer::IncrIbm1AlignmentTrainer(Ibm1AlignmentModel& model, anjiMatrix& anji)
    : anji{anji}, model{model}
{
}

void IncrIbm1AlignmentTrainer::incrTrain(pair<unsigned int, unsigned int> sentPairRange, int verbosity)
{
  // EM algorithm
  calcNewLocalSuffStats(sentPairRange, verbosity);
  incrMaximizeProbs();
}

void IncrIbm1AlignmentTrainer::calcNewLocalSuffStats(pair<unsigned int, unsigned int> sentPairRange, int verbosity)
{
  // Iterate over the training samples
  for (unsigned int n = sentPairRange.first; n <= sentPairRange.second; ++n)
  {
    // Calculate sufficient statistics

    // Init vars for n'th sample
    vector<WordIndex> srcSent = model.getSrcSent(n);
    vector<WordIndex> nsrcSent = model.extendWithNullWord(srcSent);
    vector<WordIndex> trgSent = model.getTrgSent(n);

    Count weight;
    model.sentenceHandler->getCount(n, weight);

    // Process sentence pair only if both sentences are not empty
    if (model.sentenceLengthIsOk(srcSent) && model.sentenceLengthIsOk(trgSent))
    {
      // Calculate sufficient statistics for anji values
      calc_anji(n, nsrcSent, trgSent, weight);
    }
    else
    {
      if (verbosity)
      {
        cerr << "Warning, training pair " << n + 1 << " discarded due to sentence length (slen: " << srcSent.size()
             << " , tlen: " << trgSent.size() << ")" << endl;
      }
    }
  }
}

void IncrIbm1AlignmentTrainer::calc_anji(unsigned int n, const vector<WordIndex>& nsrcSent,
                                         const vector<WordIndex>& trgSent, const Count& weight)
{
  // Initialize anji and anji_aux
  unsigned int mapped_n;
  anji.init_nth_entry(n, (PositionIndex)nsrcSent.size(), (PositionIndex)trgSent.size(), mapped_n);

  unsigned int n_aux = 1;
  unsigned int mapped_n_aux;
  anji_aux.init_nth_entry(n_aux, (PositionIndex)nsrcSent.size(), (PositionIndex)trgSent.size(), mapped_n_aux);

  // Calculate new estimation of anji
  for (PositionIndex j = 1; j <= trgSent.size(); ++j)
  {
    // Obtain sum_anji_num_forall_s
    double sum_anji_num_forall_s = 0;
    vector<double> numVec;
    for (PositionIndex i = 0; i < nsrcSent.size(); ++i)
    {
      // Smooth numerator
      double d = model.getCountNumerator(nsrcSent, trgSent, i, j);
      // Add contribution to sum
      sum_anji_num_forall_s += d;
      // Store num in numVec
      numVec.push_back(d);
    }
    // Set value of anji_aux
    for (PositionIndex i = 0; i < nsrcSent.size(); ++i)
    {
      anji_aux.set_fast(mapped_n_aux, j, i, (float)(numVec[i] / sum_anji_num_forall_s));
    }
  }

  // Gather sufficient statistics
  if (anji_aux.n_size() != 0)
  {
    for (PositionIndex j = 1; j <= trgSent.size(); ++j)
    {
      for (PositionIndex i = 0; i < nsrcSent.size(); ++i)
      {
        // Fill variables for n_aux,j,i
        incrUpdateCounts(mapped_n, mapped_n_aux, i, j, nsrcSent, trgSent, weight);

        // Update anji
        anji.set_fast(mapped_n, j, i, anji_aux.get_invp(n_aux, j, i));
      }
    }
    // clear anji_aux data structure
    anji_aux.clear();
  }
}

void IncrIbm1AlignmentTrainer::incrUpdateCounts(unsigned int mapped_n, unsigned int mapped_n_aux, PositionIndex i,
                                                PositionIndex j, const vector<WordIndex>& nsrcSent,
                                                const vector<WordIndex>& trgSent, const Count& weight)
{
  // Init vars
  float weighted_curr_anji = 0;
  float curr_anji = anji.get_fast(mapped_n, j, i);
  if (curr_anji != INVALID_ANJI_VAL)
    weighted_curr_anji = max(float{weight} * curr_anji, float{SW_PROB_SMOOTH});

  float weighted_new_anji = (float)weight * anji_aux.get_invp_fast(mapped_n_aux, j, i);
  if (weighted_new_anji != 0)
    weighted_new_anji = max(weighted_new_anji, float{SW_PROB_SMOOTH});

  WordIndex s = nsrcSent[i];
  WordIndex t = trgSent[j - 1];

  // Obtain logarithms
  float weighted_curr_lanji;
  if (weighted_curr_anji == 0)
    weighted_curr_lanji = SMALL_LG_NUM;
  else
    weighted_curr_lanji = log(weighted_curr_anji);

  float weighted_new_lanji = log(weighted_new_anji);

  // Store contributions
  while (incrLexCounts.size() <= s)
  {
    IncrLexCountsElem lexAuxVarElem;
    incrLexCounts.push_back(lexAuxVarElem);
  }

  IncrLexCountsElem::iterator lexAuxVarElemIter = incrLexCounts[s].find(t);
  if (lexAuxVarElemIter != incrLexCounts[s].end())
  {
    if (weighted_curr_lanji != SMALL_LG_NUM)
      lexAuxVarElemIter->second.first =
          MathFuncs::lns_sumlog_float(lexAuxVarElemIter->second.first, weighted_curr_lanji);
    lexAuxVarElemIter->second.second =
        MathFuncs::lns_sumlog_float(lexAuxVarElemIter->second.second, weighted_new_lanji);
  }
  else
  {
    incrLexCounts[s][t] = make_pair(weighted_curr_lanji, weighted_new_lanji);
  }
}

void IncrIbm1AlignmentTrainer::incrMaximizeProbs()
{
  float initialNumer = model.variationalBayes ? (float)log(model.alpha) : SMALL_LG_NUM;
  // Update parameters
  for (unsigned int i = 0; i < incrLexCounts.size(); ++i)
  {
    for (IncrLexCountsElem::iterator lexAuxVarElemIter = incrLexCounts[i].begin();
         lexAuxVarElemIter != incrLexCounts[i].end(); ++lexAuxVarElemIter)
    {
      WordIndex s = i; // lexAuxVarElemIter->first.first;
      WordIndex t = lexAuxVarElemIter->first;
      float log_suff_stat_curr = lexAuxVarElemIter->second.first;
      float log_suff_stat_new = lexAuxVarElemIter->second.second;

      // Update parameters only if current and new sufficient statistics
      // are different
      if (log_suff_stat_curr != log_suff_stat_new)
      {
        // Obtain lexNumer for s,t
        bool numerFound;
        float numer = model.lexTable->getNumerator(s, t, numerFound);
        if (!numerFound)
          numer = initialNumer;

        // Obtain lexDenom for s,t
        bool denomFound;
        float denom = model.lexTable->getDenominator(s, denomFound);
        if (!denomFound)
          denom = SMALL_LG_NUM;

        // Obtain new sufficient statistics
        float new_numer = obtainLogNewSuffStat(numer, log_suff_stat_curr, log_suff_stat_new);
        float new_denom = denom;
        if (numerFound)
          new_denom = MathFuncs::lns_sublog_float(denom, numer);
        new_denom = MathFuncs::lns_sumlog_float(new_denom, new_numer);

        // Set lexical numerator and denominator
        model.lexTable->set(s, t, new_numer, new_denom);
      }
    }
  }
  // Clear auxiliary variables
  incrLexCounts.clear();
}

float IncrIbm1AlignmentTrainer::obtainLogNewSuffStat(float lcurrSuffStat, float lLocalSuffStatCurr,
                                                     float lLocalSuffStatNew)
{
  float lresult = MathFuncs::lns_sublog_float(lcurrSuffStat, lLocalSuffStatCurr);
  lresult = MathFuncs::lns_sumlog_float(lresult, lLocalSuffStatNew);
  return lresult;
}

void IncrIbm1AlignmentTrainer::clear()
{
  anji_aux.clear();
  incrLexCounts.clear();
}
