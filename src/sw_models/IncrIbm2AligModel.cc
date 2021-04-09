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
 * @file IncrIbm2AligModel.cc
 *
 * @brief Definitions file for IncrIbm2AligModel.h
 */

 //--------------- Include files --------------------------------------

#include "IncrIbm2AligModel.h"

using namespace std;


IncrIbm2AligModel::IncrIbm2AligModel(): IncrIbm1AligModel()
{
}

void IncrIbm2AligModel::initTargetWord(const Sentence& nsrc, const Sentence& trg, PositionIndex j)
{
  IncrIbm1AligModel::initTargetWord(nsrc, trg, j);

  aSource as;
  as.j = j;
  as.slen = (PositionIndex)nsrc.size() - 1;
  as.tlen = (PositionIndex)trg.size();
  aSourceMask(as);
  incrIbm2AligTable.setAligDenom(as, 0);

  AligAuxVarElem& elem = aligAuxVar[as];
  if (elem.size() < nsrc.size())
    elem.resize(nsrc.size(), 0);
}

void IncrIbm2AligModel::initWordPair(const Sentence& nsrc, const Sentence& trg, PositionIndex i, PositionIndex j)
{
  IncrIbm1AligModel::initWordPair(nsrc, trg, i, j);

  aSource as;
  as.j = j;
  as.slen = (PositionIndex)nsrc.size() - 1;
  as.tlen = (PositionIndex)trg.size();
  aSourceMask(as);
  incrIbm2AligTable.setAligNumer(as, i, 0);
}

void IncrIbm2AligModel::incrementCount(const Sentence& nsrc, const Sentence& trg, PositionIndex i, PositionIndex j,
  double count)
{
  IncrIbm1AligModel::incrementCount(nsrc, trg, i, j, count);

  aSource as;
  as.j = j;
  as.slen = (PositionIndex)nsrc.size() - 1;
  as.tlen = (PositionIndex)trg.size();
  aSourceMask(as);

  #pragma omp atomic
  aligAuxVar[as][i] += count;
}

void IncrIbm2AligModel::normalizeCounts()
{
  IncrIbm1AligModel::normalizeCounts();

  #pragma omp parallel for schedule(dynamic)
  for (int asIndex = 0; asIndex < aligAuxVar.size(); ++asIndex)
  {
    double denom = 0;
    const pair<aSource, AligAuxVarElem>& p = aligAuxVar.getAt(asIndex);
    const aSource& as = p.first;
    AligAuxVarElem& elem = const_cast<AligAuxVarElem&>(p.second);
    for (PositionIndex i = 0; i < elem.size(); ++i)
    {
      double numer = elem[i];
      denom += numer;
      float logNumer = (float)log(numer);
      incrIbm2AligTable.setAligNumer(as, i, logNumer);
      elem[i] = 0.0;
    }
    if (denom == 0)
      denom = 1;
    float logDenom = (float)log(denom);
    incrIbm2AligTable.setAligDenom(as, logDenom);
  }
}

double IncrIbm2AligModel::calc_anji_num(const vector<WordIndex>& nsrcSent, const vector<WordIndex>& trgSent,
  unsigned int i, unsigned int j)
{
  double d;

  d = IncrIbm1AligModel::calc_anji_num(nsrcSent, trgSent, i, j);
  d = d * calc_anji_num_alig(i, j, (PositionIndex)nsrcSent.size() - 1, (PositionIndex)trgSent.size());
  return d;
}

double IncrIbm2AligModel::calc_anji_num_alig(PositionIndex i, PositionIndex j, PositionIndex slen, PositionIndex tlen)
{
  bool found;
  aSource as;
  as.j = j;
  as.slen = slen;
  as.tlen = tlen;
  aSourceMask(as);

  incrIbm2AligTable.getAligNumer(as, i, found);
  if (found)
  {
    // alig. parameter has previously been seen
    return unsmoothed_aProb(as.j, as.slen, as.tlen, i);
  }
  else
  {
    // alig. parameter has never been seen
    return ARBITRARY_AP;
  }
}

void IncrIbm2AligModel::fillEmAuxVars(unsigned int mapped_n, unsigned int mapped_n_aux, PositionIndex i,
  PositionIndex j, const vector<WordIndex>& nsrcSent, const vector<WordIndex>& trgSent, const Count& weight)
{
  IncrIbm1AligModel::fillEmAuxVars(mapped_n, mapped_n_aux, i, j, nsrcSent, trgSent, weight);
  fillEmAuxVarsAlig(mapped_n, mapped_n_aux, i, j, (PositionIndex)nsrcSent.size() - 1, (PositionIndex)trgSent.size(),
    weight);
}

void IncrIbm2AligModel::fillEmAuxVarsAlig(unsigned int mapped_n, unsigned int mapped_n_aux, PositionIndex i,
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
  aSourceMask(as);

  // Obtain logarithms
  float weighted_curr_lanji;
  if (weighted_curr_anji == 0)
    weighted_curr_lanji = SMALL_LG_NUM;
  else
    weighted_curr_lanji = log(weighted_curr_anji);

  float weighted_new_lanji = log(weighted_new_anji);

  // Store contributions
  IncrAligAuxVarElem& elem = incrAligAuxVar[as];
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

void IncrIbm2AligModel::updatePars()
{
  IncrIbm1AligModel::updatePars();
  updateParsAlig();
}

void IncrIbm2AligModel::updateParsAlig()
{
  // Update parameters
  for (IncrAligAuxVar::iterator aligAuxVarIter = incrAligAuxVar.begin(); aligAuxVarIter != incrAligAuxVar.end();
    ++aligAuxVarIter)
  {
    aSource as = aligAuxVarIter->first;
    IncrAligAuxVarElem& elem = aligAuxVarIter->second;
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
        float numer = incrIbm2AligTable.getAligNumer(as, i, found);
        if (!found) numer = SMALL_LG_NUM;

        // Obtain aligDenom
        float denom = incrIbm2AligTable.getAligDenom(as, found);
        if (!found) denom = SMALL_LG_NUM;

        // Obtain new sufficient statistics
        float new_numer = obtainLogNewSuffStat(numer, log_suff_stat_curr, log_suff_stat_new);
        float new_denom = denom;
        if (numer != SMALL_LG_NUM)
          new_denom = MathFuncs::lns_sublog_float(denom, numer);
        new_denom = MathFuncs::lns_sumlog_float(new_denom, new_numer);

        // Set lexical numerator and denominator
        incrIbm2AligTable.setAligNumDen(as, i, new_numer, new_denom);
      }
    }
  }
  // Clear auxiliary variables
  incrAligAuxVar.clear();
}

Prob IncrIbm2AligModel::aProb(PositionIndex j, PositionIndex slen, PositionIndex tlen, PositionIndex i)
{
  return unsmoothed_aProb(j, slen, tlen, i);
}

LgProb IncrIbm2AligModel::logaProb(PositionIndex j, PositionIndex slen, PositionIndex tlen, PositionIndex i)
{
  return unsmoothed_logaProb(j, slen, tlen, i);
}

double IncrIbm2AligModel::unsmoothed_aProb(PositionIndex j, PositionIndex slen, PositionIndex tlen, PositionIndex i)
{
  return exp(unsmoothed_logaProb(j, slen, tlen, i));
}

double IncrIbm2AligModel::unsmoothed_logaProb(PositionIndex j, PositionIndex slen, PositionIndex tlen, PositionIndex i)
{
  bool found;
  double numer;
  aSource as;

  as.j = j;
  as.slen = slen;
  as.tlen = tlen;
  aSourceMask(as);

  numer = incrIbm2AligTable.getAligNumer(as, i, found);
  if (found)
  {
    // aligNumer for pair as,i exists
    double denom;
    denom = incrIbm2AligTable.getAligDenom(as, found);
    if (!found) return SMALL_LG_NUM;
    else
    {
      return numer - denom;
    }
  }
  else
  {
    // aligNumer for pair as,i does not exist
    return SMALL_LG_NUM;
  }
}

LgProb IncrIbm2AligModel::obtainBestAlignment(vector<WordIndex> srcSentIndexVector,
  vector<WordIndex> trgSentIndexVector, WordAligMatrix& bestWaMatrix)
{
  vector<PositionIndex> bestAlig;
  LgProb lgProb = sentLenLgProb((PositionIndex)srcSentIndexVector.size(), (PositionIndex)trgSentIndexVector.size());
  lgProb += lexAligM2LpForBestAlig(addNullWordToWidxVec(srcSentIndexVector), trgSentIndexVector, bestAlig);

  bestWaMatrix.init((PositionIndex)srcSentIndexVector.size(), (PositionIndex)trgSentIndexVector.size());
  bestWaMatrix.putAligVec(bestAlig);

  return lgProb;
}

LgProb IncrIbm2AligModel::calcLgProbForAlig(const vector<WordIndex>& sSent, const vector<WordIndex>& tSent,
  WordAligMatrix aligMatrix, int verbose)
{
  PositionIndex i;

  vector<PositionIndex> alig;
  aligMatrix.getAligVec(alig);

  if (verbose)
  {
    for (i = 0; i < sSent.size(); ++i) cerr << sSent[i] << " ";
    cerr << "\n";
    for (i = 0; i < tSent.size(); ++i) cerr << tSent[i] << " ";
    cerr << "\n";
    for (i = 0; i < alig.size(); ++i) cerr << alig[i] << " ";
    cerr << "\n";
  }
  if (tSent.size() != alig.size())
  {
    cerr << "Error: the sentence t and the alignment vector have not the same size." << endl;
    return THOT_ERROR;
  }
  else
  {
    return incrIBM2LgProb(addNullWordToWidxVec(sSent), tSent, alig, verbose);
  }
}

LgProb IncrIbm2AligModel::incrIBM2LgProb(vector<WordIndex> nsSent, vector<WordIndex> tSent, vector<PositionIndex> alig,
  int verbose)
{
  PositionIndex slen = (PositionIndex)nsSent.size() - 1;
  PositionIndex tlen = (PositionIndex)tSent.size();

  if (verbose) cerr << "Obtaining IBM Model 2 logprob...\n";

  LgProb lgProb = 0;
  for (PositionIndex j = 1; j <= alig.size(); ++j)
  {
    Prob p = pts(nsSent[alig[j]], tSent[j - 1]);
    if (verbose)
      cerr << "t(" << tSent[j - 1] << "|" << nsSent[alig[j - 1]] << ")= " << p << " ; logp=" << (double)log((double)p)
        << endl;
    lgProb = lgProb + (double)log((double)p);

    p = aProb(j, slen, tlen, alig[j - 1]);
    lgProb = lgProb + (double)log((double)p);
  }
  return lgProb;
}

LgProb IncrIbm2AligModel::calcLgProb(const vector<WordIndex>& sSent, const vector<WordIndex>& tSent, int verbose)
{
  return calcSumIBM2LgProb(addNullWordToWidxVec(sSent), tSent, verbose);
}

LgProb IncrIbm2AligModel::calcSumIBM2LgProb(vector<WordIndex> nsSent, vector<WordIndex> tSent, int verbose)
{
  PositionIndex slen = (PositionIndex)nsSent.size() - 1;
  PositionIndex tlen = (PositionIndex)tSent.size();
  Prob sump;
  LgProb lexAligContrib;

  if (verbose) cerr << "Obtaining Sum IBM Model 2 logprob...\n";

  LgProb lgProb = sentLenLgProb(slen, tlen);
  if (verbose)
    cerr << "- lenLgProb(tlen=" << tSent.size() << " | slen=" << slen << ")= " << sentLenLgProb(slen, tlen) << endl;

  lexAligContrib = 0;
  for (PositionIndex j = 1; j <= tSent.size(); ++j)
  {
    sump = 0;
    for (PositionIndex i = 0; i < nsSent.size(); ++i)
    {
      sump += pts(nsSent[i], tSent[j - 1]) * aProb(j, slen, tlen, i);
      if (verbose == 2)
      {
        cerr << "t( " << tSent[j - 1] << " | " << nsSent[i] << " )= " << pts(nsSent[i], tSent[j - 1]) << endl;
        cerr << "a( " << i << "| j=" << j << ", slen=" << slen << ", tlen=" << tlen << ")= "
          << aProb(j, slen, tlen, i) << endl;
      }
    }
    lexAligContrib += (double)log((double)sump);
    if (verbose)
      cerr << "- sump(j=" << j << ")= " << sump << endl;
    if (verbose == 2) cerr << endl;
  }

  if (verbose) cerr << "- Lexical plus alignment contribution= " << lexAligContrib << endl;
  lgProb += lexAligContrib;

  return lgProb;
}

bool IncrIbm2AligModel::load(const char* prefFileName, int verbose)
{
  if (prefFileName[0] != 0)
  {
    bool retVal;

    // Load IBM 1 Model data
    retVal = IncrIbm1AligModel::load(prefFileName, verbose);
    if (retVal == THOT_ERROR) return THOT_ERROR;

    if (verbose)
      cerr << "Loading incremental IBM 2 Model data..." << endl;

    // Load file with alignment nd values
    string aligNumDenFile = prefFileName;
    aligNumDenFile = aligNumDenFile + ".ibm2_alignd";
    retVal = incrIbm2AligTable.load(aligNumDenFile.c_str(), verbose);
    if (retVal == THOT_ERROR) return THOT_ERROR;

    return THOT_OK;
  }
  else return THOT_ERROR;
}

bool IncrIbm2AligModel::print(const char* prefFileName, int verbose)
{
  bool retVal;

  // Print IBM 1 Model data
  retVal = IncrIbm1AligModel::print(prefFileName);
  if (retVal == THOT_ERROR) return THOT_ERROR;

  // Print file with alignment nd values
  string aligNumDenFile = prefFileName;
  aligNumDenFile = aligNumDenFile + ".ibm2_alignd";
  retVal = incrIbm2AligTable.print(aligNumDenFile.c_str());
  if (retVal == THOT_ERROR) return THOT_ERROR;

  return THOT_OK;
}

LgProb IncrIbm2AligModel::lexAligM2LpForBestAlig(vector<WordIndex> nSrcSentIndexVector,
  vector<WordIndex> trgSentIndexVector, vector<PositionIndex>& bestAlig)
{
  // Initialize variables
  PositionIndex slen = (PositionIndex)nSrcSentIndexVector.size() - 1;
  PositionIndex tlen = (PositionIndex)trgSentIndexVector.size();
  LgProb aligLgProb = 0;
  bestAlig.clear();

  for (PositionIndex j = 1; j <= trgSentIndexVector.size(); ++j)
  {
    unsigned int best_i = 0;
    LgProb max_lp = -FLT_MAX;
    for (unsigned int i = 0; i < nSrcSentIndexVector.size(); ++i)
    {
      // lexical logprobability
      LgProb lp = log((double)pts(nSrcSentIndexVector[i], trgSentIndexVector[j - 1]));
      // alignment logprobability
      lp += log((double)aProb(j, slen, tlen, i));

      if (max_lp <= lp)
      {
        max_lp = lp;
        best_i = i;
      }
    }
    // Add contribution
    aligLgProb = aligLgProb + max_lp;
    // Add word alignment
    bestAlig.push_back(best_i);
  }
  return aligLgProb;
}

void IncrIbm2AligModel::aSourceMask(aSource&/*as*/)
{
  // This function is left void for performing a standard estimation
}

void IncrIbm2AligModel::clear()
{
  IncrIbm1AligModel::clear();
  incrIbm2AligTable.clear();
}

void IncrIbm2AligModel::clearInfoAboutSentRange()
{
  IncrIbm1AligModel::clearInfoAboutSentRange();
  aligAuxVar.clear();
}

void IncrIbm2AligModel::clearTempVars()
{
  IncrIbm1AligModel::clearTempVars();
  aligAuxVar.clear();
}

IncrIbm2AligModel::~IncrIbm2AligModel()
{

}
