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
 * @file Ibm2AligModel.cc
 *
 * @brief Definitions file for Ibm2AligModel.h
 */

#include "Ibm2AligModel.h"

using namespace std;

Ibm2AligModel::Ibm2AligModel() : Ibm1AligModel()
{
}

void Ibm2AligModel::initTargetWord(const Sentence& nsrc, const Sentence& trg, PositionIndex j)
{
  Ibm1AligModel::initTargetWord(nsrc, trg, j);

  aSource as{j, (PositionIndex)nsrc.size() - 1, (PositionIndex)trg.size()};
  aSourceMask(as);
  aligTable.reserveSpace(as);

  AligCountsElem& elem = aligCounts[as];
  if (elem.size() < nsrc.size())
    elem.resize(nsrc.size(), 0);
}

double Ibm2AligModel::calc_anji_num(const vector<WordIndex>& nsrcSent, const vector<WordIndex>& trgSent, unsigned int i,
                                    unsigned int j)
{
  double d = Ibm1AligModel::calc_anji_num(nsrcSent, trgSent, i, j);
  d = d * aProb(i, j, (PositionIndex)nsrcSent.size() - 1, (PositionIndex)trgSent.size(), true);
  return d;
}

void Ibm2AligModel::incrementWordPairCounts(const Sentence& nsrc, const Sentence& trg, PositionIndex i, PositionIndex j,
                                            double count)
{
  Ibm1AligModel::incrementWordPairCounts(nsrc, trg, i, j, count);

  aSource as;
  as.j = j;
  as.slen = (PositionIndex)nsrc.size() - 1;
  as.tlen = (PositionIndex)trg.size();
  aSourceMask(as);

#pragma omp atomic
  aligCounts[as][i] += count;
}

void Ibm2AligModel::batchMaximizeProbs()
{
  Ibm1AligModel::batchMaximizeProbs();

#pragma omp parallel for schedule(dynamic)
  for (int asIndex = 0; asIndex < (int)aligCounts.size(); ++asIndex)
  {
    double denom = 0;
    const pair<aSource, AligCountsElem>& p = aligCounts.getAt(asIndex);
    const aSource& as = p.first;
    AligCountsElem& elem = const_cast<AligCountsElem&>(p.second);
    for (PositionIndex i = 0; i < elem.size(); ++i)
    {
      double numer = elem[i];
      denom += numer;
      float logNumer = (float)log(numer);
      aligTable.setAligNumer(as, i, logNumer);
      elem[i] = 0.0;
    }
    if (denom == 0)
      denom = 1;
    float logDenom = (float)log(denom);
    aligTable.setAligDenom(as, logDenom);
  }
}

Prob Ibm2AligModel::aProb(PositionIndex j, PositionIndex slen, PositionIndex tlen, PositionIndex i)
{
  return unsmoothed_aProb(j, slen, tlen, i);
}

LgProb Ibm2AligModel::logaProb(PositionIndex j, PositionIndex slen, PositionIndex tlen, PositionIndex i)
{
  return unsmoothed_logaProb(j, slen, tlen, i);
}

double Ibm2AligModel::unsmoothed_aProb(PositionIndex j, PositionIndex slen, PositionIndex tlen, PositionIndex i)
{
  return exp(unsmoothed_logaProb(j, slen, tlen, i));
}

double Ibm2AligModel::unsmoothed_logaProb(PositionIndex j, PositionIndex slen, PositionIndex tlen, PositionIndex i)
{
  aSource as;
  as.j = j;
  as.slen = slen;
  as.tlen = tlen;
  aSourceMask(as);

  bool found;
  double numer = aligTable.getAligNumer(as, i, found);
  if (found)
  {
    // aligNumer for pair as,i exists
    double denom = aligTable.getAligDenom(as, found);
    if (found)
      return numer - denom;
  }
  return SMALL_LG_NUM;
}

double Ibm2AligModel::aProb(PositionIndex j, PositionIndex slen, PositionIndex tlen, PositionIndex i, bool training)
{
  if (training)
  {
    double logProb = unsmoothed_aProb(j, slen, tlen, i);
    if (logProb != SMALL_LG_NUM)
      return exp(logProb);
    return 1.0 / (slen + 1);
  }
  return aProb(j, slen, tlen, i);
}

LgProb Ibm2AligModel::obtainBestAlignment(const vector<WordIndex>& srcSentIndexVector,
                                          const vector<WordIndex>& trgSentIndexVector, WordAligMatrix& bestWaMatrix)
{
  vector<PositionIndex> bestAlig;
  LgProb lgProb = sentLenLgProb((PositionIndex)srcSentIndexVector.size(), (PositionIndex)trgSentIndexVector.size());
  lgProb += lexAligM2LpForBestAlig(addNullWordToWidxVec(srcSentIndexVector), trgSentIndexVector, bestAlig);

  bestWaMatrix.init((PositionIndex)srcSentIndexVector.size(), (PositionIndex)trgSentIndexVector.size());
  bestWaMatrix.putAligVec(bestAlig);

  return lgProb;
}

LgProb Ibm2AligModel::calcLgProbForAlig(const vector<WordIndex>& sSent, const vector<WordIndex>& tSent,
                                        const WordAligMatrix& aligMatrix, int verbose)
{
  PositionIndex i;

  vector<PositionIndex> alig;
  aligMatrix.getAligVec(alig);

  if (verbose)
  {
    for (i = 0; i < sSent.size(); ++i)
      cerr << sSent[i] << " ";
    cerr << "\n";
    for (i = 0; i < tSent.size(); ++i)
      cerr << tSent[i] << " ";
    cerr << "\n";
    for (i = 0; i < alig.size(); ++i)
      cerr << alig[i] << " ";
    cerr << "\n";
  }
  if (tSent.size() != alig.size())
  {
    cerr << "Error: the sentence t and the alignment vector have not the same size." << endl;
    return THOT_ERROR;
  }
  else
  {
    return calcIbm2LgProbForAlig(addNullWordToWidxVec(sSent), tSent, alig, verbose);
  }
}

LgProb Ibm2AligModel::calcIbm2LgProbForAlig(const vector<WordIndex>& nsSent, const vector<WordIndex>& tSent,
                                            const vector<PositionIndex>& alig, int verbose)
{
  PositionIndex slen = (PositionIndex)nsSent.size() - 1;
  PositionIndex tlen = (PositionIndex)tSent.size();

  if (verbose)
    cerr << "Obtaining IBM Model 2 logprob...\n";

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

LgProb Ibm2AligModel::calcLgProb(const vector<WordIndex>& sSent, const vector<WordIndex>& tSent, int verbose)
{
  return calcSumIbm2LgProb(addNullWordToWidxVec(sSent), tSent, verbose);
}

LgProb Ibm2AligModel::calcSumIbm2LgProb(const vector<WordIndex>& nsSent, const vector<WordIndex>& tSent, int verbose)
{
  PositionIndex slen = (PositionIndex)nsSent.size() - 1;
  PositionIndex tlen = (PositionIndex)tSent.size();
  Prob sump;
  LgProb lexAligContrib;

  if (verbose)
    cerr << "Obtaining Sum IBM Model 2 logprob...\n";

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
        cerr << "a( " << i << "| j=" << j << ", slen=" << slen << ", tlen=" << tlen << ")= " << aProb(j, slen, tlen, i)
             << endl;
      }
    }
    lexAligContrib += (double)log((double)sump);
    if (verbose)
      cerr << "- sump(j=" << j << ")= " << sump << endl;
    if (verbose == 2)
      cerr << endl;
  }

  if (verbose)
    cerr << "- Lexical plus alignment contribution= " << lexAligContrib << endl;
  lgProb += lexAligContrib;

  return lgProb;
}

bool Ibm2AligModel::load(const char* prefFileName, int verbose)
{
  if (prefFileName[0] != 0)
  {
    bool retVal;

    // Load IBM 1 Model data
    retVal = Ibm1AligModel::load(prefFileName, verbose);
    if (retVal == THOT_ERROR)
      return THOT_ERROR;

    if (verbose)
      cerr << "Loading incremental IBM 2 Model data..." << endl;

    // Load file with alignment nd values
    string aligNumDenFile = prefFileName;
    aligNumDenFile = aligNumDenFile + ".ibm2_alignd";
    retVal = aligTable.load(aligNumDenFile.c_str(), verbose);
    if (retVal == THOT_ERROR)
      return THOT_ERROR;

    return THOT_OK;
  }
  else
    return THOT_ERROR;
}

bool Ibm2AligModel::print(const char* prefFileName, int verbose)
{
  bool retVal;

  // Print IBM 1 Model data
  retVal = Ibm1AligModel::print(prefFileName);
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  // Print file with alignment nd values
  string aligNumDenFile = prefFileName;
  aligNumDenFile = aligNumDenFile + ".ibm2_alignd";
  retVal = aligTable.print(aligNumDenFile.c_str());
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  return THOT_OK;
}

LgProb Ibm2AligModel::lexAligM2LpForBestAlig(const vector<WordIndex>& nSrcSentIndexVector,
                                             const vector<WordIndex>& trgSentIndexVector,
                                             vector<PositionIndex>& bestAlig)
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

void Ibm2AligModel::aSourceMask(aSource& /*as*/)
{
  // This function is left void for performing a standard estimation
}

void Ibm2AligModel::clear()
{
  Ibm1AligModel::clear();
  aligTable.clear();
}

void Ibm2AligModel::clearInfoAboutSentRange()
{
  Ibm1AligModel::clearInfoAboutSentRange();
  aligCounts.clear();
}

void Ibm2AligModel::clearTempVars()
{
  Ibm1AligModel::clearTempVars();
  aligCounts.clear();
}

Ibm2AligModel::~Ibm2AligModel()
{
}
