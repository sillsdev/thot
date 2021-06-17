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
 * @file IncrIbm1AligModel.cc
 *
 * @brief Definitions file for IncrIbm1AligModel.h
 */

//--------------- Include files --------------------------------------

#include "IncrIbm1AligModel.h"
#ifdef _WIN32
#include <Windows.h>
#endif
#include "Md.h"

using namespace std;

IncrIbm1AligModel::IncrIbm1AligModel()
{
  // Link pointers with sentence length model
  sentLengthModel.linkVocabPtr(&swVocab);
  sentLengthModel.linkSentPairInfo(&sentenceHandler);
}

void IncrIbm1AligModel::set_expval_maxnsize(unsigned int _anji_maxnsize)
{
  anji.set_maxnsize(_anji_maxnsize);
}

unsigned int IncrIbm1AligModel::numSentPairs(void)
{
  return sentenceHandler.numSentPairs();
}

void IncrIbm1AligModel::trainSentPairRange(pair<unsigned int, unsigned int> sentPairRange, int verbosity)
{
  // Train sentence length model
  sentLengthModel.trainSentPairRange(sentPairRange, verbosity);

  // EM algorithm
  calcNewLocalSuffStats(sentPairRange, verbosity);
  updatePars();
}

void IncrIbm1AligModel::trainAllSents(int verbosity)
{
  clearSentLengthModel();
  if (numSentPairs() > 0)
    trainSentPairRange(std::make_pair(0, numSentPairs() - 1), verbosity);
}

void IncrIbm1AligModel::efficientBatchTrainingForRange(pair<unsigned int, unsigned int> sentPairRange, int verbosity)
{
  if (iter == 0)
  {
    initialBatchPass(sentPairRange, verbosity);

    // Train sentence length model
    sentLengthModel.trainSentPairRange(sentPairRange, verbosity);
  }

  SentPairCont buffer;
  for (unsigned int n = sentPairRange.first; n <= sentPairRange.second; ++n)
  {
    Sentence src = getSrcSent(n);
    Sentence trg = getTrgSent(n);
    if (sentenceLengthIsOk(src) && sentenceLengthIsOk(trg))
      buffer.push_back(pair<Sentence, Sentence>(src, trg));

    if (buffer.size() >= ThreadBufferSize)
    {
      updateFromPairs(buffer);
      buffer.clear();
    }
  }
  if (buffer.size() > 0)
  {
    updateFromPairs(buffer);
    buffer.clear();
  }

  normalizeCounts();
  iter++;
}

void IncrIbm1AligModel::initialBatchPass(pair<unsigned int, unsigned int> sentPairRange, int verbose)
{
  clearTempVars();
  vector<vector<unsigned>> insertBuffer;
  size_t insertBufferItems = 0;
  for (unsigned int n = sentPairRange.first; n <= sentPairRange.second; ++n)
  {
    Sentence src = getSrcSent(n);
    Sentence trg = getTrgSent(n);

    if (sentenceLengthIsOk(src) && sentenceLengthIsOk(trg))
    {
      Sentence nsrc = extendWithNullWord(src);
      for (PositionIndex i = 0; i < nsrc.size(); ++i)
      {
        initSourceWord(nsrc, trg, i);
        WordIndex s = nsrc[i];
        if (s >= insertBuffer.size())
          insertBuffer.resize((size_t)s + 1);
        for (PositionIndex j = 1; j <= trg.size(); ++j)
        {
          if (i == 0)
            initTargetWord(nsrc, trg, j);
          initWordPair(nsrc, trg, i, j);
          insertBuffer[s].push_back(trg[j - 1]);
        }
        insertBufferItems += trg.size();
      }
      if (insertBufferItems > ThreadBufferSize * 100)
      {
        insertBufferItems = 0;
        addTranslationOptions(insertBuffer);
      }
    }
  }
  addTranslationOptions(insertBuffer);
}

void IncrIbm1AligModel::initSourceWord(const Sentence& nsrc, const Sentence& trg, PositionIndex i)
{
  WordIndex s = nsrc[i];
  incrLexTable.setLexDenom(s, 0);
}

void IncrIbm1AligModel::initTargetWord(const Sentence& nsrc, const Sentence& trg, PositionIndex j)
{
}

void IncrIbm1AligModel::initWordPair(const Sentence& nsrc, const Sentence& trg, PositionIndex i, PositionIndex j)
{
}

void IncrIbm1AligModel::addTranslationOptions(vector<vector<WordIndex>>& insertBuffer)
{
  WordIndex maxSrcWordIndex = (WordIndex)insertBuffer.size() - 1;
  if (maxSrcWordIndex >= lexAuxVar.size())
    lexAuxVar.resize((size_t)maxSrcWordIndex + 1);
  incrLexTable.reserveSpace(maxSrcWordIndex);

#pragma omp parallel for schedule(dynamic)
  for (int s = 0; s < (int)insertBuffer.size(); ++s)
  {
    for (WordIndex t : insertBuffer[s])
    {
      lexAuxVar[s][t] = 0;
      incrLexTable.setLexNumer(s, t, 0);
    }
    insertBuffer[s].clear();
  }
}

void IncrIbm1AligModel::updateFromPairs(const SentPairCont& pairs)
{
#pragma omp parallel for schedule(dynamic)
  for (int line_idx = 0; line_idx < (int)pairs.size(); ++line_idx)
  {
    Sentence src = pairs[line_idx].first;
    Sentence nsrc = extendWithNullWord(src);
    Sentence trg = pairs[line_idx].second;
    vector<double> probs(nsrc.size());
    for (PositionIndex j = 1; j <= trg.size(); ++j)
    {
      double sum = 0;
      for (PositionIndex i = 0; i < nsrc.size(); ++i)
      {
        probs[i] = calc_anji_num(nsrc, trg, i, j);
        if (probs[i] < SmoothingAnjiNum)
          probs[i] = SmoothingAnjiNum;
        sum += probs[i];
      }
      for (PositionIndex i = 0; i < nsrc.size(); ++i)
      {
        double count = probs[i] / sum;
        incrementCount(nsrc, trg, i, j, count);
      }
    }
  }
}

void IncrIbm1AligModel::incrementCount(const Sentence& nsrc, const Sentence& trg, PositionIndex i, PositionIndex j,
                                       double count)
{
  WordIndex s = nsrc[i];
  WordIndex t = trg[j - 1];

#pragma omp atomic
  lexAuxVar[s].find(t)->second += count;
}

void IncrIbm1AligModel::normalizeCounts()
{
#pragma omp parallel for schedule(dynamic)
  for (int s = 0; s < (int)lexAuxVar.size(); ++s)
  {
    double denom = 0;
    LexAuxVarElem& elem = lexAuxVar[s];
    for (LexAuxVarElem::iterator it = elem.begin(); it != elem.end(); ++it)
    {
      double numer = it->second;
      if (variationalBayes)
        numer += alpha;
      denom += numer;
      incrLexTable.setLexNumer(s, it->first, (float)log(numer));
      it->second = 0.0;
    }
    if (denom == 0)
      denom = 1;
    incrLexTable.setLexDenom(s, (float)log(denom));
  }
}

pair<double, double> IncrIbm1AligModel::loglikelihoodForPairRange(pair<unsigned int, unsigned int> sentPairRange,
                                                                  int verbosity)
{
  double loglikelihood = 0;
  unsigned int numSents = 0;

  for (unsigned int n = sentPairRange.first; n <= sentPairRange.second; ++n)
  {
    if (verbosity)
      cerr << "* Calculating log-likelihood for sentence " << n << std::endl;
    // Add log-likelihood
    vector<WordIndex> nthSrcSent = getSrcSent(n);
    vector<WordIndex> nthTrgSent = getTrgSent(n);
    if (sentenceLengthIsOk(nthSrcSent) && sentenceLengthIsOk(nthTrgSent))
    {
      loglikelihood += (double)calcLgProb(nthSrcSent, nthTrgSent, verbosity);
      ++numSents;
    }
  }
  return make_pair(loglikelihood, loglikelihood / (double)numSents);
}

vector<WordIndex> IncrIbm1AligModel::getSrcSent(unsigned int n)
{
  vector<string> srcsStr;
  vector<WordIndex> result;

  sentenceHandler.getSrcSent(n, srcsStr);
  for (unsigned int i = 0; i < srcsStr.size(); ++i)
  {
    WordIndex widx = stringToSrcWordIndex(srcsStr[i]);
    if (widx == UNK_WORD)
      widx = addSrcSymbol(srcsStr[i]);
    result.push_back(widx);
  }
  return result;
}

vector<WordIndex> IncrIbm1AligModel::extendWithNullWord(const vector<WordIndex>& srcWordIndexVec)
{
  return addNullWordToWidxVec(srcWordIndexVec);
}

vector<WordIndex> IncrIbm1AligModel::getTrgSent(unsigned int n)
{
  vector<string> trgsStr;
  vector<WordIndex> trgs;

  sentenceHandler.getTrgSent(n, trgsStr);
  for (unsigned int i = 0; i < trgsStr.size(); ++i)
  {
    WordIndex widx = stringToTrgWordIndex(trgsStr[i]);
    if (widx == UNK_WORD)
      widx = addTrgSymbol(trgsStr[i]);
    trgs.push_back(widx);
  }
  return trgs;
}

bool IncrIbm1AligModel::sentenceLengthIsOk(const vector<WordIndex> sentence)
{
  if (sentence.empty() || sentence.size() > IBM_SWM_MAX_SENT_LENGTH)
    return false;
  else
    return true;
}

void IncrIbm1AligModel::calcNewLocalSuffStats(pair<unsigned int, unsigned int> sentPairRange, int verbosity)
{
  // Iterate over the training samples
  for (unsigned int n = sentPairRange.first; n <= sentPairRange.second; ++n)
  {
    // Calculate sufficient statistics

    // Init vars for n'th sample
    vector<WordIndex> srcSent = getSrcSent(n);
    vector<WordIndex> nsrcSent = extendWithNullWord(srcSent);
    vector<WordIndex> trgSent = getTrgSent(n);

    Count weight;
    sentenceHandler.getCount(n, weight);

    // Process sentence pair only if both sentences are not empty
    if (sentenceLengthIsOk(srcSent) && sentenceLengthIsOk(trgSent))
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

void IncrIbm1AligModel::calc_anji(unsigned int n, const vector<WordIndex>& nsrcSent, const vector<WordIndex>& trgSent,
                                  const Count& weight)
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
      double d = calc_anji_num(nsrcSent, trgSent, i, j);
      if (d < SmoothingAnjiNum)
        d = SmoothingAnjiNum;
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
        fillEmAuxVars(mapped_n, mapped_n_aux, i, j, nsrcSent, trgSent, weight);

        // Update anji
        anji.set_fast(mapped_n, j, i, anji_aux.get_invp(n_aux, j, i));
      }
    }
    // clear anji_aux data structure
    anji_aux.clear();
  }
}

double IncrIbm1AligModel::calc_anji_num(const vector<WordIndex>& nsrcSent, const vector<WordIndex>& trgSent,
                                        PositionIndex i, PositionIndex j)
{
  bool found;
  WordIndex s = nsrcSent[i];
  WordIndex t = trgSent[j - 1];

  incrLexTable.getLexNumer(s, t, found);
  if (found)
  {
    // s,t has previously been seen
    return unsmoothed_pts(s, t);
  }
  else
  {
    // s,t has never been seen
    return ArbitraryPts;
  }
}

void IncrIbm1AligModel::fillEmAuxVars(unsigned int mapped_n, unsigned int mapped_n_aux, PositionIndex i,
                                      PositionIndex j, const vector<WordIndex>& nsrcSent,
                                      const vector<WordIndex>& trgSent, const Count& weight)
{
  // Init vars
  float weighted_curr_anji = 0;
  float curr_anji = anji.get_fast(mapped_n, j, i);
  if (curr_anji != INVALID_ANJI_VAL)
  {
    weighted_curr_anji = (float)weight * curr_anji;
    if (weighted_curr_anji < SmoothingWeightedAnji)
      weighted_curr_anji = SmoothingWeightedAnji;
  }

  float weighted_new_anji = (float)weight * anji_aux.get_invp_fast(mapped_n_aux, j, i);
  if (weighted_new_anji != 0 && weighted_new_anji < SmoothingWeightedAnji)
    weighted_new_anji = SmoothingWeightedAnji;

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
  while (incrLexAuxVar.size() <= s)
  {
    IncrLexAuxVarElem lexAuxVarElem;
    incrLexAuxVar.push_back(lexAuxVarElem);
  }

  IncrLexAuxVarElem::iterator lexAuxVarElemIter = incrLexAuxVar[s].find(t);
  if (lexAuxVarElemIter != incrLexAuxVar[s].end())
  {
    if (weighted_curr_lanji != SMALL_LG_NUM)
      lexAuxVarElemIter->second.first =
          MathFuncs::lns_sumlog_float(lexAuxVarElemIter->second.first, weighted_curr_lanji);
    lexAuxVarElemIter->second.second =
        MathFuncs::lns_sumlog_float(lexAuxVarElemIter->second.second, weighted_new_lanji);
  }
  else
  {
    incrLexAuxVar[s][t] = make_pair(weighted_curr_lanji, weighted_new_lanji);
  }
}

void IncrIbm1AligModel::updatePars()
{
  float initialNumer = variationalBayes ? (float)log(alpha) : SMALL_LG_NUM;
  // Update parameters
  for (unsigned int i = 0; i < incrLexAuxVar.size(); ++i)
  {
    for (IncrLexAuxVarElem::iterator lexAuxVarElemIter = incrLexAuxVar[i].begin();
         lexAuxVarElemIter != incrLexAuxVar[i].end(); ++lexAuxVarElemIter)
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
        float numer = incrLexTable.getLexNumer(s, t, numerFound);
        if (!numerFound)
          numer = initialNumer;

        // Obtain lexDenom for s,t
        bool denomFound;
        float denom = incrLexTable.getLexDenom(s, denomFound);
        if (!denomFound)
          denom = SMALL_LG_NUM;

        // Obtain new sufficient statistics
        float new_numer = obtainLogNewSuffStat(numer, log_suff_stat_curr, log_suff_stat_new);
        float new_denom = denom;
        if (numerFound)
          new_denom = MathFuncs::lns_sublog_float(denom, numer);
        new_denom = MathFuncs::lns_sumlog_float(new_denom, new_numer);

        // Set lexical numerator and denominator
        incrLexTable.setLexNumDen(s, t, new_numer, new_denom);
      }
    }
  }
  // Clear auxiliary variables
  incrLexAuxVar.clear();
}

float IncrIbm1AligModel::obtainLogNewSuffStat(float lcurrSuffStat, float lLocalSuffStatCurr, float lLocalSuffStatNew)
{
  float lresult = MathFuncs::lns_sublog_float(lcurrSuffStat, lLocalSuffStatCurr);
  lresult = MathFuncs::lns_sumlog_float(lresult, lLocalSuffStatNew);
  return lresult;
}

Prob IncrIbm1AligModel::pts(WordIndex s, WordIndex t)
{
  return unsmoothed_pts(s, t);
}

double IncrIbm1AligModel::unsmoothed_pts(WordIndex s, WordIndex t)
{
  return exp(unsmoothed_logpts(s, t));
}

LgProb IncrIbm1AligModel::logpts(WordIndex s, WordIndex t)
{
  return unsmoothed_logpts(s, t);
}

double IncrIbm1AligModel::unsmoothed_logpts(WordIndex s, WordIndex t)
{
  bool found;
  double numer;

  numer = incrLexTable.getLexNumer(s, t, found);
  if (found)
  {
    // lexNumer for pair s,t exists
    double denom;

    denom = incrLexTable.getLexDenom(s, found);
    if (!found)
      return SMALL_LG_NUM;
    else
    {
      if (variationalBayes)
      {
        numer = Md::digamma(exp(numer));
        denom = Md::digamma(exp(denom));
      }
      return numer - denom;
    }
  }
  else
  {
    // lexNumer for pair s,t does not exist
    return SMALL_LG_NUM;
  }
}

Prob IncrIbm1AligModel::aProbIbm1(PositionIndex slen, PositionIndex tlen)
{
  return (double)exp((double)logaProbIbm1(slen, tlen));
}

LgProb IncrIbm1AligModel::logaProbIbm1(PositionIndex slen, PositionIndex tlen)
{
  LgProb aligLgProb = 0;

  for (unsigned int j = 0; j < tlen; ++j)
  {
    aligLgProb = (double)aligLgProb - (double)log((double)slen + 1);
  }
  return aligLgProb;
}

Prob IncrIbm1AligModel::sentLenProb(PositionIndex slen, PositionIndex tlen)
{
  return sentLengthModel.sentLenProb(slen, tlen);
}

LgProb IncrIbm1AligModel::sentLenLgProb(PositionIndex slen, PositionIndex tlen)
{
  return sentLengthModel.sentLenLgProb(slen, tlen);
}

LgProb IncrIbm1AligModel::lexM1LpForBestAlig(vector<WordIndex> nSrcSentIndexVector,
                                             vector<WordIndex> trgSentIndexVector, vector<PositionIndex>& bestAlig)
{
  LgProb aligLgProb;
  LgProb lp;
  LgProb max_lp;
  unsigned int best_i = 0;

  bestAlig.clear();
  aligLgProb = 0;
  for (PositionIndex j = 1; j <= trgSentIndexVector.size(); ++j)
  {
    max_lp = -FLT_MAX;
    for (PositionIndex i = 0; i < nSrcSentIndexVector.size(); ++i)
    {
      lp = log((double)pts(nSrcSentIndexVector[i], trgSentIndexVector[j - 1]));
      if (max_lp <= lp)
      {
        max_lp = lp;
        best_i = i;
      }
    }
    aligLgProb = aligLgProb + max_lp;
    bestAlig.push_back(best_i);
  }

  return aligLgProb;
}

bool IncrIbm1AligModel::getEntriesForSource(WordIndex s, NbestTableNode<WordIndex>& trgtn)
{
  set<WordIndex> transSet;
  bool ret = incrLexTable.getTransForSource(s, transSet);
  if (ret == false)
    return false;

  trgtn.clear();
  set<WordIndex>::const_iterator setIter;
  for (setIter = transSet.begin(); setIter != transSet.end(); ++setIter)
  {
    WordIndex t = *setIter;
    trgtn.insert(pts(s, t), t);
  }
  return true;
}

LgProb IncrIbm1AligModel::obtainBestAlignment(vector<WordIndex> srcSentIndexVector,
                                              vector<WordIndex> trgSentIndexVector, WordAligMatrix& bestWaMatrix)
{
  if (sentenceLengthIsOk(srcSentIndexVector) && sentenceLengthIsOk(trgSentIndexVector))
  {
    vector<PositionIndex> bestAlig;
    LgProb lgProb = logaProbIbm1((PositionIndex)srcSentIndexVector.size(), (PositionIndex)trgSentIndexVector.size());
    lgProb += sentLenLgProb((PositionIndex)srcSentIndexVector.size(), (PositionIndex)trgSentIndexVector.size());
    lgProb += lexM1LpForBestAlig(addNullWordToWidxVec(srcSentIndexVector), trgSentIndexVector, bestAlig);

    bestWaMatrix.init((PositionIndex)srcSentIndexVector.size(), (PositionIndex)trgSentIndexVector.size());
    bestWaMatrix.putAligVec(bestAlig);

    return lgProb;
  }
  else
  {
    bestWaMatrix.init((PositionIndex)srcSentIndexVector.size(), (PositionIndex)trgSentIndexVector.size());
    return SMALL_LG_NUM;
  }
}

LgProb IncrIbm1AligModel::calcLgProbForAlig(const vector<WordIndex>& sSent, const vector<WordIndex>& tSent,
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
    return incrIBM1LgProb(addNullWordToWidxVec(sSent), tSent, alig, verbose);
  }
}

LgProb IncrIbm1AligModel::incrIBM1LgProb(vector<WordIndex> nsSent, vector<WordIndex> tSent, vector<PositionIndex> alig,
                                         int verbose)
{
  Prob p;
  LgProb lgProb;
  PositionIndex j;
  if (verbose)
    cerr << "Obtaining IBM Model 1 logprob...\n";

  lgProb = logaProbIbm1((PositionIndex)nsSent.size() - 1, (PositionIndex)tSent.size());
  if (verbose)
    cerr << "- aligLgProb(tlen=" << tSent.size() << " | slen=" << nsSent.size() - 1
         << ")= " << logaProbIbm1((PositionIndex)nsSent.size() - 1, (PositionIndex)tSent.size()) << endl;

  lgProb += sentLenLgProb((PositionIndex)nsSent.size() - 1, (PositionIndex)tSent.size());
  if (verbose)
    cerr << "- lenLgProb(tlen=" << tSent.size() << " | slen=" << nsSent.size() - 1
         << ")= " << sentLenLgProb((PositionIndex)nsSent.size() - 1, (PositionIndex)tSent.size()) << endl;

  for (j = 1; j <= alig.size(); ++j)
  {
    p = pts(nsSent[alig[j - 1]], tSent[j - 1]);
    if (verbose)
      cerr << "t(" << tSent[j - 1] << "|" << nsSent[alig[j - 1]] << ")= " << p << " ; logp=" << (double)log((double)p)
           << endl;
    lgProb = lgProb + (double)log((double)p);
  }

  return lgProb;
}

LgProb IncrIbm1AligModel::calcLgProb(const vector<WordIndex>& sSent, const vector<WordIndex>& tSent, int verbose)
{
  if (sentenceLengthIsOk(sSent) && sentenceLengthIsOk(tSent))
  {
    return calcSumIBM1LgProb(addNullWordToWidxVec(sSent), tSent, verbose);
  }
  else
  {
    return SMALL_LG_NUM;
  }
}

LgProb IncrIbm1AligModel::calcSumIBM1LgProb(const char* sSent, const char* tSent, int verbose)
{
  vector<string> nsSentVec, tSentVec;

  nsSentVec = StrProcUtils::charItemsToVector(sSent);
  nsSentVec = addNullWordToStrVec(nsSentVec);
  tSentVec = StrProcUtils::charItemsToVector(tSent);
  return calcSumIBM1LgProb(nsSentVec, tSentVec, verbose);
}

LgProb IncrIbm1AligModel::calcSumIBM1LgProb(vector<string> nsSent, vector<string> tSent, int verbose)
{
  vector<WordIndex> neIndexVector, fIndexVector;

  neIndexVector = strVectorToSrcIndexVector(nsSent);
  fIndexVector = strVectorToTrgIndexVector(tSent);

  return calcSumIBM1LgProb(neIndexVector, fIndexVector, verbose);
}

LgProb IncrIbm1AligModel::calcSumIBM1LgProb(vector<WordIndex> nsSent, vector<WordIndex> tSent, int verbose)
{
  Prob sump;
  LgProb lexContrib;
  LgProb lgProb;
  PositionIndex i, j;

  if (verbose)
    cerr << "Obtaining Sum IBM Model 1 logprob...\n";

  lgProb = logaProbIbm1((PositionIndex)nsSent.size() - 1, (PositionIndex)tSent.size());

  if (verbose)
    cerr << "- aligLgProb(tlen=" << tSent.size() << " | slen=" << nsSent.size() - 1
         << ")= " << logaProbIbm1((PositionIndex)nsSent.size() - 1, (PositionIndex)tSent.size()) << endl;

  lgProb += sentLenLgProb((PositionIndex)nsSent.size() - 1, (PositionIndex)tSent.size());
  if (verbose)
    cerr << "- lenLgProb(tlen=" << tSent.size() << " | slen=" << nsSent.size() - 1
         << ")= " << sentLenLgProb((PositionIndex)nsSent.size() - 1, (PositionIndex)tSent.size()) << endl;

  lexContrib = 0;
  for (j = 1; j <= tSent.size(); ++j)
  {
    sump = 0;
    for (i = 0; i < nsSent.size(); ++i)
    {
      sump += pts(nsSent[i], tSent[j - 1]);
      if (verbose == 2)
        cerr << "t( " << tSent[j - 1] << " | " << nsSent[i] << " )= " << pts(nsSent[i], tSent[j - 1]) << endl;
    }
    lexContrib += (double)log((double)sump);
    if (verbose)
      cerr << "- sumt(j=" << j << ")= " << sump << endl;
    if (verbose == 2)
      cerr << endl;
  }

  if (verbose)
    cerr << "- Lexical model contribution= " << lexContrib << endl;
  lgProb += lexContrib;

  return lgProb;
}

bool IncrIbm1AligModel::load(const char* prefFileName, int verbose)
{
  if (prefFileName[0] != 0)
  {
    bool retVal;

    if (verbose)
      cerr << "Loading incremental IBM 1 Model data..." << endl;

    // Load vocabularies if they exist
    string srcVocFileName = prefFileName;
    srcVocFileName = srcVocFileName + ".svcb";
    loadGIZASrcVocab(srcVocFileName.c_str(), verbose);

    string trgVocFileName = prefFileName;
    trgVocFileName = trgVocFileName + ".tvcb";
    loadGIZATrgVocab(trgVocFileName.c_str(), verbose);

    // Load files with source and target sentences
    // Warning: this must be made before reading file with anji
    // values
    string srcsFile = prefFileName;
    srcsFile = srcsFile + ".src";
    string trgsFile = prefFileName;
    trgsFile = trgsFile + ".trg";
    string srctrgcFile = prefFileName;
    srctrgcFile = srctrgcFile + ".srctrgc";
    pair<unsigned int, unsigned int> pui;
    retVal = readSentencePairs(srcsFile.c_str(), trgsFile.c_str(), srctrgcFile.c_str(), pui, verbose);
    if (retVal == THOT_ERROR)
      return THOT_ERROR;

    // Load file with anji values
    retVal = anji.load(prefFileName, verbose);
    if (retVal == THOT_ERROR)
      return THOT_ERROR;

    // Load file with lexical nd values
    string lexNumDenFile = prefFileName;
    lexNumDenFile = lexNumDenFile + ".ibm_lexnd";
    retVal = incrLexTable.load(lexNumDenFile.c_str(), verbose);
    if (retVal == THOT_ERROR)
      return THOT_ERROR;

    // Load average sentence lengths
    string slmodelFile = prefFileName;
    slmodelFile = slmodelFile + ".slmodel";
    retVal = sentLengthModel.load(slmodelFile.c_str(), verbose);
    if (retVal == THOT_ERROR)
      return THOT_ERROR;

    string variationalBayesFile = prefFileName;
    variationalBayesFile = variationalBayesFile + ".var_bayes";
    loadVariationalBayes(variationalBayesFile);

    return THOT_OK;
  }
  else
    return THOT_ERROR;
}

bool IncrIbm1AligModel::print(const char* prefFileName, int verbose)
{
  bool retVal;

  // Print vocabularies
  string srcVocFileName = prefFileName;
  srcVocFileName = srcVocFileName + ".svcb";
  retVal = printGIZASrcVocab(srcVocFileName.c_str());
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  string trgVocFileName = prefFileName;
  trgVocFileName = trgVocFileName + ".tvcb";
  retVal = printGIZATrgVocab(trgVocFileName.c_str());
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  // Print files with source and target sentences to temp files
  string srcsFileTemp = prefFileName;
  srcsFileTemp = srcsFileTemp + ".src.tmp";
  string trgsFileTemp = prefFileName;
  trgsFileTemp = trgsFileTemp + ".trg.tmp";
  string srctrgcFileTemp = prefFileName;
  srctrgcFileTemp = srctrgcFileTemp + ".srctrgc.tmp";
  retVal = printSentPairs(srcsFileTemp.c_str(), trgsFileTemp.c_str(), srctrgcFileTemp.c_str());
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  // close sentence files
  sentenceHandler.clear();

  string srcsFile = prefFileName;
  srcsFile = srcsFile + ".src";
  string trgsFile = prefFileName;
  trgsFile = trgsFile + ".trg";
  string srctrgcFile = prefFileName;
  srctrgcFile = srctrgcFile + ".srctrgc";

  // move temp files to real destination
#ifdef _WIN32
  if (!MoveFileExA(srcsFileTemp.c_str(), srcsFile.c_str(), MOVEFILE_REPLACE_EXISTING))
    return THOT_ERROR;
  if (!MoveFileExA(trgsFileTemp.c_str(), trgsFile.c_str(), MOVEFILE_REPLACE_EXISTING))
    return THOT_ERROR;
  if (!MoveFileExA(srctrgcFileTemp.c_str(), srctrgcFile.c_str(), MOVEFILE_REPLACE_EXISTING))
    return THOT_ERROR;
#else
  if (rename(srcsFileTemp.c_str(), srcsFile.c_str()) != 0)
    return THOT_ERROR;
  if (rename(trgsFileTemp.c_str(), trgsFile.c_str()) != 0)
    return THOT_ERROR;
  if (rename(srctrgcFileTemp.c_str(), srctrgcFile.c_str()) != 0)
    return THOT_ERROR;
#endif

  // reload sentence files
  pair<unsigned int, unsigned int> pui;
  retVal = readSentencePairs(srcsFile.c_str(), trgsFile.c_str(), srctrgcFile.c_str(), pui, verbose);
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  // Print file anji values
  retVal = anji.print(prefFileName);
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  // Print file with lexical nd values
  string lexNumDenFile = prefFileName;
  lexNumDenFile = lexNumDenFile + ".ibm_lexnd";
  retVal = incrLexTable.print(lexNumDenFile.c_str());
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  // Print file with sentence length model
  string slmodelFile = prefFileName;
  slmodelFile = slmodelFile + ".slmodel";
  retVal = sentLengthModel.print(slmodelFile.c_str());
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  string variationalBayesFile = prefFileName;
  variationalBayesFile = variationalBayesFile + ".var_bayes";
  return printVariationalBayes(variationalBayesFile);
}

void IncrIbm1AligModel::clear()
{
  _swAligModel::clear();
  clearSentLengthModel();
  clearTempVars();
  anji.clear();
  incrLexTable.clear();
}

void IncrIbm1AligModel::clearInfoAboutSentRange()
{
  // Clear info about sentence range
  sentenceHandler.clear();
  iter = 0;
  anji.clear();
  anji_aux.clear();
  lexAuxVar.clear();
  incrLexAuxVar.clear();
  clearSentLengthModel();
}

void IncrIbm1AligModel::clearTempVars()
{
  bestLgProbForTrgWord.clear();
  iter = 0;
  anji_aux.clear();
  lexAuxVar.clear();
  incrLexAuxVar.clear();
}

void IncrIbm1AligModel::clearSentLengthModel()
{
  sentLengthModel.clear();
}

IncrIbm1AligModel::~IncrIbm1AligModel()
{
}
