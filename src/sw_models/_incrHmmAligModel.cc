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
 * @file _incrHmmAligModel.cc
 *
 * @brief Definitions file for _incrHmmAligModel.h
 */

#include "sw_models/_incrHmmAligModel.h"

#include "sw_models/Md.h"

#ifdef _WIN32
#include <Windows.h>
#endif

using namespace std;

_incrHmmAligModel::_incrHmmAligModel()
{
  // Link pointers with sentence length model
  sentLengthModel.linkVocabPtr(&swVocab);
  sentLengthModel.linkSentPairInfo(&sentenceHandler);

  // Set default value for aligSmoothInterpFactor
  aligSmoothInterpFactor = DEFAULT_ALIG_SMOOTH_INTERP_FACTOR;

  // Set default value for lexSmoothInterpFactor
  lexSmoothInterpFactor = DEFAULT_LEX_SMOOTH_INTERP_FACTOR;
}

void _incrHmmAligModel::set_expval_maxnsize(unsigned int _expval_maxnsize)
{
  lanji.set_maxnsize(_expval_maxnsize);
  lanjm1ip_anji.set_maxnsize(_expval_maxnsize);
}

unsigned int _incrHmmAligModel::numSentPairs(void)
{
  return sentenceHandler.numSentPairs();
}

void _incrHmmAligModel::trainSentPairRange(pair<unsigned int, unsigned int> sentPairRange, int verbosity)
{
  // Train sentence length model
  sentLengthModel.trainSentPairRange(sentPairRange, verbosity);

  // EM algorithm
#ifdef THOT_ENABLE_VITERBI_TRAINING
  calcNewLocalSuffStatsVit(sentPairRange, verbosity);
#else
  calcNewLocalSuffStats(sentPairRange, verbosity);
#endif
  incrMaximizeProbsLex();
  incrMaximizeProbsAlig();
}

void _incrHmmAligModel::trainAllSents(int verbosity)
{
  clearSentLengthModel();
  if (numSentPairs() > 0)
    trainSentPairRange(std::make_pair(0, numSentPairs() - 1), verbosity);
}

void _incrHmmAligModel::efficientBatchTrainingForRange(pair<unsigned int, unsigned int> sentPairRange, int verbosity)
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
      batchUpdateCounts(buffer);
      buffer.clear();
    }
  }
  if (buffer.size() > 0)
  {
    batchUpdateCounts(buffer);
    buffer.clear();
  }

  batchMaximizeProbs();
  iter++;
}

void _incrHmmAligModel::initialBatchPass(pair<unsigned int, unsigned int> sentPairRange, int verbose)
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
      Sentence nsrcAlig = extendWithNullWordAlig(src);

      PositionIndex slen = (PositionIndex)src.size();

      // Make room for data structure to cache alignment log-probs
      cachedAligLogProbs.makeRoomGivenSrcSentLen(slen);

      aSourceHmm asHmm0;
      asHmm0.prev_i = 0;
      asHmm0.slen = slen;
      aligTable.setAligDenom(asHmm0, 0);
      AligCountsEntry& elem = aligCounts[asHmm0];
      if (elem.size() < nsrcAlig.size())
        elem.resize(nsrcAlig.size(), 0);

      for (PositionIndex i = 1; i <= nsrc.size() || i <= nsrcAlig.size(); ++i)
      {
        if (i <= nsrc.size())
        {
          WordIndex s = nsrc[i - 1];
          lexTable->setLexDenom(s, 0);
          if (s >= insertBuffer.size())
            insertBuffer.resize((size_t)s + 1);
          for (const WordIndex t : trg)
            insertBuffer[s].push_back(t);
          insertBufferItems += trg.size();
        }

        if (i <= nsrcAlig.size())
        {
          aligTable.setAligNumer(asHmm0, i, SMALL_LG_NUM);

          aSourceHmm asHmm;
          asHmm.prev_i = i;
          asHmm.slen = slen;
          aligTable.setAligDenom(asHmm, 0);
          for (PositionIndex i2 = 1; i2 <= nsrcAlig.size(); ++i2)
          {
            if (isValidAlig(i, slen, i2))
              aligTable.setAligNumer(asHmm, i2, SMALL_LG_NUM);
          }
          AligCountsEntry& elem = aligCounts[asHmm];
          if (elem.size() < nsrcAlig.size())
            elem.resize(nsrcAlig.size(), 0);
        }
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

void _incrHmmAligModel::addTranslationOptions(vector<vector<WordIndex>>& insertBuffer)
{
  WordIndex maxSrcWordIndex = (WordIndex)insertBuffer.size() - 1;
  if (maxSrcWordIndex >= lexCounts.size())
    lexCounts.resize((size_t)maxSrcWordIndex + 1);
  lexTable->reserveSpace(maxSrcWordIndex);

#pragma omp parallel for schedule(dynamic)
  for (int s = 0; s < (int)insertBuffer.size(); ++s)
  {
    for (WordIndex t : insertBuffer[s])
    {
      lexCounts[s][t] = 0;
      lexTable->setLexNumer(s, t, SMALL_LG_NUM);
    }
    insertBuffer[s].clear();
  }
}

void _incrHmmAligModel::batchUpdateCounts(const SentPairCont& pairs)
{
#pragma omp parallel for schedule(dynamic)
  for (int line_idx = 0; line_idx < (int)pairs.size(); ++line_idx)
  {
    Sentence src = pairs[line_idx].first;
    Sentence nsrc = extendWithNullWord(src);
    Sentence nsrcAlig = extendWithNullWordAlig(src);
    Sentence trg = pairs[line_idx].second;

    PositionIndex slen = (PositionIndex)src.size();

    // Calculate alpha and beta matrices
    vector<vector<double>> lexLogProbs;
    vector<vector<double>> alphaMatrix;
    vector<vector<double>> betaMatrix;
    calcAlphaBetaMatrices(nsrc, trg, slen, lexLogProbs, alphaMatrix, betaMatrix);

    vector<double> lexNums(nsrc.size() + 1);
    vector<double> innerAligNums(nsrcAlig.size() + 1);
    vector<vector<double>> aligNums(nsrcAlig.size() + 1, innerAligNums);
    for (PositionIndex j = 1; j <= trg.size(); ++j)
    {
      double lexSum = INVALID_ANJI_VAL;
      double aligSum = INVALID_ANJM1IP_ANJI_VAL;
      for (PositionIndex i = 1; i <= nsrc.size() || i <= nsrcAlig.size(); ++i)
      {
        if (i <= nsrc.size())
        {
          // Obtain numerator
          lexNums[i] = calc_lanji_num(i, j, alphaMatrix, betaMatrix);

          // Add contribution to sum
          lexSum = lexSum == INVALID_ANJI_VAL ? lexNums[i] : MathFuncs::lns_sumlog(lexSum, lexNums[i]);
        }
        if (i <= nsrcAlig.size())
        {
          aligNums[i][0] = 0;
          if (j == 1)
          {
            // Obtain numerator
            if (isNullAlig(0, slen, i))
            {
              if (isFirstNullAligPar(0, slen, i))
                aligNums[i][0] = calc_lanjm1ip_anji_num_je1(slen, i, lexLogProbs, betaMatrix);
              else
                aligNums[i][0] = aligNums[slen + 1][0];
            }
            else
            {
              aligNums[i][0] = calc_lanjm1ip_anji_num_je1(slen, i, lexLogProbs, betaMatrix);
            }

            // Add contribution to sum
            aligSum = aligSum == INVALID_ANJI_VAL ? aligNums[i][0] : MathFuncs::lns_sumlog(aligSum, aligNums[i][0]);
          }
          else
          {
            for (PositionIndex ip = 1; ip <= nsrcAlig.size(); ++ip)
            {
              // Obtain numerator
              if (isValidAlig(ip, slen, i))
              {
                aligNums[i][ip] = calc_lanjm1ip_anji_num_jg1(ip, slen, i, j, lexLogProbs, alphaMatrix, betaMatrix);
              }
              else
              {
                aligNums[i][ip] = SMALL_LG_NUM;
              }

              // Add contribution to sum
              aligSum = aligSum == INVALID_ANJM1IP_ANJI_VAL ? aligNums[i][ip]
                                                            : MathFuncs::lns_sumlog(aligSum, aligNums[i][ip]);
            }
          }
        }
      }
      for (PositionIndex i = 1; i <= nsrc.size() || i <= nsrcAlig.size(); ++i)
      {
        if (i <= nsrc.size())
        {
          // Obtain expected value
          double logLexCount = lexNums[i] - lexSum;
          // Smooth expected value
          if (logLexCount > ExpValLogMax)
            logLexCount = ExpValLogMax;
          if (logLexCount < ExpValLogMin)
            logLexCount = ExpValLogMin;

          // Store expected value
          WordIndex s = nsrc[i - 1];
          WordIndex t = trg[j - 1];
#pragma omp atomic
          lexCounts[s].find(t)->second += exp(logLexCount);
        }

        if (i <= nsrcAlig.size())
        {
          if (j == 1)
          {
            // Obtain expected value
            double logAligCount = aligNums[i][0] - aligSum;
            // Smooth expected value
            if (logAligCount > ExpValLogMax)
              logAligCount = ExpValLogMax;
            if (logAligCount < ExpValLogMin)
              logAligCount = ExpValLogMin;

            // Store expected value
            aSourceHmm asHmm;
            asHmm.prev_i = 0;
            asHmm.slen = slen;
#pragma omp atomic
            aligCounts[asHmm][i - 1] += exp(logAligCount);
          }
          else
          {
            for (PositionIndex ip = 1; ip <= nsrcAlig.size(); ++ip)
            {
              // Obtain information about alignment
              if (isValidAlig(ip, slen, i))
              {
                // Obtain expected value
                double aligCount = aligNums[i][ip] - aligSum;
                // Smooth expected value
                if (aligCount > ExpValLogMax)
                  aligCount = ExpValLogMax;
                if (aligCount < ExpValLogMin)
                  aligCount = ExpValLogMin;

                // Store expected value
                aSourceHmm asHmm;
                asHmm.prev_i = ip;
                asHmm.slen = slen;
#pragma omp atomic
                aligCounts[asHmm][i - 1] += exp(aligCount);
              }
            }
          }
        }
      }
    }
  }
}

void _incrHmmAligModel::batchMaximizeProbs()
{
#pragma omp parallel for schedule(dynamic)
  for (int s = 0; s < (int)lexCounts.size(); ++s)
  {
    double denom = 0;
    LexCountsEntry& elem = lexCounts[s];
    for (LexCountsEntry::iterator it = elem.begin(); it != elem.end(); ++it)
    {
      double numer = it->second;
      if (variationalBayes)
        numer += alpha;
      denom += numer;
      lexTable->setLexNumer(s, it->first, (float)log(numer));
      it->second = 0.0;
    }
    if (denom == 0)
      denom = 1;
    lexTable->setLexDenom(s, (float)log(denom));
  }

#pragma omp parallel for schedule(dynamic)
  for (int asHmmIndex = 0; asHmmIndex < (int)aligCounts.size(); ++asHmmIndex)
  {
    double denom = 0;
    const pair<aSourceHmm, AligCountsEntry>& p = aligCounts.getAt(asHmmIndex);
    const aSourceHmm& asHmm = p.first;
    AligCountsEntry& elem = const_cast<AligCountsEntry&>(p.second);
    for (PositionIndex i = 1; i <= elem.size() || i <= asHmm.slen * 2; ++i)
    {
      if (i <= elem.size())
      {
        double numer = elem[i - 1];
        denom += numer;
        float logNumer = (float)log(numer);
        aligTable.setAligNumer(asHmm, i, logNumer);
        elem[i - 1] = 0.0;
      }
      cachedAligLogProbs.set(asHmm.prev_i, asHmm.slen, i, (double)CACHED_HMM_ALIG_LGPROB_VIT_INVALID_VAL);
    }
    if (denom == 0)
      denom = 1;
    float logDenom = (float)log(denom);
    aligTable.setAligDenom(asHmm, logDenom);
  }
}

pair<double, double> _incrHmmAligModel::loglikelihoodForPairRange(pair<unsigned int, unsigned int> sentPairRange,
                                                                  int verbosity)
{
  double loglikelihood = 0;
  unsigned int numSents = 0;

  for (unsigned int n = sentPairRange.first; n <= sentPairRange.second; ++n)
  {
    if (verbosity)
      cerr << "* Calculating log-likelihood for sentence " << n << endl;
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

pair<double, double> _incrHmmAligModel::vitLoglikelihoodForPairRange(pair<unsigned int, unsigned int> sentPairRange,
                                                                     int verbosity)
{
  double vitLoglikelihood = 0;
  unsigned int numSents = 0;

  for (unsigned int n = sentPairRange.first; n <= sentPairRange.second; ++n)
  {
    if (verbosity)
      cerr << "* Calculating log-likelihood for sentence " << n << endl;
    // Add Viterbi log-likelihood
    std::vector<WordIndex> nthSrcSent = getSrcSent(n);
    std::vector<WordIndex> nthTrgSent = getTrgSent(n);
    if (sentenceLengthIsOk(nthSrcSent) && sentenceLengthIsOk(nthTrgSent))
    {
      WordAligMatrix bestWaMatrix;
      vitLoglikelihood += (double)obtainBestAlignment(nthSrcSent, nthTrgSent, bestWaMatrix);
      ++numSents;
    }
  }
  return make_pair(vitLoglikelihood, vitLoglikelihood / (double)numSents);
}

void _incrHmmAligModel::setLexSmIntFactor(double _lexSmoothInterpFactor, int verbose)
{
  lexSmoothInterpFactor = _lexSmoothInterpFactor;
  if (verbose)
    cerr << "Lexical smoothing interpolation factor has been set to " << lexSmoothInterpFactor << endl;
}

Prob _incrHmmAligModel::pts(WordIndex s, WordIndex t)
{
  return exp((double)logpts(s, t));
}

double _incrHmmAligModel::unsmoothed_logpts(WordIndex s, WordIndex t)
{
  bool found;
  double numer;

  numer = lexTable->getLexNumer(s, t, found);
  if (found)
  {
    // lexNumer for pair s,t exists
    double denom;

    denom = lexTable->getLexDenom(s, found);
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

LgProb _incrHmmAligModel::logpts(WordIndex s, WordIndex t)
{
  LgProb lexLgProb = (LgProb)log(1.0 - lexSmoothInterpFactor) + unsmoothed_logpts(s, t);
  LgProb smoothLgProb = log(lexSmoothInterpFactor) + log(1.0 / (double)(getTrgVocabSize()));
  return MathFuncs::lns_sumlog(lexLgProb, smoothLgProb);
}

void _incrHmmAligModel::setAlSmIntFactor(double _aligSmoothInterpFactor, int verbose)
{
  aligSmoothInterpFactor = _aligSmoothInterpFactor;
  if (verbose)
    cerr << "Alignment smoothing interpolation factor has been set to " << aligSmoothInterpFactor << endl;
}

Prob _incrHmmAligModel::aProb(PositionIndex prev_i, PositionIndex slen, PositionIndex i)
{
  return exp((double)logaProb(prev_i, slen, i));
}

double _incrHmmAligModel::unsmoothed_logaProb(PositionIndex prev_i, PositionIndex slen, PositionIndex i)
{
  HmmAligInfo hmmAligInfo;
  getHmmAligInfo(prev_i, slen, i, hmmAligInfo);
  if (!hmmAligInfo.validAlig)
  {
    return SMALL_LG_NUM;
  }
  else
  {
    bool found;
    double numer;
    aSourceHmm asHmm;
    asHmm.prev_i = hmmAligInfo.modified_ip;
    asHmm.slen = slen;

    if (hmmAligInfo.nullAlig)
    {
      nullAligSpecialPar(prev_i, slen, asHmm, i);
    }

    numer = aligTable.getAligNumer(asHmm, i, found);
    if (found)
    {
      // aligNumer for pair asHmm,i exists
      double denom;
      denom = aligTable.getAligDenom(asHmm, found);
      if (!found)
        return SMALL_LG_NUM;
      else
      {
        return numer - denom;
      }
    }
    else
    {
      // aligNumer for pair asHmm,i does not exist
      return SMALL_LG_NUM;
    }
  }
}

double _incrHmmAligModel::cached_logaProb(PositionIndex prev_i, PositionIndex slen, PositionIndex i)
{
  double d = cachedAligLogProbs.get(prev_i, slen, i);
  if (d < CACHED_HMM_ALIG_LGPROB_VIT_INVALID_VAL)
  {
    return d;
  }
  else
  {
    double d = (double)logaProb(prev_i, slen, i);
    cachedAligLogProbs.set(prev_i, slen, i, d);
    return d;
  }
}

void _incrHmmAligModel::nullAligSpecialPar(unsigned int ip, unsigned int slen, aSourceHmm& asHmm, unsigned int& i)
{
  asHmm.slen = slen;
  if (ip == 0)
  {
    asHmm.prev_i = 0;
    i = slen + 1;
  }
  else
  {
    if (ip > slen)
      asHmm.prev_i = ip - slen;
    else
      asHmm.prev_i = ip;

    i = asHmm.prev_i + slen;
  }
}

LgProb _incrHmmAligModel::logaProb(PositionIndex prev_i, PositionIndex slen, PositionIndex i)
{
  LgProb lp = unsmoothed_logaProb(prev_i, slen, i);
  if (isValidAlig(prev_i, slen, i))
  {
    LgProb aligLgProb = (LgProb)log(1.0 - aligSmoothInterpFactor) + lp;
    LgProb smoothLgProb;
    if (prev_i == 0)
    {
      smoothLgProb = log(aligSmoothInterpFactor) + log(1.0 / (double)(2 * slen));
    }
    else
    {
      smoothLgProb = log(aligSmoothInterpFactor) + log(1.0 / (double)(slen + 1));
    }
    return MathFuncs::lns_sumlog(aligLgProb, smoothLgProb);
  }
  else
    return lp;
}

vector<WordIndex> _incrHmmAligModel::getSrcSent(unsigned int n)
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

vector<WordIndex> _incrHmmAligModel::extendWithNullWord(const vector<WordIndex>& srcWordIndexVec)
{
  // Initialize result using srcWordIndexVec
  vector<WordIndex> result = srcWordIndexVec;

  // Add NULL words
  WordIndex nullWidx = stringToSrcWordIndex(NULL_WORD_STR);
  for (unsigned int i = 0; i < srcWordIndexVec.size(); ++i)
    result.push_back(nullWidx);

  return result;
}

vector<WordIndex> _incrHmmAligModel::extendWithNullWordAlig(const vector<WordIndex>& srcWordIndexVec)
{
  return extendWithNullWord(srcWordIndexVec);
}

PositionIndex _incrHmmAligModel::getSrcLen(const vector<WordIndex>& nsrcWordIndexVec)
{
  unsigned int result = 0;
  WordIndex nullWidx = stringToSrcWordIndex(NULL_WORD_STR);
  for (unsigned int i = 0; i < nsrcWordIndexVec.size(); ++i)
  {
    if (nsrcWordIndexVec[i] != nullWidx)
      ++result;
  }
  return result;
}

vector<WordIndex> _incrHmmAligModel::getTrgSent(unsigned int n)
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

bool _incrHmmAligModel::sentenceLengthIsOk(const vector<WordIndex> sentence)
{
  if (sentence.empty() || sentence.size() > HMM_SWM_MAX_SENT_LENGTH)
    return false;
  else
    return true;
}

bool _incrHmmAligModel::loadLexSmIntFactor(const char* lexSmIntFactorFile, int verbose)
{
  if (verbose)
    cerr << "Loading file with lexical smoothing interpolation factor from " << lexSmIntFactorFile << endl;

  AwkInputStream awk;

  if (awk.open(lexSmIntFactorFile) == THOT_ERROR)
  {
    if (verbose)
      cerr << "Error in file with lexical smoothing interpolation factor, file " << lexSmIntFactorFile
           << " does not exist. Assuming default value." << endl;
    setLexSmIntFactor(DEFAULT_LEX_SMOOTH_INTERP_FACTOR, verbose);
    return THOT_OK;
  }
  else
  {
    if (awk.getln())
    {
      if (awk.NF == 1)
      {
        setLexSmIntFactor((Prob)atof(awk.dollar(1).c_str()), verbose);
        return THOT_OK;
      }
      else
      {
        if (verbose)
          cerr << "Error: anomalous .lsifactor file, " << lexSmIntFactorFile << endl;
        return THOT_ERROR;
      }
    }
    else
    {
      if (verbose)
        cerr << "Error: anomalous .lsifactor file, " << lexSmIntFactorFile << endl;
      return THOT_ERROR;
    }
  }
}

bool _incrHmmAligModel::printLexSmIntFactor(const char* lexSmIntFactorFile, int verbose)
{
  ofstream outF;
  outF.open(lexSmIntFactorFile, ios::out);
  if (!outF)
  {
    if (verbose)
      cerr << "Error while printing file with lexical smoothing interpolation factor." << endl;
    return THOT_ERROR;
  }
  else
  {
    outF << lexSmoothInterpFactor << endl;
    return THOT_OK;
  }
}

bool _incrHmmAligModel::loadAlSmIntFactor(const char* alSmIntFactorFile, int verbose)
{
  if (verbose)
    cerr << "Loading file with alignment smoothing interpolation factor from " << alSmIntFactorFile << endl;

  AwkInputStream awk;

  if (awk.open(alSmIntFactorFile) == THOT_ERROR)
  {
    if (verbose)
      cerr << "Error in file with alignment smoothing interpolation factor, file " << alSmIntFactorFile
           << " does not exist. Assuming default value." << endl;
    setAlSmIntFactor(DEFAULT_ALIG_SMOOTH_INTERP_FACTOR, verbose);
    return THOT_OK;
  }
  else
  {
    if (awk.getln())
    {
      if (awk.NF == 1)
      {
        setAlSmIntFactor((Prob)atof(awk.dollar(1).c_str()), verbose);
        return THOT_OK;
      }
      else
      {
        if (verbose)
          cerr << "Error: anomalous .asifactor file, " << alSmIntFactorFile << endl;
        return THOT_ERROR;
      }
    }
    else
    {
      if (verbose)
        cerr << "Error: anomalous .asifactor file, " << alSmIntFactorFile << endl;
      return THOT_ERROR;
    }
  }
}

bool _incrHmmAligModel::printAlSmIntFactor(const char* alSmIntFactorFile, int verbose)
{
  ofstream outF;
  outF.open(alSmIntFactorFile, ios::out);
  if (!outF)
  {
    if (verbose)
      cerr << "Error while printing file with alignment smoothing interpolation factor." << endl;
    return THOT_ERROR;
  }
  else
  {
    outF << aligSmoothInterpFactor << endl;
    return THOT_OK;
  }
}

void _incrHmmAligModel::calcNewLocalSuffStats(pair<unsigned int, unsigned int> sentPairRange, int verbosity)
{
  // Iterate over the training samples
  for (unsigned int n = sentPairRange.first; n <= sentPairRange.second; ++n)
  {
    // Init vars for n'th sample
    vector<WordIndex> srcSent = getSrcSent(n);
    vector<WordIndex> nsrcSent = extendWithNullWord(srcSent);
    vector<WordIndex> trgSent = getTrgSent(n);

    // Do not process sentence pair if sentences are empty or exceed the maximum length
    if (sentenceLengthIsOk(srcSent) && sentenceLengthIsOk(trgSent))
    {
      Count weight;
      sentenceHandler.getCount(n, weight);

      PositionIndex slen = (PositionIndex)srcSent.size();

      // Make room for data structure to cache alignment log-probs
      cachedAligLogProbs.makeRoomGivenSrcSentLen(slen);

      // Calculate alpha and beta matrices
      vector<vector<double>> lexLogProbs;
      vector<vector<double>> alphaMatrix;
      vector<vector<double>> betaMatrix;
      calcAlphaBetaMatrices(nsrcSent, trgSent, slen, lexLogProbs, alphaMatrix, betaMatrix);

      // Calculate sufficient statistics for anji values
      calc_lanji(n, nsrcSent, trgSent, slen, weight, alphaMatrix, betaMatrix);

      // Calculate sufficient statistics for anjm1ip_anji values
      calc_lanjm1ip_anji(n, extendWithNullWordAlig(srcSent), trgSent, slen, weight, lexLogProbs, alphaMatrix,
                         betaMatrix);
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
  // Clear cached alignment log probs
  cachedAligLogProbs.clear();
}

void _incrHmmAligModel::calcNewLocalSuffStatsVit(pair<unsigned int, unsigned int> sentPairRange, int verbosity)
{
  // Define variable to cache alignment log probs
  CachedHmmAligLgProb cached_logap;

  // Iterate over the training samples
  for (unsigned int n = sentPairRange.first; n <= sentPairRange.second; ++n)
  {
    // Init vars for n'th sample
    vector<WordIndex> srcSent = getSrcSent(n);
    vector<WordIndex> nsrcSent = extendWithNullWord(srcSent);
    vector<WordIndex> trgSent = getTrgSent(n);

    // Do not process sentence pair if sentences are empty or exceed the maximum length
    if (sentenceLengthIsOk(srcSent) && sentenceLengthIsOk(trgSent))
    {
      Count weight;
      sentenceHandler.getCount(n, weight);

      PositionIndex slen = (PositionIndex)srcSent.size();

      // Execute Viterbi algorithm
      vector<vector<double>> vitMatrix;
      vector<vector<PositionIndex>> predMatrix;
      viterbiAlgorithmCached(nsrcSent, trgSent, cached_logap, vitMatrix, predMatrix);

      // Obtain Viterbi alignment
      vector<PositionIndex> bestAlig;
      bestAligGivenVitMatricesRaw(vitMatrix, predMatrix, bestAlig);

      // Calculate sufficient statistics for anji values
      calc_lanji_vit(n, nsrcSent, trgSent, bestAlig, weight);

      // Calculate sufficient statistics for anjm1ip_anji values
      calc_lanjm1ip_anji_vit(n, extendWithNullWordAlig(srcSent), trgSent, slen, bestAlig, weight);
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

void _incrHmmAligModel::calcAlphaBetaMatrices(const vector<WordIndex>& nsrcSent, const vector<WordIndex>& trgSent,
                                              PositionIndex slen, vector<vector<double>>& lexLogProbs,
                                              vector<vector<double>>& alphaMatrix, vector<vector<double>>& betaMatrix)
{
  // Create data structure to cache lexical log-probs
  lexLogProbs.clear();
  vector<double> innerLexLogProbs(trgSent.size() + 1, SMALL_LG_NUM);
  lexLogProbs.resize(nsrcSent.size() + 1, innerLexLogProbs);

  // Initialize alphaMatrix
  alphaMatrix.clear();
  vector<double> innerMatrix(trgSent.size() + 1, 0.0);
  alphaMatrix.resize(nsrcSent.size() + 1, innerMatrix);

  // Fill alphaMatrix
  for (PositionIndex j = 1; j <= trgSent.size(); ++j)
  {
    for (PositionIndex i = 1; i <= nsrcSent.size(); ++i)
    {
      lexLogProbs[i][j] = logpts(nsrcSent[i - 1], trgSent[j - 1]);

      if (j == 1)
      {
        alphaMatrix[i][j] = cached_logaProb(0, slen, i) + lexLogProbs[i][j];
      }
      else
      {
        for (PositionIndex i_tilde = 1; i_tilde <= nsrcSent.size(); ++i_tilde)
        {
          double lp = alphaMatrix[i_tilde][j - 1] + cached_logaProb(i_tilde, slen, i) + lexLogProbs[i][j];
          if (i_tilde == 1)
            alphaMatrix[i][j] = lp;
          else
            alphaMatrix[i][j] = MathFuncs::lns_sumlog(lp, alphaMatrix[i][j]);
        }
      }
    }
  }

  // Initialize betaMatrix
  betaMatrix.clear();
  betaMatrix.resize(nsrcSent.size() + 1, innerMatrix);

  // Fill betaMatrix
  for (PositionIndex j = trgSent.size(); j >= 1; --j)
  {
    for (PositionIndex i = 1; i <= nsrcSent.size(); ++i)
    {
      if (j == trgSent.size())
      {
        betaMatrix[i][j] = Log1;
      }
      else
      {
        for (PositionIndex i_tilde = 1; i_tilde <= nsrcSent.size(); ++i_tilde)
        {
          double lp = betaMatrix[i_tilde][j + 1] + cached_logaProb(i, slen, i_tilde) + lexLogProbs[i_tilde][j + 1];
          if (i_tilde == 1)
            betaMatrix[i][j] = lp;
          else
            betaMatrix[i][j] = MathFuncs::lns_sumlog(lp, betaMatrix[i][j]);
        }
      }
    }
  }
}

void _incrHmmAligModel::calc_lanji(unsigned int n, const vector<WordIndex>& nsrcSent, const vector<WordIndex>& trgSent,
                                   PositionIndex slen, const Count& weight, const vector<vector<double>>& alphaMatrix,
                                   const vector<vector<double>>& betaMatrix)
{
  // Initialize data structures
  unsigned int mapped_n;
  lanji.init_nth_entry(n, nsrcSent.size(), trgSent.size(), mapped_n);

  unsigned int n_aux = 1;
  unsigned int mapped_n_aux;
  lanji_aux.init_nth_entry(n_aux, nsrcSent.size(), trgSent.size(), mapped_n_aux);

  vector<double> numVec(nsrcSent.size() + 1, 0);

  // Calculate new estimation of lanji
  for (unsigned int j = 1; j <= trgSent.size(); ++j)
  {
    // Obtain sum_lanji_num_forall_s
    double sum_lanji_num_forall_s = INVALID_ANJI_VAL;
    for (unsigned int i = 1; i <= nsrcSent.size(); ++i)
    {
      // Obtain numerator
      double d = calc_lanji_num(i, j, alphaMatrix, betaMatrix);

      // Add contribution to sum
      if (sum_lanji_num_forall_s == INVALID_ANJI_VAL)
        sum_lanji_num_forall_s = d;
      else
        sum_lanji_num_forall_s = MathFuncs::lns_sumlog(sum_lanji_num_forall_s, d);
      // Store num in numVec
      numVec[i] = d;
    }
    // Set value of lanji_aux
    for (unsigned int i = 1; i <= nsrcSent.size(); ++i)
    {
      // Obtain expected value
      double lanji_val = numVec[i] - sum_lanji_num_forall_s;
      // Smooth expected value
      if (lanji_val > ExpValLogMax)
        lanji_val = ExpValLogMax;
      if (lanji_val < ExpValLogMin)
        lanji_val = ExpValLogMin;
      // Store expected value
      lanji_aux.set_fast(mapped_n_aux, j, i, lanji_val);
    }
  }
  // Gather lexical sufficient statistics
  gatherLexSuffStats(mapped_n, mapped_n_aux, nsrcSent, trgSent, weight);

  // clear lanji_aux data structure
  lanji_aux.clear();
}

void _incrHmmAligModel::calc_lanji_vit(unsigned int n, const vector<WordIndex>& nsrcSent,
                                       const vector<WordIndex>& trgSent, const vector<PositionIndex>& bestAlig,
                                       const Count& weight)
{
  // Initialize data structures
  unsigned int mapped_n;
  lanji.init_nth_entry(n, nsrcSent.size(), trgSent.size(), mapped_n);

  unsigned int n_aux = 1;
  unsigned int mapped_n_aux;
  lanji_aux.init_nth_entry(n_aux, nsrcSent.size(), trgSent.size(), mapped_n_aux);

  // Calculate new estimation of lanji
  for (unsigned int j = 1; j <= trgSent.size(); ++j)
  {
    // Set value of lanji_aux
    for (unsigned int i = 1; i <= nsrcSent.size(); ++i)
    {
      if (bestAlig[j - 1] == i)
      {
        // Obtain expected value
        double lanji_val = 0;
        // Store expected value
        lanji_aux.set_fast(mapped_n_aux, j, i, lanji_val);
      }
    }
  }

  // Gather lexical sufficient statistics
  gatherLexSuffStats(mapped_n, mapped_n_aux, nsrcSent, trgSent, weight);

  // clear lanji_aux data structure
  lanji_aux.clear();
}

void _incrHmmAligModel::gatherLexSuffStats(unsigned int mapped_n, unsigned int mapped_n_aux,
                                           const vector<WordIndex>& nsrcSent, const vector<WordIndex>& trgSent,
                                           const Count& weight)
{
  // Gather lexical sufficient statistics
  for (unsigned int j = 1; j <= trgSent.size(); ++j)
  {
    for (unsigned int i = 1; i <= nsrcSent.size(); ++i)
    {
      // Reestimate lexical parameters
      incrUpdateCountsLex(mapped_n, mapped_n_aux, i, j, nsrcSent, trgSent, weight);

      // Update lanji
      lanji.set_fast(mapped_n, j, i, lanji_aux.get_invlogp(mapped_n_aux, j, i));
    }
  }
}

void _incrHmmAligModel::incrUpdateCountsLex(unsigned int mapped_n, unsigned int mapped_n_aux, PositionIndex i,
                                         PositionIndex j, const vector<WordIndex>& nsrcSent,
                                         const vector<WordIndex>& trgSent, const Count& weight)
{
  // Init vars
  float curr_lanji = lanji.get_fast(mapped_n, j, i);
  float weighted_curr_lanji = SMALL_LG_NUM;
  if (curr_lanji != INVALID_ANJI_VAL)
  {
    weighted_curr_lanji = (float)log((float)weight) + curr_lanji;
    if (weighted_curr_lanji < SMALL_LG_NUM)
      weighted_curr_lanji = SMALL_LG_NUM;
  }

  float weighted_new_lanji = (float)log((float)weight) + lanji_aux.get_invlogp_fast(mapped_n_aux, j, i);
  if (weighted_new_lanji < SMALL_LG_NUM)
    weighted_new_lanji = SMALL_LG_NUM;

  WordIndex s = nsrcSent[i - 1];
  WordIndex t = trgSent[j - 1];

  // Store contributions
  while (incrLexCounts.size() <= s)
  {
    IncrLexCountsEntry lexAuxVarElem;
    incrLexCounts.push_back(lexAuxVarElem);
  }

  IncrLexCountsEntry::iterator lexAuxVarElemIter = incrLexCounts[s].find(t);
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
    incrLexCounts[s][t] = std::make_pair(weighted_curr_lanji, weighted_new_lanji);
  }
}

void _incrHmmAligModel::calc_lanjm1ip_anji(unsigned int n, const vector<WordIndex>& nsrcSent,
                                           const vector<WordIndex>& trgSent, PositionIndex slen, const Count& weight,
                                           const vector<vector<double>>& lexLogProbs,
                                           const vector<vector<double>>& alphaMatrix,
                                           const vector<vector<double>>& betaMatrix)
{
  // Initialize data structures
  unsigned int mapped_n;
  lanjm1ip_anji.init_nth_entry(n, nsrcSent.size(), trgSent.size(), mapped_n);

  unsigned int n_aux = 1;
  unsigned int mapped_n_aux;
  lanjm1ip_anji_aux.init_nth_entry(n_aux, nsrcSent.size(), trgSent.size(), mapped_n_aux);

  vector<double> numVec(nsrcSent.size() + 1, 0);
  vector<vector<double>> numVecVec(nsrcSent.size() + 1, numVec);

  // Calculate new estimation of lanjm1ip_anji
  for (unsigned int j = 1; j <= trgSent.size(); ++j)
  {
    // Obtain sum_lanjm1ip_anji_num_forall_i_ip
    double sum_lanjm1ip_anji_num_forall_i_ip = INVALID_ANJM1IP_ANJI_VAL;

    for (unsigned int i = 1; i <= nsrcSent.size(); ++i)
    {
      numVecVec[i][0] = 0;
      if (j == 1)
      {
        // Obtain numerator

        // Obtain information about alignment
        bool nullAlig = isNullAlig(0, slen, i);
        double d;
        if (nullAlig)
        {
          if (isFirstNullAligPar(0, slen, i))
            d = calc_lanjm1ip_anji_num_je1(slen, i, lexLogProbs, betaMatrix);
          else
            d = numVecVec[slen + 1][0];
        }
        else
          d = calc_lanjm1ip_anji_num_je1(slen, i, lexLogProbs, betaMatrix);
        // Add contribution to sum
        if (sum_lanjm1ip_anji_num_forall_i_ip == INVALID_ANJM1IP_ANJI_VAL)
          sum_lanjm1ip_anji_num_forall_i_ip = d;
        else
          sum_lanjm1ip_anji_num_forall_i_ip = MathFuncs::lns_sumlog(sum_lanjm1ip_anji_num_forall_i_ip, d);
        // Store num in numVec
        numVecVec[i][0] = d;
      }
      else
      {
        for (unsigned int ip = 1; ip <= nsrcSent.size(); ++ip)
        {
          // Obtain numerator

          // Obtain information about alignment
          double d;
          bool validAlig = isValidAlig(ip, slen, i);
          if (!validAlig)
          {
            d = SMALL_LG_NUM;
          }
          else
          {
            d = calc_lanjm1ip_anji_num_jg1(ip, slen, i, j, lexLogProbs, alphaMatrix, betaMatrix);
          }
          // Add contribution to sum
          if (sum_lanjm1ip_anji_num_forall_i_ip == INVALID_ANJM1IP_ANJI_VAL)
            sum_lanjm1ip_anji_num_forall_i_ip = d;
          else
            sum_lanjm1ip_anji_num_forall_i_ip = MathFuncs::lns_sumlog(sum_lanjm1ip_anji_num_forall_i_ip, d);
          // Store num in numVec
          numVecVec[i][ip] = d;
        }
      }
    }
    // Set value of lanjm1ip_anji_aux
    for (unsigned int i = 1; i <= nsrcSent.size(); ++i)
    {
      double lanjm1ip_anji_val;
      if (j == 1)
      {
        // Obtain expected value
        lanjm1ip_anji_val = numVecVec[i][0] - sum_lanjm1ip_anji_num_forall_i_ip;
        // Smooth expected value
        if (lanjm1ip_anji_val > ExpValLogMax)
          lanjm1ip_anji_val = ExpValLogMax;
        if (lanjm1ip_anji_val < ExpValLogMin)
          lanjm1ip_anji_val = ExpValLogMin;
        // Store expected value
        lanjm1ip_anji_aux.set_fast(mapped_n_aux, j, i, 0, lanjm1ip_anji_val);
      }
      else
      {
        for (unsigned int ip = 1; ip <= nsrcSent.size(); ++ip)
        {
          // Obtain information about alignment
          bool validAlig = isValidAlig(ip, slen, i);
          if (validAlig)
          {
            // Obtain expected value
            lanjm1ip_anji_val = numVecVec[i][ip] - sum_lanjm1ip_anji_num_forall_i_ip;
            // Smooth expected value
            if (lanjm1ip_anji_val > ExpValLogMax)
              lanjm1ip_anji_val = ExpValLogMax;
            if (lanjm1ip_anji_val < ExpValLogMin)
              lanjm1ip_anji_val = ExpValLogMin;
            // Store expected value
            lanjm1ip_anji_aux.set_fast(mapped_n_aux, j, i, ip, lanjm1ip_anji_val);
          }
        }
      }
    }
  }
  // Gather alignment sufficient statistics
  gatherAligSuffStats(mapped_n, mapped_n_aux, nsrcSent, trgSent, slen, weight);

  // clear lanjm1ip_anji_aux data structure
  lanjm1ip_anji_aux.clear();
}

void _incrHmmAligModel::calc_lanjm1ip_anji_vit(unsigned int n, const vector<WordIndex>& nsrcSent,
                                               const vector<WordIndex>& trgSent, PositionIndex slen,
                                               const vector<PositionIndex>& bestAlig, const Count& weight)
{
  // Initialize data structures
  unsigned int mapped_n;
  lanjm1ip_anji.init_nth_entry(n, nsrcSent.size(), trgSent.size(), mapped_n);

  unsigned int n_aux = 1;
  unsigned int mapped_n_aux;
  lanjm1ip_anji_aux.init_nth_entry(n_aux, nsrcSent.size(), trgSent.size(), mapped_n_aux);

  // Calculate new estimation of lanjm1ip_anji
  for (unsigned int j = 1; j <= trgSent.size(); ++j)
  {
    for (unsigned int i = 1; i <= nsrcSent.size(); ++i)
    {
      if (j == 1)
      {
        if (bestAlig[0] == i)
        {
          double lanjm1ip_anji_val = 0;
          // Store expected value
          lanjm1ip_anji_aux.set_fast(mapped_n_aux, j, i, 0, lanjm1ip_anji_val);
        }
      }
      else
      {
        for (unsigned int ip = 1; ip <= nsrcSent.size(); ++ip)
        {
          PositionIndex aligModifiedIp = getModifiedIp(bestAlig[j - 2], slen, i);

          if (bestAlig[j - 1] == i && aligModifiedIp == ip)
          {
            double lanjm1ip_anji_val = 0;
            // Store expected value
            lanjm1ip_anji_aux.set_fast(mapped_n_aux, j, i, ip, lanjm1ip_anji_val);
          }
        }
      }
    }
  }

  // Gather alignment sufficient statistics
  gatherAligSuffStats(mapped_n, mapped_n_aux, nsrcSent, trgSent, slen, weight);

  // clear lanjm1ip_anji_aux data structure
  lanjm1ip_anji_aux.clear();
}

void _incrHmmAligModel::gatherAligSuffStats(unsigned int mapped_n, unsigned int mapped_n_aux,
                                            const vector<WordIndex>& nsrcSent, const vector<WordIndex>& trgSent,
                                            PositionIndex slen, const Count& weight)
{
  // Maximize alignment parameters
  for (unsigned int j = 1; j <= trgSent.size(); ++j)
  {
    for (unsigned int i = 1; i <= nsrcSent.size(); ++i)
    {
      if (j == 1)
      {
        // Reestimate alignment parameters
        incrUpdateCountsAlig(mapped_n, mapped_n_aux, slen, 0, i, j, weight);

        // Update lanjm1ip_anji
        lanjm1ip_anji.set_fast(mapped_n, j, i, 0, lanjm1ip_anji_aux.get_invlogp_fast(mapped_n_aux, j, i, 0));
      }
      else
      {
        for (unsigned int ip = 1; ip <= nsrcSent.size(); ++ip)
        {
          // Obtain information about alignment
          bool validAlig = isValidAlig(ip, slen, i);
          if (validAlig)
          {
            // Reestimate alignment parameters
            incrUpdateCountsAlig(mapped_n, mapped_n_aux, slen, ip, i, j, weight);
            // Update lanjm1ip_anji
            lanjm1ip_anji.set_fast(mapped_n, j, i, ip, lanjm1ip_anji_aux.get_invlogp_fast(mapped_n_aux, j, i, ip));
          }
        }
      }
    }
  }
}

void _incrHmmAligModel::incrUpdateCountsAlig(unsigned int mapped_n, unsigned int mapped_n_aux, PositionIndex slen,
                                          PositionIndex ip, PositionIndex i, PositionIndex j, const Count& weight)
{
  // Init vars
  float curr_lanjm1ip_anji = lanjm1ip_anji.get_fast(mapped_n, j, i, ip);
  float weighted_curr_lanjm1ip_anji = SMALL_LG_NUM;
  if (curr_lanjm1ip_anji != INVALID_ANJM1IP_ANJI_VAL)
  {
    weighted_curr_lanjm1ip_anji = (float)log((float)weight) + curr_lanjm1ip_anji;
    if (weighted_curr_lanjm1ip_anji < SMALL_LG_NUM)
      weighted_curr_lanjm1ip_anji = SMALL_LG_NUM;
  }

  float weighted_new_lanjm1ip_anji =
      (float)log((float)weight) + lanjm1ip_anji_aux.get_invlogp_fast(mapped_n_aux, j, i, ip);
  if (weighted_new_lanjm1ip_anji < SMALL_LG_NUM)
    weighted_new_lanjm1ip_anji = SMALL_LG_NUM;

  // Init aSourceHmm data structure
  aSourceHmm asHmm;
  asHmm.prev_i = ip;
  asHmm.slen = slen;

  // Gather local suff. statistics
  IncrAligCounts::iterator aligAuxVarIter = incrAligCounts.find(std::make_pair(asHmm, i));
  if (aligAuxVarIter != incrAligCounts.end())
  {
    if (weighted_curr_lanjm1ip_anji != SMALL_LG_NUM)
      aligAuxVarIter->second.first =
          MathFuncs::lns_sumlog_float(aligAuxVarIter->second.first, weighted_curr_lanjm1ip_anji);
    aligAuxVarIter->second.second =
        MathFuncs::lns_sumlog_float(aligAuxVarIter->second.second, weighted_new_lanjm1ip_anji);
  }
  else
  {
    incrAligCounts[std::make_pair(asHmm, i)] = std::make_pair(weighted_curr_lanjm1ip_anji, weighted_new_lanjm1ip_anji);
  }
}

bool _incrHmmAligModel::isFirstNullAligPar(PositionIndex ip, unsigned int slen, PositionIndex i)
{
  if (ip == 0)
  {
    if (i == slen + 1)
      return true;
    else
      return false;
  }
  else
  {
    if (i > slen && i - slen == ip)
      return true;
    else
      return false;
  }
}

double _incrHmmAligModel::calc_lanji_num(PositionIndex i, PositionIndex j, const vector<vector<double>>& alphaMatrix,
                                         const vector<vector<double>>& betaMatrix)
{
  double result = alphaMatrix[i][j] + betaMatrix[i][j];
  if (result < SMALL_LG_NUM)
    result = SMALL_LG_NUM;
  return result;
}

double _incrHmmAligModel::calc_lanjm1ip_anji_num_je1(PositionIndex slen, PositionIndex i,
                                                     const vector<vector<double>>& lexLogProbs,
                                                     const vector<vector<double>>& betaMatrix)
{
  double result = cached_logaProb(0, slen, i) + lexLogProbs[i][1] + betaMatrix[i][1];
  if (result < SMALL_LG_NUM)
    result = SMALL_LG_NUM;
  return result;
}

double _incrHmmAligModel::calc_lanjm1ip_anji_num_jg1(PositionIndex ip, PositionIndex slen, PositionIndex i,
                                                     PositionIndex j, const vector<vector<double>>& lexLogProbs,
                                                     const vector<vector<double>>& alphaMatrix,
                                                     const vector<vector<double>>& betaMatrix)
{
  double result = alphaMatrix[ip][j - 1] + cached_logaProb(ip, slen, i) + lexLogProbs[i][j] + betaMatrix[i][j];
  if (result < SMALL_LG_NUM)
    result = SMALL_LG_NUM;
  return result;
}

void _incrHmmAligModel::getHmmAligInfo(PositionIndex ip, unsigned int slen, PositionIndex i, HmmAligInfo& hmmAligInfo)
{
  hmmAligInfo.validAlig = isValidAlig(ip, slen, i);
  if (hmmAligInfo.validAlig)
  {
    hmmAligInfo.nullAlig = isNullAlig(ip, slen, i);
    hmmAligInfo.modified_ip = getModifiedIp(ip, slen, i);
  }
  else
  {
    hmmAligInfo.nullAlig = false;
    hmmAligInfo.modified_ip = ip;
  }
}

bool _incrHmmAligModel::isValidAlig(PositionIndex ip, unsigned int slen, PositionIndex i)
{
  if (i <= slen)
    return true;
  else
  {
    if (ip == 0)
      return true;
    i = i - slen;
    if (ip > slen)
      ip = ip - slen;
    if (i != ip)
      return false;
    else
      return true;
  }
}

bool _incrHmmAligModel::isNullAlig(PositionIndex ip, unsigned int slen, PositionIndex i)
{
  if (i <= slen)
    return false;
  else
  {
    if (ip == 0)
      return true;
    i = i - slen;
    if (ip > slen)
      ip = ip - slen;
    if (i != ip)
      return false;
    else
      return true;
  }
}

PositionIndex _incrHmmAligModel::getModifiedIp(PositionIndex ip, unsigned int slen, PositionIndex i)
{
  if (i <= slen && ip > slen)
  {
    return ip - slen;
  }
  else
    return ip;
}

void _incrHmmAligModel::incrMaximizeProbsLex()
{
  float initialNumer = variationalBayes ? (float)log(alpha) : SMALL_LG_NUM;
  // Update parameters
  for (unsigned int i = 0; i < incrLexCounts.size(); ++i)
  {
    for (IncrLexCountsEntry::iterator lexAuxVarElemIter = incrLexCounts[i].begin();
      lexAuxVarElemIter != incrLexCounts[i].end(); ++lexAuxVarElemIter)
    {
      WordIndex s = i;
      WordIndex t = lexAuxVarElemIter->first;
      float log_suff_stat_curr = lexAuxVarElemIter->second.first;
      float log_suff_stat_new = lexAuxVarElemIter->second.second;

      // Update parameters only if current and new sufficient statistics
      // are different
      if (log_suff_stat_curr != log_suff_stat_new)
      {
        // Obtain lexNumer for s,t
        bool numerFound;
        float numer = lexTable->getLexNumer(s, t, numerFound);
        if (!numerFound)
          numer = initialNumer;

        // Obtain lexDenom for s,t
        bool denomFound;
        float denom = lexTable->getLexDenom(s, denomFound);
        if (!denomFound)
          denom = SMALL_LG_NUM;

        // Obtain new sufficient statistics
        float new_numer = obtainLogNewSuffStat(numer, log_suff_stat_curr, log_suff_stat_new);
        float new_denom = denom;
        if (numerFound)
          new_denom = MathFuncs::lns_sublog_float(denom, numer);
        new_denom = MathFuncs::lns_sumlog_float(new_denom, new_numer);

        // Set lexical numerator and denominator
        lexTable->setLexNumDen(s, t, new_numer, new_denom);
      }
    }
  }
  // Clear auxiliary variables
  incrLexCounts.clear();
}

void _incrHmmAligModel::incrMaximizeProbsAlig()
{
  // Update parameters
  for (IncrAligCounts::iterator aligAuxVarIter = incrAligCounts.begin(); aligAuxVarIter != incrAligCounts.end(); ++aligAuxVarIter)
       ++aligAuxVarIter)
  {
    aSourceHmm asHmm = aligAuxVarIter->first.first;
    unsigned int i = aligAuxVarIter->first.second;
    float log_suff_stat_curr = aligAuxVarIter->second.first;
    float log_suff_stat_new = aligAuxVarIter->second.second;

    // Update parameters only if current and new sufficient statistics
    // are different
    if (log_suff_stat_curr != log_suff_stat_new)
    {
      // Obtain aligNumer
      bool found;
      float numer = aligTable.getAligNumer(asHmm, i, found);
      if (!found)
        numer = SMALL_LG_NUM;

      // Obtain aligDenom
      float denom = aligTable.getAligDenom(asHmm, found);
      if (!found)
        denom = SMALL_LG_NUM;

      // Obtain new sufficient statistics
      float new_numer = obtainLogNewSuffStat(numer, log_suff_stat_curr, log_suff_stat_new);
      float new_denom = MathFuncs::lns_sublog_float(denom, numer);
      new_denom = MathFuncs::lns_sumlog_float(new_denom, new_numer);

      // Set lexical numerator and denominator
      aligTable.setAligNumDen(asHmm, i, new_numer, new_denom);
    }
  }
  // Clear auxiliary variables
  incrAligCounts.clear();
}

float _incrHmmAligModel::obtainLogNewSuffStat(float lcurrSuffStat, float lLocalSuffStatCurr, float lLocalSuffStatNew)
{
  float lresult = MathFuncs::lns_sublog_float(lcurrSuffStat, lLocalSuffStatCurr);
  lresult = MathFuncs::lns_sumlog_float(lresult, lLocalSuffStatNew);
  return lresult;
}

Prob _incrHmmAligModel::sentLenProb(unsigned int slen, unsigned int tlen)
{
  return sentLengthModel.sentLenProb(slen, tlen);
}

LgProb _incrHmmAligModel::sentLenLgProb(unsigned int slen, unsigned int tlen)
{
  return sentLengthModel.sentLenLgProb(slen, tlen);
}

bool _incrHmmAligModel::getEntriesForSource(WordIndex s, NbestTableNode<WordIndex>& trgtn)
{
  set<WordIndex> transSet;
  bool ret = lexTable->getTransForSource(s, transSet);
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

LgProb _incrHmmAligModel::obtainBestAlignmentVecStrCached(const vector<string>& srcSentenceVector,
  const vector<string>& trgSentenceVector, CachedHmmAligLgProb& cached_logap, WordAligMatrix& bestWaMatrix)
                                                          CachedHmmAligLgProb& cached_logap,
                                                          WordAligMatrix& bestWaMatrix)
{
  LgProb lp;
  vector<WordIndex> srcSentIndexVector, trgSentIndexVector;

  srcSentIndexVector = strVectorToSrcIndexVector(srcSentenceVector);
  trgSentIndexVector = strVectorToTrgIndexVector(trgSentenceVector);
  lp = obtainBestAlignmentCached(srcSentIndexVector, trgSentIndexVector, cached_logap, bestWaMatrix);

  return lp;
}

LgProb _incrHmmAligModel::obtainBestAlignment(const vector<WordIndex>& srcSentIndexVector,
  const vector<WordIndex>& trgSentIndexVector, WordAligMatrix& bestWaMatrix)
{
  CachedHmmAligLgProb cached_logap;
  return obtainBestAlignmentCached(srcSentIndexVector, trgSentIndexVector, cached_logap, bestWaMatrix);
}

LgProb _incrHmmAligModel::obtainBestAlignmentCached(const std::vector<WordIndex>& srcSentIndexVector,
  const vector<WordIndex>& trgSentIndexVector, CachedHmmAligLgProb& cached_logap, WordAligMatrix& bestWaMatrix)
                                                    CachedHmmAligLgProb& cached_logap, WordAligMatrix& bestWaMatrix)
{
  if (sentenceLengthIsOk(srcSentIndexVector) && sentenceLengthIsOk(trgSentIndexVector))
  {
    // Obtain extended source vector
    vector<WordIndex> nSrcSentIndexVector = extendWithNullWord(srcSentIndexVector);
    // Call function to obtain best lgprob and viterbi alignment
    vector<vector<double>> vitMatrix;
    vector<vector<PositionIndex>> predMatrix;
    viterbiAlgorithmCached(nSrcSentIndexVector, trgSentIndexVector, cached_logap, vitMatrix, predMatrix);
    vector<PositionIndex> bestAlig;
    LgProb vit_lp = bestAligGivenVitMatrices(srcSentIndexVector.size(), vitMatrix, predMatrix, bestAlig);
    // Obtain best word alignment vector from the Viterbi matrices
    bestWaMatrix.init(srcSentIndexVector.size(), trgSentIndexVector.size());
    bestWaMatrix.putAligVec(bestAlig);

    // Calculate sentence length model lgprob
    LgProb slm_lp = sentLenLgProb(srcSentIndexVector.size(), trgSentIndexVector.size());

    return slm_lp + vit_lp;
  }
  else
  {
    bestWaMatrix.init(srcSentIndexVector.size(), trgSentIndexVector.size());
    return SMALL_LG_NUM;
  }
}

void _incrHmmAligModel::viterbiAlgorithm(const vector<WordIndex>& nSrcSentIndexVector,
                                         const vector<WordIndex>& trgSentIndexVector, vector<vector<double>>& vitMatrix,
                                         vector<vector<PositionIndex>>& predMatrix)
{
  CachedHmmAligLgProb cached_logap;
  viterbiAlgorithmCached(nSrcSentIndexVector, trgSentIndexVector, cached_logap, vitMatrix, predMatrix);
}

void _incrHmmAligModel::viterbiAlgorithmCached(const vector<WordIndex>& nSrcSentIndexVector,
                                               const vector<WordIndex>& trgSentIndexVector,
                                               CachedHmmAligLgProb& cached_logap, vector<vector<double>>& vitMatrix,
                                               vector<vector<PositionIndex>>& predMatrix)
{
  // Obtain slen
  PositionIndex slen = getSrcLen(nSrcSentIndexVector);

  // Clear matrices
  vitMatrix.clear();
  predMatrix.clear();

  // Make room for matrices
  vector<double> dVec;
  dVec.insert(dVec.begin(), trgSentIndexVector.size() + 1, SMALL_LG_NUM);
  vitMatrix.insert(vitMatrix.begin(), nSrcSentIndexVector.size() + 1, dVec);

  vector<PositionIndex> pidxVec;
  pidxVec.insert(pidxVec.begin(), trgSentIndexVector.size() + 1, 0);
  predMatrix.insert(predMatrix.begin(), nSrcSentIndexVector.size() + 1, pidxVec);

  // Fill matrices
  for (PositionIndex j = 1; j <= trgSentIndexVector.size(); ++j)
  {
    for (PositionIndex i = 1; i <= nSrcSentIndexVector.size(); ++i)
    {
      double logPts = logpts(nSrcSentIndexVector[i - 1], trgSentIndexVector[j - 1]);
      if (j == 1)
      {
        // Update cached alignment log-probs if required
        if (!cached_logap.isDefined(0, slen, i))
          cached_logap.set_boundary_check(0, slen, i, logaProb(0, slen, i));

        // Update matrices
        vitMatrix[i][j] = cached_logap.get(0, slen, i) + logPts;
        predMatrix[i][j] = 0;
      }
      else
      {
        for (PositionIndex i_tilde = 1; i_tilde <= nSrcSentIndexVector.size(); ++i_tilde)
        {
          // Update cached alignment log-probs if required
          if (!cached_logap.isDefined(i_tilde, slen, i))
            cached_logap.set_boundary_check(i_tilde, slen, i, logaProb(i_tilde, slen, i));

          // Update matrices
          double lp = vitMatrix[i_tilde][j - 1] + cached_logap.get(i_tilde, slen, i) + logPts;
          if (lp > vitMatrix[i][j])
          {
            vitMatrix[i][j] = lp;
            predMatrix[i][j] = i_tilde;
          }
        }
      }
    }
  }
}

double _incrHmmAligModel::bestAligGivenVitMatricesRaw(const vector<vector<double>>& vitMatrix,
                                                      const vector<vector<PositionIndex>>& predMatrix,
                                                      vector<PositionIndex>& bestAlig)
{
  if (vitMatrix.size() <= 1 || predMatrix.size() <= 1)
  {
    // if vitMatrix.size()==1 or predMatrix.size()==1, then the
    // source or the target sentences respectively were empty, so
    // there is no word alignment to be returned
    bestAlig.clear();
    return 0;
  }
  else
  {
    // Initialize bestAlig
    bestAlig.clear();
    bestAlig.insert(bestAlig.begin(), predMatrix[0].size() - 1, 0);

    // Find last word alignment
    PositionIndex last_j = predMatrix[1].size() - 1;
    double bestLgProb = vitMatrix[1][last_j];
    bestAlig[last_j - 1] = 1;
    for (unsigned int i = 2; i <= vitMatrix.size() - 1; ++i)
    {
      if (bestLgProb < vitMatrix[i][last_j])
      {
        bestLgProb = vitMatrix[i][last_j];
        bestAlig[last_j - 1] = i;
      }
    }

    // Retrieve remaining alignments
    for (unsigned int j = last_j; j > 1; --j)
    {
      bestAlig[j - 2] = predMatrix[bestAlig[j - 1]][j];
    }

    // Return best log-probability
    return bestLgProb;
  }
}

double _incrHmmAligModel::bestAligGivenVitMatrices(PositionIndex slen, const vector<vector<double>>& vitMatrix,
                                                   const vector<vector<PositionIndex>>& predMatrix,
                                                   vector<PositionIndex>& bestAlig)
{
  double LgProb = bestAligGivenVitMatricesRaw(vitMatrix, predMatrix, bestAlig);

  // Set null word alignments appropriately
  for (unsigned int j = 0; j < bestAlig.size(); ++j)
  {
    if (bestAlig[j] > slen)
      bestAlig[j] = NULL_WORD;
  }

  return LgProb;
}

LgProb _incrHmmAligModel::calcLgProbForAlig(const vector<WordIndex>& sSent, const vector<WordIndex>& tSent,
                                            const WordAligMatrix& aligMatrix, int verbose)
{
  // TO-DO (post-thesis)
  return 0;
}

LgProb _incrHmmAligModel::calcLgProb(const vector<WordIndex>& sSent, const vector<WordIndex>& tSent, int verbose)
{
  if (sentenceLengthIsOk(sSent) && sentenceLengthIsOk(tSent))
  {
    // Calculate sentence length model lgprob
    LgProb slp = sentLenLgProb(sSent.size(), tSent.size());

    // Obtain extended source vector
    vector<WordIndex> nSrcSentIndexVector = extendWithNullWord(sSent);

    // Calculate hmm lgprob
    LgProb flp = forwardAlgorithm(nSrcSentIndexVector, tSent, verbose);

    if (verbose)
      cerr << "lp= " << slp + flp << " ; slm_lp= " << slp << " ; lp-slm_lp= " << flp << endl;

    return slp + flp;
  }
  else
  {
    return SMALL_LG_NUM;
  }
}

double _incrHmmAligModel::forwardAlgorithm(const vector<WordIndex>& nSrcSentIndexVector,
                                           const vector<WordIndex>& trgSentIndexVector, int verbose)
{
  // Obtain slen
  PositionIndex slen = getSrcLen(nSrcSentIndexVector);

  // Make room for matrix
  vector<vector<double>> forwardMatrix;
  vector<double> dVec;
  dVec.insert(dVec.begin(), trgSentIndexVector.size() + 1, 0.0);
  forwardMatrix.insert(forwardMatrix.begin(), nSrcSentIndexVector.size() + 1, dVec);

  // Fill matrix
  for (PositionIndex j = 1; j <= trgSentIndexVector.size(); ++j)
  {
    for (PositionIndex i = 1; i <= nSrcSentIndexVector.size(); ++i)
    {
      double logPts = logpts(nSrcSentIndexVector[i - 1], trgSentIndexVector[j - 1]);
      if (j == 1)
      {
        forwardMatrix[i][j] = logaProb(0, slen, i) + logPts;
      }
      else
      {
        for (PositionIndex i_tilde = 1; i_tilde <= nSrcSentIndexVector.size(); ++i_tilde)
        {
          double lp = forwardMatrix[i_tilde][j - 1] + (double)logaProb(i_tilde, slen, i) + logPts;
          if (i_tilde == 1)
            forwardMatrix[i][j] = lp;
          else
            forwardMatrix[i][j] = MathFuncs::lns_sumlog(lp, forwardMatrix[i][j]);
        }
      }
    }
  }

  // Obtain lgProb from forward matrix
  double lp = lgProbGivenForwardMatrix(forwardMatrix);

  // Print verbose info
  if (verbose > 1)
  {
    // Clear cached alpha and beta values
    for (PositionIndex j = 1; j <= trgSentIndexVector.size(); ++j)
    {
      for (PositionIndex i = 1; i <= nSrcSentIndexVector.size(); ++i)
      {
        cerr << "i=" << i << ",j=" << j << " " << forwardMatrix[i][j];
        if (i < nSrcSentIndexVector.size())
          cerr << " ; ";
      }
      cerr << endl;
    }
  }

  // Return result
  return lp;
}

double _incrHmmAligModel::lgProbGivenForwardMatrix(const vector<vector<double>>& forwardMatrix)
{
  // Sum lgprob for each i
  double lp = SMALL_LG_NUM;
  PositionIndex last_j = forwardMatrix[1].size() - 1;
  for (unsigned int i = 1; i <= forwardMatrix.size() - 1; ++i)
  {
    if (i == 1)
    {
      lp = forwardMatrix[i][last_j];
    }
    else
    {
      lp = MathFuncs::lns_sumlog(lp, forwardMatrix[i][last_j]);
    }
  }

  // Return result
  return lp;
}

LgProb _incrHmmAligModel::calcLgProbPhr(const vector<WordIndex>& sPhr, const vector<WordIndex>& tPhr, int verbose)
{
  //  return calcVitIbm1LgProb(sPhr,tPhr);
  //  return calcSumIBM1LgProb(sPhr,tPhr,verbose);
  return noisyOrLgProb(sPhr, tPhr, verbose);
}

LgProb _incrHmmAligModel::calcVitIbm1LgProb(const vector<WordIndex>& srcSentIndexVector,
                                            const vector<WordIndex>& trgSentIndexVector)
{
  LgProb aligLgProb;
  LgProb lp;
  LgProb max_lp;
  vector<WordIndex> nSrcSentIndexVector = addNullWordToWidxVec(srcSentIndexVector);

  aligLgProb = 0;
  for (unsigned int j = 0; j < trgSentIndexVector.size(); ++j)
  {
    max_lp = -FLT_MAX;
    for (unsigned int i = 0; i < nSrcSentIndexVector.size(); ++i)
    {
      lp = log((double)pts(nSrcSentIndexVector[i], trgSentIndexVector[j]));
      if (max_lp <= lp)
      {
        max_lp = lp;
      }
    }
    aligLgProb = aligLgProb + max_lp;
  }

  return aligLgProb;
}

LgProb _incrHmmAligModel::calcSumIBM1LgProb(const vector<WordIndex>& sSent, const vector<WordIndex>& tSent, int verbose)
{
  LgProb sumlp;
  LgProb lexContrib;
  LgProb lgProb;
  unsigned int i, j;
  vector<WordIndex> nsSent = addNullWordToWidxVec(sSent);

  if (verbose)
    cerr << "Obtaining Sum IBM Model 1 logprob...\n";

  lgProb = logaProbIbm1(sSent.size(), tSent.size());

  if (verbose)
    cerr << "- aligLgProb(tlen=" << tSent.size() << " | slen=" << sSent.size()
         << ")= " << logaProbIbm1(sSent.size(), tSent.size()) << endl;

  lgProb += sentLenLgProb(sSent.size(), tSent.size());
  if (verbose)
    cerr << "- lenLgProb(tlen=" << tSent.size() << " | slen=" << sSent.size()
         << ")= " << sentLenLgProb(sSent.size(), tSent.size()) << endl;

  lexContrib = 0;
  for (j = 0; j < tSent.size(); ++j)
  {
    for (i = 0; i < nsSent.size(); ++i)
    {
      if (i == 0)
        sumlp = logpts(nsSent[i], tSent[j]);
      else
        sumlp = MathFuncs::lns_sumlog(logpts(nsSent[i], tSent[j]), sumlp);
      if (verbose == 2)
        cerr << "log(t( " << tSent[j] << " | " << nsSent[i] << " ))= " << logpts(nsSent[i], tSent[j]) << endl;
    }
    lexContrib += sumlp;
    if (verbose)
      cerr << "- log(sumt(j=" << j << "))= " << sumlp << endl;
    if (verbose == 2)
      cerr << endl;
  }
  if (verbose)
    cerr << "- Lexical model contribution= " << lexContrib << endl;
  lgProb += lexContrib;

  return lgProb;
}

LgProb _incrHmmAligModel::logaProbIbm1(PositionIndex slen, PositionIndex tlen)
{
  LgProb aligLgProb = 0;

  for (unsigned int j = 0; j < tlen; ++j)
  {
    aligLgProb = (double)aligLgProb - (double)log((double)slen + 1);
  }
  return aligLgProb;
}

LgProb _incrHmmAligModel::noisyOrLgProb(const vector<WordIndex>& sSent, const vector<WordIndex>& tSent, int verbose)
{
  LgProb lp = 0;
  for (unsigned int j = 0; j < tSent.size(); ++j)
  {
    Prob prob = 1;
    for (unsigned int i = 0; i < sSent.size(); ++i)
    {
      prob = prob * (1.0 - (double)pts(sSent[i], tSent[j]));

      if (verbose == 2)
        cerr << "t( " << tSent[j] << " | " << sSent[i] << " )= " << pts(sSent[i], tSent[j]) << endl;
    }
    Prob compProb = 1.0 - (double)prob;
    if ((double)compProb == 0.0)
      lp = lp + (double)SMALL_LG_NUM;
    else
      lp = lp + compProb.get_lp();

    if (verbose)
      cerr << "- log(1-prod(j=" << j << "))= " << lp << endl;
    if (verbose == 2)
      cerr << endl;
  }
  return lp;
}

bool _incrHmmAligModel::load(const char* prefFileName, int verbose)
{
  if (prefFileName[0] != 0)
  {
    bool retVal;

    if (verbose)
      cerr << "Loading incremental HMM Model data..." << endl;

    // Load vocabularies if they exist
    string srcVocFileName = prefFileName;
    srcVocFileName = srcVocFileName + ".svcb";
    loadGIZASrcVocab(srcVocFileName.c_str(), verbose);

    string trgVocFileName = prefFileName;
    trgVocFileName = trgVocFileName + ".tvcb";
    loadGIZATrgVocab(trgVocFileName.c_str(), verbose);

    // Load files with source and target sentences
    // Warning: this must be made before reading file with lanji
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

    // Load file with lanji values
    retVal = lanji.load(prefFileName, verbose);
    if (retVal == THOT_ERROR)
      return THOT_ERROR;

    // Load file with lanjm1ip_anji values
    retVal = lanjm1ip_anji.load(prefFileName, verbose);
    if (retVal == THOT_ERROR)
      return THOT_ERROR;

    // Load file with lexical nd values
    string lexNumDenFile = prefFileName;
    lexNumDenFile = lexNumDenFile + lexNumDenFileExtension;
    retVal = lexTable->load(lexNumDenFile.c_str(), verbose);
    if (retVal == THOT_ERROR)
      return THOT_ERROR;

    // Load file with alignment nd values
    string aligNumDenFile = prefFileName;
    aligNumDenFile = aligNumDenFile + ".hmm_alignd";
    retVal = aligTable.load(aligNumDenFile.c_str(), verbose);
    if (retVal == THOT_ERROR)
      return THOT_ERROR;

    // Load file with with lexical smoothing interpolation factor
    string lsifFile = prefFileName;
    lsifFile = lsifFile + ".lsifactor";
    retVal = loadLexSmIntFactor(lsifFile.c_str(), verbose);
    if (retVal == THOT_ERROR)
      return THOT_ERROR;

    // Load file with with alignment smoothing interpolation factor
    string asifFile = prefFileName;
    asifFile = asifFile + ".asifactor";
    retVal = loadAlSmIntFactor(asifFile.c_str(), verbose);
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

bool _incrHmmAligModel::print(const char* prefFileName, int verbose)
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

  // close source and target sentence files
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

  // Print file with lanji values
  retVal = lanji.print(prefFileName);
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  // Print file with lanjm1ip_anji values
  retVal = lanjm1ip_anji.print(prefFileName);
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  // Print file with lexical nd values
  string lexNumDenFile = prefFileName;
  lexNumDenFile = lexNumDenFile + lexNumDenFileExtension;
  retVal = lexTable->print(lexNumDenFile.c_str());
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  // Print file with alignment nd values
  string aligNumDenFile = prefFileName;
  aligNumDenFile = aligNumDenFile + ".hmm_alignd";
  retVal = aligTable.print(aligNumDenFile.c_str());
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  // Print file with with lexical smoothing interpolation factor
  string lsifFile = prefFileName;
  lsifFile = lsifFile + ".lsifactor";
  retVal = printLexSmIntFactor(lsifFile.c_str(), verbose);
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  // Print file with with alignment smoothing interpolation factor
  string asifFile = prefFileName;
  asifFile = asifFile + ".asifactor";
  retVal = printAlSmIntFactor(asifFile.c_str(), verbose);
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

void _incrHmmAligModel::clear()
{
  _swAligModel::clear();
  clearSentLengthModel();
  clearTempVars();
  lanji.clear();
  lanjm1ip_anji.clear();
  lexTable->clear();
  aligTable.clear();
}

void _incrHmmAligModel::clearInfoAboutSentRange(void)
{
  // Clear info about sentence range
  sentenceHandler.clear();
  iter = 0;
  lanji.clear();
  lanji_aux.clear();
  lanjm1ip_anji.clear();
  lanjm1ip_anji_aux.clear();
  clearSentLengthModel();
}

void _incrHmmAligModel::clearTempVars()
{
  iter = 0;
  lanji_aux.clear();
  lanjm1ip_anji_aux.clear();
  lexCounts.clear();
  incrLexCounts.clear();
  aligCounts.clear();
  incrAligCounts.clear();
  cachedAligLogProbs.clear();
}

void _incrHmmAligModel::clearSentLengthModel()
{
  sentLengthModel.clear();
}

_incrHmmAligModel::~_incrHmmAligModel()
{
  delete lexTable;
}
