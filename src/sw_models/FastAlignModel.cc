#include "FastAlignModel.h"

#include "nlp_common/ErrorDefs.h"
#include "nlp_common/MathFuncs.h"
#include "sw_models/DiagonalAlignment.h"
#include "sw_models/FastAlignModel.h"
#include "sw_models/Md.h"

#include <algorithm>
#include <iomanip>
#include <sstream>
#ifdef _WIN32
#include <Windows.h>
#endif

using namespace std;

FastAlignModel::FastAlignModel()
{
  variationalBayes = true;
}

void FastAlignModel::set_expval_maxnsize(unsigned int _anji_maxnsize)
{
  anji.set_maxnsize(_anji_maxnsize);
}

double FastAlignModel::getFastAlignP0() const
{
  return fastAlignP0;
}

void FastAlignModel::setFastAlignP0(double value)
{
  fastAlignP0 = value;
}

unsigned int FastAlignModel::startTraining(int verbosity)
{
  clearTempVars();
  vector<vector<WordIndex>> insertBuffer;
  size_t insertBufferItems = 0;
  unsigned int count = 0;
  for (unsigned int n = 0; n < numSentencePairs(); ++n)
  {
    vector<WordIndex> src = getSrcSent(n);
    vector<WordIndex> trg = getTrgSent(n);

    if (sentenceLengthIsOk(src) && sentenceLengthIsOk(trg))
    {
      unsigned int slen = (unsigned int)src.size();
      unsigned int tlen = (unsigned int)trg.size();
      totLenRatio += static_cast<double>(tlen) / static_cast<double>(slen);
      trgTokenCount += tlen;
      incrementSizeCount(tlen, slen);

      lexTable.setDenominator(NULL_WORD, 0);
      for (const WordIndex t : trg)
      {
        lexTable.setNumerator(NULL_WORD, t, 0);
        initCountSlot(NULL_WORD, t);
      }
      for (const WordIndex s : src)
      {
        lexTable.setDenominator(s, 0);
        if (s >= insertBuffer.size())
          insertBuffer.resize((size_t)s + 1);
        for (const WordIndex t : trg)
          insertBuffer[s].push_back(t);
        insertBufferItems += tlen;
      }
      if (insertBufferItems > ThreadBufferSize * 100)
      {
        insertBufferItems = 0;
        addTranslationOptions(insertBuffer);
      }
      ++count;
    }
  }
  if (!insertBuffer.empty())
    addTranslationOptions(insertBuffer);

  if (verbosity)
  {
    double meanSrclenMultiplier = totLenRatio / numSentencePairs();
    cerr << "expected target length = source length * " << meanSrclenMultiplier << endl;
  }
  return count;
}

void FastAlignModel::train(int verbosity)
{
  empFeatSum = 0;
  vector<pair<vector<WordIndex>, vector<WordIndex>>> buffer;
  for (unsigned int n = 0; n < numSentencePairs(); ++n)
  {
    vector<WordIndex> src = getSrcSent(n);
    vector<WordIndex> trg = getTrgSent(n);
    if (sentenceLengthIsOk(src) && sentenceLengthIsOk(trg))
      buffer.push_back(make_pair(src, trg));

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

  if (iter > 0)
    optimizeDiagonalTension(8, verbosity);
  batchMaximizeProbs();
  iter++;
}

void FastAlignModel::endTraining()
{
  clearTempVars();
}

void FastAlignModel::optimizeDiagonalTension(unsigned int nIters, int verbose)
{
  double empFeat = empFeatSum / trgTokenCount;
  if (verbose)
  {
    cerr << " posterior al-feat: " << empFeat << endl;
    cerr << "       size counts: " << sizeCounts.size() << endl;
  }

  for (unsigned int ii = 0; ii < nIters; ++ii)
  {
    double modFeat = 0;
#pragma omp parallel for reduction(+ : modFeat)
    for (int i = 0; i < (int)sizeCounts.size(); ++i)
    {
      const pair<short, short>& p = sizeCounts.getAt(i).first;
      for (short j = 1; j <= p.first; ++j)
      {
        double dLogZ = DiagonalAlignment::ComputeDLogZ(j, p.first, p.second, diagonalTension);
        modFeat += sizeCounts.getAt(i).second * dLogZ;
      }
    }
    modFeat /= trgTokenCount;
    if (verbose)
      cerr << "  " << ii + 1 << "  model al-feat: " << modFeat << " (tension=" << diagonalTension << ")\n";
    diagonalTension += (empFeat - modFeat) * 20.0;
    if (diagonalTension <= 0.1)
      diagonalTension = 0.1;
    if (diagonalTension > 14)
      diagonalTension = 14;
  }
  if (verbose)
    cerr << "     final tension: " << diagonalTension << endl;
}

void FastAlignModel::addTranslationOptions(vector<vector<WordIndex>>& insertBuffer)
{
  WordIndex maxSrcWordIndex = (WordIndex)insertBuffer.size() - 1;
  if (maxSrcWordIndex >= lexCounts.size())
    lexCounts.resize((size_t)maxSrcWordIndex + 1);
  lexTable.reserveSpace(maxSrcWordIndex);

#pragma omp parallel for schedule(dynamic)
  for (int s = 0; s < (int)insertBuffer.size(); ++s)
  {
    for (WordIndex t : insertBuffer[s])
    {
      initCountSlot(s, t);
      lexTable.setNumerator(s, t, 0);
    }
    insertBuffer[s].clear();
  }
}

void FastAlignModel::incrementSizeCount(unsigned int tlen, unsigned int slen)
{
  pair<short, short> key((short)tlen, (short)slen);
  unsigned int* countPtr = sizeCounts.findPtr(key);
  if (countPtr == NULL)
    sizeCounts.insert(key, 1);
  else
    (*countPtr)++;
}

void FastAlignModel::batchUpdateCounts(const vector<pair<vector<WordIndex>, vector<WordIndex>>>& pairs)
{
  double curEmpFeatSum = 0.0;
#pragma omp parallel for schedule(dynamic) reduction(+ : curEmpFeatSum)
  for (int line_idx = 0; line_idx < (int)pairs.size(); ++line_idx)
  {
    vector<WordIndex> src = pairs[line_idx].first;
    vector<WordIndex> trg = pairs[line_idx].second;
    unsigned int slen = (unsigned int)src.size();
    unsigned int tlen = (unsigned int)trg.size();
    vector<double> probs(src.size() + 1);
    for (PositionIndex j = 1; j <= trg.size(); ++j)
    {
      const WordIndex& fj = trg[j - 1];
      double sum = 0;
      probs[0] = translationProb(NULL_WORD, fj) * (double)alignmentProb(j, slen, tlen, 0);
      sum += probs[0];
      double az = computeAZ(j, slen, tlen);
      for (PositionIndex i = 1; i <= src.size(); ++i)
      {
        probs[i] = translationProb(src[i - 1], fj) * (double)alignmentProb(az, j, slen, tlen, i);
        sum += probs[i];
      }
      double count = probs[0] / sum;
      incrementCount(NULL_WORD, fj, count);
      for (PositionIndex i = 1; i <= src.size(); ++i)
      {
        double p = probs[i] / sum;
        incrementCount(src[i - 1], fj, p);
        curEmpFeatSum += DiagonalAlignment::Feature(j - 1, i, tlen, slen) * p;
      }
    }
  }
  empFeatSum += curEmpFeatSum;
}

void FastAlignModel::batchMaximizeProbs(void)
{
#pragma omp parallel for schedule(dynamic)
  for (int s = 0; s < (int)lexCounts.size(); ++s)
  {
    double denom = 0;
    LexCountsElem& elem = lexCounts[s];
    for (LexCountsElem::iterator it = elem.begin(); it != elem.end(); ++it)
    {
      double numer = it->second;
      if (variationalBayes)
        numer += alpha;
      denom += numer;
      lexTable.setNumerator(s, it->first, (float)log(numer));
      it->second = 0.0;
    }
    if (denom == 0)
      denom = 1;
    lexTable.setDenominator(s, (float)log(denom));
  }
}

void FastAlignModel::startIncrTraining(pair<unsigned int, unsigned int> sentPairRange, int verbosity)
{
  clearTempVars();
  for (unsigned int n = sentPairRange.first; n <= sentPairRange.second; ++n)
  {
    vector<WordIndex> srcSent = getSrcSent(n);
    vector<WordIndex> trgSent = getTrgSent(n);

    unsigned int slen = (unsigned int)srcSent.size();
    unsigned int tlen = (unsigned int)trgSent.size();

    totLenRatio += static_cast<double>(tlen) / static_cast<double>(slen);
    trgTokenCount += tlen;
    incrementSizeCount(tlen, slen);
  }

  if (verbosity)
  {
    double meanSrclenMultiplier = totLenRatio / numSentencePairs();
    cerr << "expected target length = source length * " << meanSrclenMultiplier << endl;
  }
}

void FastAlignModel::incrTrain(pair<unsigned int, unsigned int> sentPairRange, int verbosity)
{
  calcNewLocalSuffStats(sentPairRange, verbosity);

  optimizeDiagonalTension(2, verbosity);
  incrMaximizeProbs();
  iter++;
}

void FastAlignModel::endIncrTraining()
{
  clearTempVars();
}

void FastAlignModel::calcNewLocalSuffStats(pair<unsigned int, unsigned int> sentPairRange, int verbosity)
{
  // Iterate over the training samples
  for (unsigned int n = sentPairRange.first; n <= sentPairRange.second; ++n)
  {
    // Calculate sufficient statistics

    // Init vars for n'th sample
    vector<WordIndex> srcSent = getSrcSent(n);
    vector<WordIndex> nsrcSent = addNullWordToWidxVec(srcSent);
    vector<WordIndex> trgSent = getTrgSent(n);

    Count weight;
    sentenceHandler->getCount(n, weight);

    // Calculate sufficient statistics for anji values
    calc_anji(n, nsrcSent, trgSent, weight);
  }
}

void FastAlignModel::calc_anji(unsigned int n, const vector<WordIndex>& nsrcSent, const vector<WordIndex>& trgSent,
                               const Count& weight)
{
  PositionIndex slen = (PositionIndex)nsrcSent.size() - 1;
  PositionIndex tlen = (PositionIndex)trgSent.size();

  // Initialize anji and anji_aux
  unsigned int mapped_n;
  anji.init_nth_entry(n, (PositionIndex)nsrcSent.size(), (PositionIndex)trgSent.size(), mapped_n);

  unsigned int n_aux = 1;
  unsigned int mapped_n_aux;
  anji_aux.init_nth_entry(n_aux, (PositionIndex)nsrcSent.size(), (PositionIndex)trgSent.size(), mapped_n_aux);

  // Calculate new estimation of anji
  for (unsigned int j = 1; j <= trgSent.size(); ++j)
  {
    double az = computeAZ(j, slen, tlen);
    // Obtain sum_anji_num_forall_s
    double sum_anji_num_forall_s = 0;
    vector<double> numVec;
    for (unsigned int i = 0; i < nsrcSent.size(); ++i)
    {
      // Smooth numerator
      double d = calc_anji_num(az, nsrcSent, trgSent, i, j);
      if (d < SmoothingAnjiNum)
        d = SmoothingAnjiNum;
      // Add contribution to sum
      sum_anji_num_forall_s += d;
      // Store num in numVec
      numVec.push_back(d);
    }
    // Set value of anji_aux
    for (unsigned int i = 0; i < nsrcSent.size(); ++i)
    {
      double p = numVec[i] / sum_anji_num_forall_s;
      anji_aux.set_fast(mapped_n_aux, j, i, (float)p);
      if (i > 0)
        empFeatSum += DiagonalAlignment::Feature(j - 1, i, tlen, slen) * p;
    }
  }

  // Gather sufficient statistics
  if (anji_aux.n_size() != 0)
  {
    for (unsigned int j = 1; j <= trgSent.size(); ++j)
    {
      for (unsigned int i = 0; i < nsrcSent.size(); ++i)
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

double FastAlignModel::calc_anji_num(double az, const vector<WordIndex>& nsrcSent, const vector<WordIndex>& trgSent,
                                     unsigned int i, unsigned int j)
{
  bool found;
  WordIndex s = nsrcSent[i];
  WordIndex t = trgSent[j - 1];

  double prob;
  lexTable.getNumerator(s, t, found);
  if (found)
  {
    // s,t has previously been seen
    prob = translationProb(s, t);
  }
  else
  {
    // s,t has never been seen
    prob = ArbitraryPts;
  }

  return prob * (double)alignmentProb(az, j, (PositionIndex)nsrcSent.size() - 1, (PositionIndex)trgSent.size(), i);
}

void FastAlignModel::incrUpdateCounts(unsigned int mapped_n, unsigned int mapped_n_aux, PositionIndex i,
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
  while (incrLexCounts.size() <= s)
  {
    IncrLexCountsElem lexAuxVarElem;
    incrLexCounts.push_back(lexAuxVarElem);
  }

  IncrLexCountsElem::iterator lexAuxVarElemIter = incrLexCounts[s].find(t);
  if (lexAuxVarElemIter != incrLexCounts[s].end())
  {
    if (weighted_curr_lanji != SMALL_LG_NUM)
    {
      lexAuxVarElemIter->second.first =
          MathFuncs::lns_sumlog_float(lexAuxVarElemIter->second.first, weighted_curr_lanji);
    }
    lexAuxVarElemIter->second.second =
        MathFuncs::lns_sumlog_float(lexAuxVarElemIter->second.second, weighted_new_lanji);
  }
  else
  {
    incrLexCounts[s][t] = make_pair(weighted_curr_lanji, weighted_new_lanji);
  }
}

void FastAlignModel::incrMaximizeProbs(void)
{
  float initialNumer = variationalBayes ? (float)log(alpha) : SMALL_LG_NUM;
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
        float numer = lexTable.getNumerator(s, t, numerFound);
        if (!numerFound)
          numer = initialNumer;

        // Obtain lexDenom for s,t
        bool denomFound;
        float denom = lexTable.getDenominator(s, denomFound);
        if (!denomFound)
          denom = SMALL_LG_NUM;

        // Obtain new sufficient statistics
        float new_numer = obtainLogNewSuffStat(numer, log_suff_stat_curr, log_suff_stat_new);
        float new_denom = denom;
        if (numerFound)
          new_denom = MathFuncs::lns_sublog_float(denom, numer);
        new_denom = MathFuncs::lns_sumlog_float(new_denom, new_numer);

        // Set lexical numerator and denominator
        lexTable.set(s, t, new_numer, new_denom);
      }
    }
  }
  // Clear auxiliary variables
  incrLexCounts.clear();
}

float FastAlignModel::obtainLogNewSuffStat(float lcurrSuffStat, float lLocalSuffStatCurr, float lLocalSuffStatNew)
{
  float lresult = MathFuncs::lns_sublog_float(lcurrSuffStat, lLocalSuffStatCurr);
  lresult = MathFuncs::lns_sumlog_float(lresult, lLocalSuffStatNew);
  return lresult;
}

void FastAlignModel::loadConfig(const YAML::Node& config)
{
  AlignmentModelBase::loadConfig(config);

  fastAlignP0 = config["fastAlignP0"].as<double>();
}

void FastAlignModel::createConfig(YAML::Emitter& out)
{
  AlignmentModelBase::createConfig(out);

  out << YAML::Key << "fastAlignP0" << YAML::Value << fastAlignP0;
}

void FastAlignModel::initCountSlot(WordIndex s, WordIndex t)
{
  // NOT thread safe
  if (s >= lexCounts.size())
    lexCounts.resize((size_t)s + 1);
  lexCounts[s][t] = 0;
}

void FastAlignModel::incrementCount(WordIndex s, WordIndex t, double x)
{
#pragma omp atomic
  lexCounts[s].find(t)->second += x;
}

LgProb FastAlignModel::getBestAlignment(const vector<WordIndex>& srcSentence, const vector<WordIndex>& trgSentence,
                                        vector<PositionIndex>& bestAlignment)
{
  if (sentenceLengthIsOk(srcSentence) && sentenceLengthIsOk(trgSentence))
  {
    bestAlignment.clear();

    unsigned int slen = (unsigned int)srcSentence.size();
    unsigned int tlen = (unsigned int)trgSentence.size();

    double logProb = sentenceLengthLogProb(slen, tlen);

    // compute likelihood
    for (PositionIndex j = 0; j < trgSentence.size(); ++j)
    {
      WordIndex t = trgSentence[j];
      int best_i = 0;
      double bestProb = translationProb(NULL_WORD, t) * alignmentProb(j + 1, slen, tlen, 0);
      double az = computeAZ(j + 1, slen, tlen);
      for (PositionIndex i = 1; i <= srcSentence.size(); ++i)
      {
        double prob = translationProb(srcSentence[i - 1], t) * alignmentProb(az, j + 1, slen, tlen, i);
        if (prob > bestProb)
        {
          bestProb = prob;
          best_i = i;
        }
      }
      logProb += log(bestProb);
      bestAlignment.push_back(best_i);
    }
    return logProb;
  }
  else
  {
    bestAlignment.resize(trgSentence.size(), 0);
    return SMALL_LG_NUM;
  }
}

Prob FastAlignModel::translationProb(WordIndex s, WordIndex t)
{
  return translationLogProb(s, t).get_p();
}

LgProb FastAlignModel::translationLogProb(WordIndex s, WordIndex t)
{
  bool found;
  double numer;

  numer = lexTable.getNumerator(s, t, found);
  if (found)
  {
    // lexNumer for pair s,t exists
    double denom;

    denom = lexTable.getDenominator(s, found);
    if (!found)
      return SmoothingLogProb;
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
    return SmoothingLogProb;
  }
}

double FastAlignModel::computeAZ(PositionIndex j, PositionIndex slen, PositionIndex tlen)
{
  double z = DiagonalAlignment::ComputeZ(j, tlen, slen, diagonalTension);
  return z / (1.0 - fastAlignP0);
}

Prob FastAlignModel::alignmentProb(double az, PositionIndex j, PositionIndex slen, PositionIndex tlen, PositionIndex i)
{
  double unnormalizedProb = DiagonalAlignment::UnnormalizedProb(j, i, tlen, slen, diagonalTension);
  return unnormalizedProb / az;
}

Prob FastAlignModel::alignmentProb(PositionIndex j, PositionIndex slen, PositionIndex tlen, PositionIndex i)
{
  if (i == 0)
    return fastAlignP0;

  double az = computeAZ(j, slen, tlen);
  return alignmentProb(az, j, slen, tlen, i);
}

LgProb FastAlignModel::alignmentLogProb(PositionIndex j, PositionIndex slen, PositionIndex tlen, PositionIndex i)
{
  return alignmentProb(j, slen, tlen, i).get_lp();
}

Prob FastAlignModel::sentenceLengthProb(unsigned int slen, unsigned int tlen)
{
  return sentenceLengthLogProb(slen, tlen).get_p();
}

LgProb FastAlignModel::sentenceLengthLogProb(unsigned int slen, unsigned int tlen)
{
  unsigned int sentenceCount = numSentencePairs();
  double meanSrcLenMultipler = totLenRatio == 0 || sentenceCount == 0 ? 1.0 : totLenRatio / sentenceCount;
  return Md::log_poisson(tlen, 0.05 + slen * meanSrcLenMultipler);
}

bool FastAlignModel::getEntriesForSource(WordIndex s, NbestTableNode<WordIndex>& trgtn)
{
  set<WordIndex> transSet;
  bool ret = lexTable.getTransForSource(s, transSet);
  if (ret == false)
    return false;

  trgtn.clear();
  set<WordIndex>::const_iterator setIter;
  for (setIter = transSet.begin(); setIter != transSet.end(); ++setIter)
  {
    WordIndex t = *setIter;
    trgtn.insert(translationProb(s, t), t);
  }
  return true;
}

pair<double, double> FastAlignModel::loglikelihoodForPairRange(pair<unsigned int, unsigned int> sentPairRange,
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
    loglikelihood += (double)computeSumLogProb(nthSrcSent, nthTrgSent, verbosity);
    ++numSents;
  }
  return make_pair(loglikelihood, loglikelihood / (double)numSents);
}

LgProb FastAlignModel::computeSumLogProb(const vector<WordIndex>& srcSentence, const vector<WordIndex>& trgSentence,
                                         int verbose)
{
  unsigned int slen = (unsigned int)srcSentence.size();
  unsigned int tlen = (unsigned int)trgSentence.size();

  double logProb = sentenceLengthLogProb(slen, tlen);

  // compute likelihood
  for (PositionIndex j = 0; j < trgSentence.size(); ++j)
  {
    WordIndex t = trgSentence[j];
    double sum = translationProb(NULL_WORD, t) * alignmentProb(j + 1, slen, tlen, 0);
    double az = computeAZ(j + 1, slen, tlen);
    for (PositionIndex i = 1; i <= srcSentence.size(); ++i)
    {
      double prob = translationProb(srcSentence[i - 1], t) * alignmentProb(az, j + 1, slen, tlen, i);
      sum += prob;
    }
    logProb += log(sum);
  }
  return logProb;
}

LgProb FastAlignModel::computeLogProb(const vector<WordIndex>& srcSentence, const vector<WordIndex>& trgSentence,
                                      const WordAlignmentMatrix& aligMatrix, int verbose)
{
  vector<WordIndex> nsSent = addNullWordToWidxVec(srcSentence);
  vector<PositionIndex> alig;
  aligMatrix.getAligVec(alig);

  unsigned int slen = (unsigned int)srcSentence.size();
  unsigned int tlen = (unsigned int)trgSentence.size();

  double logProb = sentenceLengthLogProb(slen, tlen);

  // compute likelihood
  for (PositionIndex j = 0; j < alig.size(); ++j)
  {
    PositionIndex i = alig[j];
    double pat = translationProb(nsSent[i], trgSentence[j]) * alignmentProb(j + 1, slen, tlen, i);
    logProb += log(pat);
  }
  return logProb;
}

bool FastAlignModel::load(const char* prefFileName, int verbose)
{
  if (prefFileName[0] != 0)
  {
    bool retVal;

    retVal = AlignmentModelBase::load(prefFileName, verbose);
    if (retVal == THOT_ERROR)
      return THOT_ERROR;

    if (verbose)
      cerr << "Loading FastAlign Model data..." << endl;

    // Load file with anji values
    anji.load(prefFileName, verbose);

    string lexNumDenFile = prefFileName;
    lexNumDenFile = lexNumDenFile + ".fa_lexnd";
    retVal = lexTable.load(lexNumDenFile.c_str(), verbose);
    if (retVal == THOT_ERROR)
      return THOT_ERROR;

    string sizeCountsFile = prefFileName;
    sizeCountsFile = sizeCountsFile + ".size_counts";
    retVal = loadSizeCounts(sizeCountsFile);
    if (retVal == THOT_ERROR)
      return THOT_ERROR;

    string paramsFile = prefFileName;
    paramsFile = paramsFile + ".params";
    retVal = loadParams(paramsFile);
    if (retVal == THOT_ERROR)
      return THOT_ERROR;

    return THOT_OK;
  }
  else
    return THOT_ERROR;
}

bool FastAlignModel::loadParams(const string& filename)
{
  ifstream in(filename);
  if (!in)
    return THOT_ERROR;
  in >> empFeatSum >> diagonalTension;

  return THOT_OK;
}

bool FastAlignModel::loadSizeCounts(const string& filename)
{
  ifstream in(filename);
  if (!in)
    return THOT_ERROR;

  trgTokenCount = 0;
  totLenRatio = 0;
  unsigned int tlen, slen, count;
  while (in >> tlen >> slen >> count)
  {
    sizeCounts[make_pair<short, short>((short)tlen, (short)slen)] = count;
    trgTokenCount += static_cast<double>(tlen) * static_cast<double>(count);
    totLenRatio += (static_cast<double>(tlen) / static_cast<double>(slen)) * static_cast<double>(count);
  }

  return THOT_OK;
}

bool FastAlignModel::print(const char* prefFileName, int verbose)
{
  bool retVal;

  retVal = AlignmentModelBase::print(prefFileName);
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  // Print file anji values
  retVal = anji.print(prefFileName);
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  string lexNumDenFile = prefFileName;
  lexNumDenFile = lexNumDenFile + ".fa_lexnd";
  retVal = lexTable.print(lexNumDenFile.c_str());
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  string sizeCountsFile = prefFileName;
  sizeCountsFile = sizeCountsFile + ".size_counts";
  retVal = printSizeCounts(sizeCountsFile);
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  string paramsFile = prefFileName;
  paramsFile = paramsFile + ".params";
  retVal = printParams(paramsFile);
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  return THOT_OK;
}

bool FastAlignModel::printParams(const string& filename)
{
  ofstream out(filename);
  if (!out)
    return THOT_ERROR;
  out << setprecision(numeric_limits<double>::max_digits10) << empFeatSum << " " << diagonalTension;
  return THOT_OK;
}

bool FastAlignModel::printSizeCounts(const string& filename)
{
  ofstream out(filename, ios::binary);
  if (!out)
    return THOT_ERROR;

  for (SizeCounts::iterator iter = sizeCounts.begin(); iter != sizeCounts.end(); ++iter)
    out << iter->first.first << " " << iter->first.second << " " << iter->second << endl;

  return THOT_OK;
}

vector<WordIndex> FastAlignModel::getSrcSent(unsigned int n)
{
  vector<string> srcsStr;
  vector<WordIndex> result;

  sentenceHandler->getSrcSentence(n, srcsStr);
  for (unsigned int i = 0; i < srcsStr.size(); ++i)
  {
    WordIndex widx = stringToSrcWordIndex(srcsStr[i]);
    if (widx == UNK_WORD)
      widx = addSrcSymbol(srcsStr[i]);
    result.push_back(widx);
  }
  return result;
}

vector<WordIndex> FastAlignModel::getTrgSent(unsigned int n)
{
  vector<string> trgsStr;
  vector<WordIndex> trgs;

  sentenceHandler->getTrgSentence(n, trgsStr);
  for (unsigned int i = 0; i < trgsStr.size(); ++i)
  {
    WordIndex widx = stringToTrgWordIndex(trgsStr[i]);
    if (widx == UNK_WORD)
      widx = addTrgSymbol(trgsStr[i]);
    trgs.push_back(widx);
  }
  return trgs;
}

void FastAlignModel::clearSentenceLengthModel()
{
  totLenRatio = 0;
}

void FastAlignModel::clearTempVars()
{
  iter = 0;
  lexCounts.clear();
  incrLexCounts.clear();
  anji_aux.clear();
}

void FastAlignModel::clear()
{
  AlignmentModelBase::clear();
  diagonalTension = 4.0;
  lexTable.clear();
  anji.clear();
  sizeCounts.clear();
  empFeatSum = 0;
  trgTokenCount = 0;
}
