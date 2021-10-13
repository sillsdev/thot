#include "sw_models/IncrHmmAlignmentTrainer.h"

using namespace std;

IncrHmmAlignmentTrainer::IncrHmmAlignmentTrainer(HmmAlignmentModel& model, anjiMatrix& lanji,
                                                 anjm1ip_anjiMatrix& lanjm1ip_anji)
    : lanji{lanji}, lanjm1ip_anji{lanjm1ip_anji}, model{model}
{
}

void IncrHmmAlignmentTrainer::incrTrain(pair<unsigned int, unsigned int> sentPairRange, int verbosity)
{
  // EM algorithm
#ifdef THOT_ENABLE_VITERBI_TRAINING
  calcNewLocalSuffStatsVit(sentPairRange, verbosity);
#else
  calcNewLocalSuffStats(sentPairRange, verbosity);
#endif
  incrMaximizeProbs();
}

void IncrHmmAlignmentTrainer::clear()
{
  lanji_aux.clear();
  lanjm1ip_anji_aux.clear();
  incrLexCounts.clear();
  incrHmmAlignmentCounts.clear();
}

void IncrHmmAlignmentTrainer::calcNewLocalSuffStats(pair<unsigned int, unsigned int> sentPairRange, int verbosity)
{
  // Iterate over the training samples
  for (unsigned int n = sentPairRange.first; n <= sentPairRange.second; ++n)
  {
    // Init vars for n'th sample
    vector<WordIndex> srcSent = model.getSrcSent(n);
    vector<WordIndex> nsrcSent = model.extendWithNullWord(srcSent);
    vector<WordIndex> trgSent = model.getTrgSent(n);

    // Do not process sentence pair if sentences are empty or exceed the maximum length
    if (model.sentenceLengthIsOk(srcSent) && model.sentenceLengthIsOk(trgSent))
    {
      Count weight;
      model.sentenceHandler->getCount(n, weight);

      PositionIndex slen = (PositionIndex)srcSent.size();

      // Calculate alpha and beta matrices
      vector<vector<double>> lexProbs;
      vector<vector<double>> alignProbs;
      vector<vector<double>> alphaMatrix;
      vector<vector<double>> betaMatrix;
      model.calcAlphaBetaMatrices(nsrcSent, trgSent, slen, lexProbs, alignProbs, alphaMatrix, betaMatrix);

      // Calculate sufficient statistics for anji values
      calc_lanji(n, nsrcSent, trgSent, slen, weight, alphaMatrix, betaMatrix);

      // Calculate sufficient statistics for anjm1ip_anji values
      calc_lanjm1ip_anji(n, srcSent, trgSent, slen, weight, lexProbs, alignProbs, alphaMatrix, betaMatrix);
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

void IncrHmmAlignmentTrainer::calcNewLocalSuffStatsVit(pair<unsigned int, unsigned int> sentPairRange, int verbosity)
{
  // Define variable to cache alignment log probs
  CachedHmmAligLgProb cached_logap;

  // Iterate over the training samples
  for (unsigned int n = sentPairRange.first; n <= sentPairRange.second; ++n)
  {
    // Init vars for n'th sample
    vector<WordIndex> srcSent = model.getSrcSent(n);
    vector<WordIndex> nsrcSent = model.extendWithNullWord(srcSent);
    vector<WordIndex> trgSent = model.getTrgSent(n);

    // Do not process sentence pair if sentences are empty or exceed the maximum length
    if (model.sentenceLengthIsOk(srcSent) && model.sentenceLengthIsOk(trgSent))
    {
      Count weight;
      model.sentenceHandler->getCount(n, weight);

      PositionIndex slen = (PositionIndex)srcSent.size();

      // Execute Viterbi algorithm
      vector<vector<double>> vitMatrix;
      vector<vector<PositionIndex>> predMatrix;
      model.viterbiAlgorithmCached(nsrcSent, trgSent, cached_logap, vitMatrix, predMatrix);

      // Obtain Viterbi alignment
      vector<PositionIndex> bestAlig;
      model.bestAligGivenVitMatricesRaw(vitMatrix, predMatrix, bestAlig);

      // Calculate sufficient statistics for anji values
      calc_lanji_vit(n, nsrcSent, trgSent, bestAlig, weight);

      // Calculate sufficient statistics for anjm1ip_anji values
      calc_lanjm1ip_anji_vit(n, srcSent, trgSent, slen, bestAlig, weight);
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

void IncrHmmAlignmentTrainer::calc_lanji(unsigned int n, const vector<WordIndex>& nsrcSent,
                                         const vector<WordIndex>& trgSent, PositionIndex slen, const Count& weight,
                                         const vector<vector<double>>& alphaMatrix,
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
    double sum = 0;
    int count = 0;
    for (unsigned int i = 1; i <= nsrcSent.size(); ++i)
    {
      // Obtain numerator
      double num = alphaMatrix[i][j] * betaMatrix[i][j];
      // Add contribution to sum
      sum += num;
      // Store num in numVec
      numVec[i] = num;
      ++count;
    }
    // Set value of lanji_aux
    for (unsigned int i = 1; i <= nsrcSent.size(); ++i)
    {
      // Obtain expected value
      double lanji_val = sum == 0 ? 1.0 / count : numVec[i] / sum;
      // Smooth expected value
      if (lanji_val > model.ExpValMax)
        lanji_val = model.ExpValMax;
      if (lanji_val < model.ExpValMin)
        lanji_val = model.ExpValMin;
      // Store expected value
      lanji_aux.set_fast(mapped_n_aux, j, i, (float)log(lanji_val));
    }
  }
  // Gather lexical sufficient statistics
  gatherLexSuffStats(mapped_n, mapped_n_aux, nsrcSent, trgSent, weight);

  // clear lanji_aux data structure
  lanji_aux.clear();
}

void IncrHmmAlignmentTrainer::calc_lanji_vit(unsigned int n, const vector<WordIndex>& nsrcSent,
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
        lanji_aux.set_fast(mapped_n_aux, j, i, (float)lanji_val);
      }
    }
  }

  // Gather lexical sufficient statistics
  gatherLexSuffStats(mapped_n, mapped_n_aux, nsrcSent, trgSent, weight);

  // clear lanji_aux data structure
  lanji_aux.clear();
}

void IncrHmmAlignmentTrainer::calc_lanjm1ip_anji(unsigned int n, const vector<WordIndex>& srcSent,
                                                 const vector<WordIndex>& trgSent, PositionIndex slen,
                                                 const Count& weight, const vector<vector<double>>& lexLogProbs,
                                                 const vector<vector<double>>& alignProbs,
                                                 const vector<vector<double>>& alphaMatrix,
                                                 const vector<vector<double>>& betaMatrix)
{
  // Initialize data structures
  unsigned int mapped_n;
  lanjm1ip_anji.init_nth_entry(n, srcSent.size(), trgSent.size(), mapped_n);

  unsigned int n_aux = 1;
  unsigned int mapped_n_aux;
  lanjm1ip_anji_aux.init_nth_entry(n_aux, srcSent.size(), trgSent.size(), mapped_n_aux);

  vector<double> numVec(srcSent.size() + 1, 0);
  vector<vector<double>> numVecVec(srcSent.size() + 1, numVec);

  // Calculate new estimation of lanjm1ip_anji
  for (unsigned int j = 1; j <= trgSent.size(); ++j)
  {
    double sum = 0;
    for (unsigned int i = 1; i <= srcSent.size(); ++i)
    {
      numVecVec[i][0] = 1.0;
      if (j == 1)
      {
        // Obtain numerator

        // Obtain information about alignment
        bool nullAlig = model.isNullAlignment(0, slen, i);
        double num;
        if (nullAlig)
        {
          if (model.isFirstNullAlignmentPar(0, slen, i))
            num = alignProbs[i][0] * lexLogProbs[i][1] * betaMatrix[i][1];
          else
            num = numVecVec[size_t{slen} + 1][0];
        }
        else
          num = alignProbs[i][0] * lexLogProbs[i][1] * betaMatrix[i][1];

        // Add contribution to sum
        sum += num;
        // Store num in numVec
        numVecVec[i][0] = num;
      }
      else
      {
        for (unsigned int ip = 1; ip <= srcSent.size(); ++ip)
        {
          // Obtain numerator

          // Obtain information about alignment
          double num;
          bool validAlig = model.isValidAlignment(ip, slen, i);
          if (!validAlig)
          {
            num = 0;
          }
          else
          {
            num = alphaMatrix[ip][j - 1] * alignProbs[i][ip] * lexLogProbs[i][j] * betaMatrix[i][j];
          }
          // Add contribution to sum
          sum += num;
          // Store num in numVec
          numVecVec[i][ip] = num;
        }
      }
    }
    // Set value of lanjm1ip_anji_aux
    for (unsigned int i = 1; i <= srcSent.size(); ++i)
    {
      if (j == 1)
      {
        // Obtain expected value
        double lanjm1ip_anji_val = sum == 0 ? 0 : numVecVec[i][0] / sum;
        // Smooth expected value
        if (lanjm1ip_anji_val > model.ExpValMax)
          lanjm1ip_anji_val = model.ExpValMax;
        if (lanjm1ip_anji_val < model.ExpValMin)
          lanjm1ip_anji_val = model.ExpValMin;
        // Store expected value
        lanjm1ip_anji_aux.set_fast(mapped_n_aux, j, i, 0, (float)log(lanjm1ip_anji_val * slen));
      }
      else
      {
        for (unsigned int ip = 1; ip <= srcSent.size(); ++ip)
        {
          // Obtain information about alignment
          bool validAlig = model.isValidAlignment(ip, slen, i);
          if (validAlig)
          {
            // Obtain expected value
            double lanjm1ip_anji_val = sum == 0 ? 0 : numVecVec[i][ip] / sum;
            // Smooth expected value
            if (lanjm1ip_anji_val > model.ExpValMax)
              lanjm1ip_anji_val = model.ExpValMax;
            if (lanjm1ip_anji_val < model.ExpValMin)
              lanjm1ip_anji_val = model.ExpValMin;
            // Store expected value
            lanjm1ip_anji_aux.set_fast(mapped_n_aux, j, i, ip, (float)log(lanjm1ip_anji_val * slen));
          }
        }
      }
    }
  }
  // Gather alignment sufficient statistics
  gatherAligSuffStats(mapped_n, mapped_n_aux, srcSent, trgSent, slen, weight);

  // clear lanjm1ip_anji_aux data structure
  lanjm1ip_anji_aux.clear();
}

void IncrHmmAlignmentTrainer::calc_lanjm1ip_anji_vit(unsigned int n, const vector<WordIndex>& srcSent,
                                                     const vector<WordIndex>& trgSent, PositionIndex slen,
                                                     const vector<PositionIndex>& bestAlig, const Count& weight)
{
  // Initialize data structures
  unsigned int mapped_n;
  lanjm1ip_anji.init_nth_entry(n, srcSent.size(), trgSent.size(), mapped_n);

  unsigned int n_aux = 1;
  unsigned int mapped_n_aux;
  lanjm1ip_anji_aux.init_nth_entry(n_aux, srcSent.size(), trgSent.size(), mapped_n_aux);

  // Calculate new estimation of lanjm1ip_anji
  for (unsigned int j = 1; j <= trgSent.size(); ++j)
  {
    for (unsigned int i = 1; i <= srcSent.size(); ++i)
    {
      if (j == 1)
      {
        if (bestAlig[0] == i)
        {
          double lanjm1ip_anji_val = 0;
          // Store expected value
          lanjm1ip_anji_aux.set_fast(mapped_n_aux, j, i, 0, (float)lanjm1ip_anji_val);
        }
      }
      else
      {
        for (unsigned int ip = 1; ip <= srcSent.size(); ++ip)
        {
          PositionIndex aligModifiedIp = model.getModifiedIp(bestAlig[j - 2], slen, i);

          if (bestAlig[j - 1] == i && aligModifiedIp == ip)
          {
            double lanjm1ip_anji_val = 0;
            // Store expected value
            lanjm1ip_anji_aux.set_fast(mapped_n_aux, j, i, ip, (float)lanjm1ip_anji_val);
          }
        }
      }
    }
  }

  // Gather alignment sufficient statistics
  gatherAligSuffStats(mapped_n, mapped_n_aux, srcSent, trgSent, slen, weight);

  // clear lanjm1ip_anji_aux data structure
  lanjm1ip_anji_aux.clear();
}

void IncrHmmAlignmentTrainer::gatherLexSuffStats(unsigned int mapped_n, unsigned int mapped_n_aux,
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

void IncrHmmAlignmentTrainer::incrUpdateCountsLex(unsigned int mapped_n, unsigned int mapped_n_aux, PositionIndex i,
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
    incrLexCounts[s][t] = std::make_pair(weighted_curr_lanji, weighted_new_lanji);
  }
}

void IncrHmmAlignmentTrainer::gatherAligSuffStats(unsigned int mapped_n, unsigned int mapped_n_aux,
                                                  const vector<WordIndex>& srcSent, const vector<WordIndex>& trgSent,
                                                  PositionIndex slen, const Count& weight)
{
  // Maximize alignment parameters
  for (unsigned int j = 1; j <= trgSent.size(); ++j)
  {
    for (unsigned int i = 1; i <= srcSent.size(); ++i)
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
        for (unsigned int ip = 1; ip <= srcSent.size(); ++ip)
        {
          // Obtain information about alignment
          bool validAlig = model.isValidAlignment(ip, slen, i);
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

void IncrHmmAlignmentTrainer::incrUpdateCountsAlig(unsigned int mapped_n, unsigned int mapped_n_aux, PositionIndex slen,
                                                   PositionIndex ip, PositionIndex i, PositionIndex j,
                                                   const Count& weight)
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
  HmmAlignmentKey asHmm;
  asHmm.prev_i = ip;
  asHmm.slen = model.getCompactedSentenceLength(slen);

  // Gather local suff. statistics
  IncrHmmAlignmentCounts::iterator aligAuxVarIter = incrHmmAlignmentCounts.find(std::make_pair(asHmm, i));
  if (aligAuxVarIter != incrHmmAlignmentCounts.end())
  {
    if (weighted_curr_lanjm1ip_anji != SMALL_LG_NUM)
      aligAuxVarIter->second.first =
          MathFuncs::lns_sumlog_float(aligAuxVarIter->second.first, weighted_curr_lanjm1ip_anji);
    aligAuxVarIter->second.second =
        MathFuncs::lns_sumlog_float(aligAuxVarIter->second.second, weighted_new_lanjm1ip_anji);
  }
  else
  {
    incrHmmAlignmentCounts[std::make_pair(asHmm, i)] =
        std::make_pair(weighted_curr_lanjm1ip_anji, weighted_new_lanjm1ip_anji);
  }
}

void IncrHmmAlignmentTrainer::incrMaximizeProbs()
{
  float initialNumer = model.variationalBayes ? (float)log(model.alpha) : SMALL_LG_NUM;
  // Update parameters
  for (unsigned int i = 0; i < incrLexCounts.size(); ++i)
  {
    for (IncrLexCountsElem::iterator lexAuxVarElemIter = incrLexCounts[i].begin();
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

  // Update parameters
  for (IncrHmmAlignmentCounts::iterator aligAuxVarIter = incrHmmAlignmentCounts.begin();
       aligAuxVarIter != incrHmmAlignmentCounts.end(); ++aligAuxVarIter)
  {
    HmmAlignmentKey asHmm = aligAuxVarIter->first.first;
    unsigned int i = aligAuxVarIter->first.second;
    float log_suff_stat_curr = aligAuxVarIter->second.first;
    float log_suff_stat_new = aligAuxVarIter->second.second;

    // Update parameters only if current and new sufficient statistics
    // are different
    if (log_suff_stat_curr != log_suff_stat_new)
    {
      // Obtain aligNumer
      bool found;
      float numer = model.hmmAlignmentTable->getNumerator(asHmm.prev_i, asHmm.slen, i, found);
      if (!found)
        numer = SMALL_LG_NUM;

      // Obtain aligDenom
      float denom = model.hmmAlignmentTable->getDenominator(asHmm.prev_i, asHmm.slen, found);
      if (!found)
        denom = SMALL_LG_NUM;

      // Obtain new sufficient statistics
      float new_numer = obtainLogNewSuffStat(numer, log_suff_stat_curr, log_suff_stat_new);
      float new_denom = MathFuncs::lns_sublog_float(denom, numer);
      new_denom = MathFuncs::lns_sumlog_float(new_denom, new_numer);

      // Set lexical numerator and denominator
      model.hmmAlignmentTable->set(asHmm.prev_i, asHmm.slen, i, new_numer, new_denom);
    }
  }
  // Clear auxiliary variables
  incrHmmAlignmentCounts.clear();
}

float IncrHmmAlignmentTrainer::obtainLogNewSuffStat(float lcurrSuffStat, float lLocalSuffStatCurr,
                                                    float lLocalSuffStatNew)
{
  float lresult = MathFuncs::lns_sublog_float(lcurrSuffStat, lLocalSuffStatCurr);
  lresult = MathFuncs::lns_sumlog_float(lresult, lLocalSuffStatNew);
  return lresult;
}
