#include "sw_models/HmmAlignmentModel.h"

#include "nlp_common/ErrorDefs.h"

using namespace std;

HmmAlignmentModel::HmmAlignmentModel() : hmmAlignmentTable{make_shared<HmmAlignmentTable>()}
{
  lexNumDenFileExtension = ".hmm_lexnd";
}

HmmAlignmentModel::HmmAlignmentModel(Ibm1AlignmentModel& model)
    : Ibm1AlignmentModel{model}, hmmAlignmentTable{make_shared<HmmAlignmentTable>()}
{
  lexNumDenFileExtension = ".hmm_lexnd";
}

HmmAlignmentModel::HmmAlignmentModel(HmmAlignmentModel& model)
    : Ibm1AlignmentModel{model}, aligSmoothInterpFactor{model.aligSmoothInterpFactor},
      lexSmoothInterpFactor{model.lexSmoothInterpFactor}, hmmAlignmentTable{model.hmmAlignmentTable}, hmm_p0{
                                                                                                          model.hmm_p0}
{
  lexNumDenFileExtension = ".hmm_lexnd";
}

void HmmAlignmentModel::setLexSmIntFactor(double _lexSmoothInterpFactor, int verbose)
{
  lexSmoothInterpFactor = _lexSmoothInterpFactor;
  if (verbose)
    cerr << "Lexical smoothing interpolation factor has been set to " << lexSmoothInterpFactor << endl;
}

void HmmAlignmentModel::setAlSmIntFactor(double _aligSmoothInterpFactor, int verbose)
{
  aligSmoothInterpFactor = _aligSmoothInterpFactor;
  if (verbose)
    cerr << "Alignment smoothing interpolation factor has been set to " << aligSmoothInterpFactor << endl;
}

void HmmAlignmentModel::set_hmm_p0(Prob _hmm_p0)
{
  hmm_p0 = _hmm_p0;
}

void HmmAlignmentModel::startTraining(int verbosity)
{
  clearTempVars();
  vector<vector<unsigned>> insertBuffer;
  size_t insertBufferItems = 0;
  for (unsigned int n = 0; n < numSentencePairs(); ++n)
  {
    vector<WordIndex> src = getSrcSent(n);
    vector<WordIndex> trg = getTrgSent(n);

    if (sentenceLengthIsOk(src) && sentenceLengthIsOk(trg))
    {
      vector<WordIndex> nsrc = extendWithNullWord(src);

      PositionIndex slen = (PositionIndex)src.size();

      // Make room for data structure to cache alignment log-probs
      cachedAligLogProbs.makeRoomGivenSrcSentLen(slen);

      HmmAlignmentKey asHmm0{0, slen};
      hmmAlignmentTable->reserveSpace(asHmm0.prev_i, asHmm0.slen);
      HmmAlignmentCountsElem& elem = hmmAlignmentCounts[asHmm0];
      if (elem.size() < src.size())
        elem.resize(src.size(), 0);

      for (PositionIndex i = 1; i <= nsrc.size(); ++i)
      {
        if (i <= nsrc.size())
        {
          WordIndex s = nsrc[i - 1];
          lexTable->setDenominator(s, 0);
          if (s >= insertBuffer.size())
            insertBuffer.resize((size_t)s + 1);
          for (const WordIndex t : trg)
            insertBuffer[s].push_back(t);
          insertBufferItems += trg.size();
        }

        if (i <= src.size())
        {
          HmmAlignmentKey asHmm{i, slen};
          hmmAlignmentTable->reserveSpace(asHmm.prev_i, asHmm.slen);
          HmmAlignmentCountsElem& elem = hmmAlignmentCounts[asHmm];
          if (elem.size() < src.size())
            elem.resize(src.size(), 0);
        }
      }
      if (insertBufferItems > ThreadBufferSize * 100)
      {
        insertBufferItems = 0;
        addTranslationOptions(insertBuffer);
      }
    }
  }
  if (insertBufferItems > 0)
    addTranslationOptions(insertBuffer);

  if (numSentencePairs() > 0)
  {
    // Train sentence length model
    sentLengthModel->trainSentencePairRange(make_pair(0, numSentencePairs() - 1), verbosity);
  }
}

void HmmAlignmentModel::batchUpdateCounts(const vector<pair<vector<WordIndex>, vector<WordIndex>>>& pairs)
{
#pragma omp parallel for schedule(dynamic)
  for (int line_idx = 0; line_idx < (int)pairs.size(); ++line_idx)
  {
    vector<WordIndex> src = pairs[line_idx].first;
    vector<WordIndex> nsrc = extendWithNullWord(src);
    vector<WordIndex> trg = pairs[line_idx].second;

    PositionIndex slen = (PositionIndex)src.size();

    // Calculate alpha and beta matrices
    vector<vector<double>> lexLogProbs;
    vector<vector<double>> alphaMatrix;
    vector<vector<double>> betaMatrix;
    calcAlphaBetaMatrices(nsrc, trg, slen, lexLogProbs, alphaMatrix, betaMatrix);

    vector<double> lexNums(nsrc.size() + 1);
    vector<double> innerAligNums(src.size() + 1);
    vector<vector<double>> aligNums(src.size() + 1, innerAligNums);
    for (PositionIndex j = 1; j <= trg.size(); ++j)
    {
      double lexSum = INVALID_ANJI_VAL;
      double aligSum = INVALID_ANJI_VAL;
      for (PositionIndex i = 1; i <= nsrc.size(); ++i)
      {
        if (i <= nsrc.size())
        {
          // Obtain numerator
          lexNums[i] = calc_lanji_num(i, j, alphaMatrix, betaMatrix);

          // Add contribution to sum
          lexSum = lexSum == INVALID_ANJI_VAL ? lexNums[i] : MathFuncs::lns_sumlog(lexSum, lexNums[i]);
        }
        if (i <= src.size())
        {
          aligNums[i][0] = 0;
          if (j == 1)
          {
            // Obtain numerator
            if (isNullAlignment(0, slen, i))
            {
              if (isFirstNullAlignmentPar(0, slen, i))
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
            for (PositionIndex ip = 1; ip <= src.size(); ++ip)
            {
              // Obtain numerator
              if (isValidAlignment(ip, slen, i))
              {
                aligNums[i][ip] = calc_lanjm1ip_anji_num_jg1(ip, slen, i, j, lexLogProbs, alphaMatrix, betaMatrix);
              }
              else
              {
                aligNums[i][ip] = SMALL_LG_NUM;
              }

              // Add contribution to sum
              aligSum = aligSum == INVALID_ANJI_VAL ? aligNums[i][ip] : MathFuncs::lns_sumlog(aligSum, aligNums[i][ip]);
            }
          }
        }
      }
      for (PositionIndex i = 1; i <= nsrc.size(); ++i)
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
          incrementWordPairCounts(nsrc, trg, i - 1, j, exp(logLexCount));
        }

        if (i <= src.size())
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
            HmmAlignmentKey asHmm{0, slen};
#pragma omp atomic
            hmmAlignmentCounts[asHmm][i - 1] += exp(logAligCount);
          }
          else
          {
            for (PositionIndex ip = 1; ip <= src.size(); ++ip)
            {
              // Obtain information about alignment
              if (isValidAlignment(ip, slen, i))
              {
                // Obtain expected value
                double aligCount = aligNums[i][ip] - aligSum;
                // Smooth expected value
                if (aligCount > ExpValLogMax)
                  aligCount = ExpValLogMax;
                if (aligCount < ExpValLogMin)
                  aligCount = ExpValLogMin;

                // Store expected value
                HmmAlignmentKey asHmm{ip, slen};
#pragma omp atomic
                hmmAlignmentCounts[asHmm][i - 1] += exp(aligCount);
              }
            }
          }
        }
      }
    }
  }
}

void HmmAlignmentModel::batchMaximizeProbs()
{
  Ibm1AlignmentModel::batchMaximizeProbs();

#pragma omp parallel for schedule(dynamic)
  for (int asHmmIndex = 0; asHmmIndex < (int)hmmAlignmentCounts.size(); ++asHmmIndex)
  {
    double denom = 0;
    const pair<HmmAlignmentKey, HmmAlignmentCountsElem>& p = hmmAlignmentCounts.getAt(asHmmIndex);
    const HmmAlignmentKey& asHmm = p.first;
    HmmAlignmentCountsElem& elem = const_cast<HmmAlignmentCountsElem&>(p.second);
    for (PositionIndex i = 1; i <= elem.size() || i <= asHmm.slen * 2; ++i)
    {
      if (i <= elem.size())
      {
        double numer = elem[i - 1];
        denom += numer;
        float logNumer = (float)log(numer);
        hmmAlignmentTable->setNumerator(asHmm.prev_i, asHmm.slen, i, logNumer);
        elem[i - 1] = 0.0;
      }
      cachedAligLogProbs.set(asHmm.prev_i, asHmm.slen, i, (double)CACHED_HMM_ALIG_LGPROB_VIT_INVALID_VAL);
    }
    if (denom == 0)
      denom = 1;
    float logDenom = (float)log(denom);
    hmmAlignmentTable->setDenominator(asHmm.prev_i, asHmm.slen, logDenom);
  }
}

Prob HmmAlignmentModel::pts(WordIndex s, WordIndex t)
{
  double uniformProb = 1.0 / getTrgVocabSize();
  double logProb = unsmoothed_logpts(s, t);
  double prob = (1.0 - lexSmoothInterpFactor) * (logProb == SMALL_LG_NUM ? uniformProb : exp(logProb));
  double smoothProb = lexSmoothInterpFactor * uniformProb;
  return prob + smoothProb;
}

LgProb HmmAlignmentModel::logpts(WordIndex s, WordIndex t)
{
  double uniformLogProb = log(1.0 / getTrgVocabSize());
  double logProb = unsmoothed_logpts(s, t);
  logProb = log(1.0 - lexSmoothInterpFactor) + (logProb == SMALL_LG_NUM ? uniformLogProb : logProb);
  double smoothLgProb = log(lexSmoothInterpFactor) + uniformLogProb;
  return MathFuncs::lns_sumlog(logProb, smoothLgProb);
}

Prob HmmAlignmentModel::aProb(PositionIndex prev_i, PositionIndex slen, PositionIndex i)
{
  double logProb = unsmoothed_logaProb(prev_i, slen, i);
  if (isValidAlignment(prev_i, slen, i))
  {
    double uniformProb;
    if (prev_i == 0)
    {
      uniformProb = 1.0 / (double)(2 * slen);
    }
    else
    {
      uniformProb = 1.0 / (double)(slen + 1);
    }
    double prob = logProb == SMALL_LG_NUM ? uniformProb : exp(logProb);
    double aligProb = (1.0 - aligSmoothInterpFactor) * prob;
    double smoothProb = aligSmoothInterpFactor * uniformProb;
    return aligProb + smoothProb;
  }
  else
  {
    return logProb;
  }
}

LgProb HmmAlignmentModel::logaProb(PositionIndex prev_i, PositionIndex slen, PositionIndex i)
{
  double logProb = unsmoothed_logaProb(prev_i, slen, i);
  if (isValidAlignment(prev_i, slen, i))
  {
    double uniformLogProb;
    if (prev_i == 0)
    {
      uniformLogProb = log(1.0 / (double)(2 * slen));
    }
    else
    {
      uniformLogProb = log(1.0 / (double)(slen + 1));
    }
    if (logProb == SMALL_LG_NUM)
      logProb = uniformLogProb;
    LgProb aligLogProb = (LgProb)log(1.0 - aligSmoothInterpFactor) + logProb;
    double smoothLogProb = log(aligSmoothInterpFactor) + uniformLogProb;
    return MathFuncs::lns_sumlog(aligLogProb, smoothLogProb);
  }
  else
  {
    return logProb;
  }
}

LgProb HmmAlignmentModel::getBestAlignment(const vector<WordIndex>& srcSentence, const vector<WordIndex>& trgSentence,
                                           vector<PositionIndex>& bestAlignment)
{
  CachedHmmAligLgProb cached_logap;
  return getBestAlignmentCached(srcSentence, trgSentence, cached_logap, bestAlignment);
}

LgProb HmmAlignmentModel::getAlignmentLgProb(const vector<WordIndex>& srcSentence, const vector<WordIndex>& trgSentence,
                                             const WordAlignmentMatrix& aligMatrix, int verbose)
{
  PositionIndex slen = (PositionIndex)srcSentence.size();
  PositionIndex tlen = (PositionIndex)trgSentence.size();

  vector<PositionIndex> aligVec;
  aligMatrix.getAligVec(aligVec);

  if (verbose)
  {
    for (PositionIndex i = 0; i < slen; ++i)
      cerr << srcSentence[i] << " ";
    cerr << "\n";
    for (PositionIndex j = 0; j < tlen; ++j)
      cerr << trgSentence[j] << " ";
    cerr << "\n";
    for (PositionIndex j = 0; j < tlen; ++j)
      cerr << aligVec[j] << " ";
    cerr << "\n";
  }
  if (trgSentence.size() != aligVec.size())
  {
    cerr << "Error: the sentence t and the alignment vector have not the same size." << endl;
    return THOT_ERROR;
  }
  else
  {
    AlignmentInfo alignment(slen, tlen);
    alignment.setAlignment(aligVec);
    CachedHmmAligLgProb cached_logap;
    return getSentenceLengthLgProb(slen, tlen)
         + calcProbOfAlignment(cached_logap, srcSentence, trgSentence, alignment, verbose).get_lp();
  }
}

LgProb HmmAlignmentModel::getSumLgProb(const vector<WordIndex>& srcSentence, const vector<WordIndex>& trgSentence,
                                       int verbose)
{
  if (sentenceLengthIsOk(srcSentence) && sentenceLengthIsOk(trgSentence))
  {
    // Calculate sentence length model lgprob
    LgProb slp = getSentenceLengthLgProb(srcSentence.size(), trgSentence.size());

    // Obtain extended source vector
    vector<WordIndex> nSrcSentIndexVector = extendWithNullWord(srcSentence);

    // Calculate hmm lgprob
    LgProb flp = forwardAlgorithm(nSrcSentIndexVector, trgSentence, verbose);

    if (verbose)
      cerr << "lp= " << slp + flp << " ; slm_lp= " << slp << " ; lp-slm_lp= " << flp << endl;

    return slp + flp;
  }
  else
  {
    return SMALL_LG_NUM;
  }
}

Prob HmmAlignmentModel::searchForBestAlignment(PositionIndex maxFertility, const vector<WordIndex>& src,
                                               const vector<WordIndex>& trg, AlignmentInfo& bestAlignment,
                                               Matrix<double>* moveScores, Matrix<double>* swapScores)
{
  PositionIndex slen = (PositionIndex)src.size();
  PositionIndex tlen = (PositionIndex)trg.size();

  // Call function to obtain best lgprob and viterbi alignment
  vector<vector<double>> vitMatrix;
  vector<vector<PositionIndex>> predMatrix;
  viterbiAlgorithmCached(extendWithNullWord(src), trg, cachedAligLogProbs, vitMatrix, predMatrix);
  vector<PositionIndex> aligVec;
  double vit_lp = bestAligGivenVitMatrices(slen, vitMatrix, predMatrix, aligVec);
  bestAlignment.setAlignment(aligVec);
  bestAlignment.setProb(exp(vit_lp));

  if (moveScores != nullptr || swapScores != nullptr)
  {
    if (moveScores != nullptr)
      moveScores->resize(slen + 1, tlen + 1);
    if (swapScores != nullptr)
      swapScores->resize(tlen + 1, tlen + 1);

    for (PositionIndex j = 1; j <= tlen; j++)
    {
      PositionIndex iAlig = bestAlignment.get(j);

      if (swapScores != nullptr)
      {
        // swap alignments
        for (PositionIndex j1 = j + 1; j1 <= tlen; j1++)
        {
          if (iAlig != bestAlignment.get(j1))
          {
            double changeScore = swapScore(cachedAligLogProbs, src, trg, j, j1, bestAlignment);
            swapScores->set(j, j1, changeScore);
          }
          else
          {
            swapScores->set(j, j1, 1.0);
          }
        }
      }

      if (moveScores != nullptr)
      {
        // move alignment by one position
        for (PositionIndex i = 0; i <= slen; i++)
        {
          if (i != iAlig && (i != 0 || (tlen >= 2 * (bestAlignment.getFertility(0) + 1)))
              && bestAlignment.getFertility(i) + 1 < maxFertility)
          {
            double changeScore = moveScore(cachedAligLogProbs, src, trg, i, j, bestAlignment);
            moveScores->set(i, j, changeScore);
          }
          else
          {
            moveScores->set(i, j, 1.0);
          }
        }
      }
    }
  }

  return bestAlignment.getProb();
}

bool HmmAlignmentModel::load(const char* prefFileName, int verbose)
{
  if (prefFileName[0] != 0)
  {
    bool retVal;

    // Load IBM 1 Model data
    retVal = Ibm1AlignmentModel::load(prefFileName, verbose);
    if (retVal == THOT_ERROR)
      return THOT_ERROR;

    if (verbose)
      cerr << "Loading incremental HMM Model data..." << endl;

    // Load file with alignment nd values
    string aligNumDenFile = prefFileName;
    aligNumDenFile = aligNumDenFile + ".hmm_alignd";
    retVal = hmmAlignmentTable->load(aligNumDenFile.c_str(), verbose);
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

    // Load file with hmm p0 value
    std::string hmmP0File = prefFileName;
    hmmP0File = hmmP0File + ".hmm_p0";
    retVal = loadHmmP0(hmmP0File.c_str(), verbose);

    return retVal;
  }
  else
    return THOT_ERROR;
}

bool HmmAlignmentModel::print(const char* prefFileName, int verbose)
{
  bool retVal;

  // Print IBM 1 Model data
  retVal = Ibm1AlignmentModel::print(prefFileName);
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  // Print file with alignment nd values
  string aligNumDenFile = prefFileName;
  aligNumDenFile = aligNumDenFile + ".hmm_alignd";
  retVal = hmmAlignmentTable->print(aligNumDenFile.c_str());
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

  // Print file with hmm p0 value
  std::string hmmP0File = prefFileName;
  hmmP0File = hmmP0File + ".hmm_p0";
  retVal = printHmmP0(hmmP0File.c_str());

  return retVal;
}

void HmmAlignmentModel::clear()
{
  Ibm1AlignmentModel::clear();
  hmmAlignmentTable->clear();
  hmm_p0 = DEFAULT_HMM_P0;
}

void HmmAlignmentModel::clearTempVars()
{
  Ibm1AlignmentModel::clearTempVars();
  hmmAlignmentCounts.clear();
  cachedAligLogProbs.clear();
}

LgProb HmmAlignmentModel::getBestAlignmentCached(const vector<WordIndex>& srcSentence,
                                                 const vector<WordIndex>& trgSentence,
                                                 CachedHmmAligLgProb& cached_logap,
                                                 vector<PositionIndex>& bestAlignment)
{
  if (sentenceLengthIsOk(srcSentence) && sentenceLengthIsOk(trgSentence))
  {
    // Obtain extended source vector
    vector<WordIndex> nSrcSentIndexVector = extendWithNullWord(srcSentence);
    // Call function to obtain best lgprob and viterbi alignment
    vector<vector<double>> vitMatrix;
    vector<vector<PositionIndex>> predMatrix;
    viterbiAlgorithmCached(nSrcSentIndexVector, trgSentence, cached_logap, vitMatrix, predMatrix);
    LgProb vit_lp = bestAligGivenVitMatrices(srcSentence.size(), vitMatrix, predMatrix, bestAlignment);

    // Calculate sentence length model lgprob
    LgProb slm_lp = getSentenceLengthLgProb(srcSentence.size(), trgSentence.size());

    return slm_lp + vit_lp;
  }
  else
  {
    bestAlignment.resize(trgSentence.size(), 0);
    return SMALL_LG_NUM;
  }
}

void HmmAlignmentModel::viterbiAlgorithm(const vector<WordIndex>& nSrcSentIndexVector,
                                         const vector<WordIndex>& trgSentIndexVector, vector<vector<double>>& vitMatrix,
                                         vector<vector<PositionIndex>>& predMatrix)
{
  CachedHmmAligLgProb cached_logap;
  viterbiAlgorithmCached(nSrcSentIndexVector, trgSentIndexVector, cached_logap, vitMatrix, predMatrix);
}

void HmmAlignmentModel::viterbiAlgorithmCached(const vector<WordIndex>& nSrcSentIndexVector,
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

double HmmAlignmentModel::bestAligGivenVitMatricesRaw(const vector<vector<double>>& vitMatrix,
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

double HmmAlignmentModel::bestAligGivenVitMatrices(PositionIndex slen, const vector<vector<double>>& vitMatrix,
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

double HmmAlignmentModel::forwardAlgorithm(const vector<WordIndex>& nSrcSentIndexVector,
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

double HmmAlignmentModel::lgProbGivenForwardMatrix(const vector<vector<double>>& forwardMatrix)
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

PositionIndex HmmAlignmentModel::getSrcLen(const vector<WordIndex>& nsrcWordIndexVec)
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

Prob HmmAlignmentModel::calcProbOfAlignment(CachedHmmAligLgProb& cached_logap, const vector<WordIndex>& src,
                                            const vector<WordIndex>& trg, AlignmentInfo& alignment, int verbose)
{
  if (alignment.getProb() >= 0.0)
    return alignment.getProb();

  PositionIndex slen = alignment.getSourceLength();

  double logProb = 0;
  PositionIndex prev_i = 0;
  for (PositionIndex j = 1; j <= trg.size(); ++j)
  {
    PositionIndex i = alignment.get(j);
    WordIndex s = i == 0 ? NULL_WORD : src[i - 1];
    WordIndex t = trg[j - 1];
    if (!cached_logap.isDefined(prev_i, slen, i))
      cached_logap.set_boundary_check(prev_i, slen, i, logaProb(prev_i, slen, i));
    logProb += cached_logap.get(prev_i, slen, i) + double{logpts(s, t)};
    prev_i = i;
  }
  double prob = exp(logProb);
  alignment.setProb(prob);
  return prob;
}

double HmmAlignmentModel::swapScore(CachedHmmAligLgProb& cached_logap, const vector<WordIndex>& src,
                                    const vector<WordIndex>& trg, PositionIndex j1, PositionIndex j2,
                                    AlignmentInfo& alignment)
{
  PositionIndex i1 = alignment.get(j1);
  PositionIndex i2 = alignment.get(j2);
  if (i1 == i2)
    return 1.0;

  Prob oldProb = calcProbOfAlignment(cached_logap, src, trg, alignment);

  alignment.set(j1, i2);
  alignment.set(j2, i1);
  Prob newProb = calcProbOfAlignment(cached_logap, src, trg, alignment);
  alignment.set(j1, i1);
  alignment.set(j2, i2);
  alignment.setProb(oldProb);

  return newProb / oldProb;
}

double HmmAlignmentModel::moveScore(CachedHmmAligLgProb& cached_logap, const vector<WordIndex>& src,
                                    const vector<WordIndex>& trg, PositionIndex iNew, PositionIndex j,
                                    AlignmentInfo& alignment)
{
  PositionIndex iOld = alignment.get(j);

  Prob oldProb = calcProbOfAlignment(cached_logap, src, trg, alignment);

  alignment.set(j, iNew);
  Prob newProb = calcProbOfAlignment(cached_logap, src, trg, alignment);
  alignment.set(j, iOld);
  alignment.setProb(oldProb);

  return newProb / oldProb;
}

vector<WordIndex> HmmAlignmentModel::extendWithNullWord(const vector<WordIndex>& srcWordIndexVec)
{
  // Initialize result using srcWordIndexVec
  vector<WordIndex> result = srcWordIndexVec;

  // Add NULL words
  WordIndex nullWidx = stringToSrcWordIndex(NULL_WORD_STR);
  for (unsigned int i = 0; i < srcWordIndexVec.size(); ++i)
    result.push_back(nullWidx);

  return result;
}

void HmmAlignmentModel::calcAlphaBetaMatrices(const vector<WordIndex>& nsrcSent, const vector<WordIndex>& trgSent,
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

bool HmmAlignmentModel::isFirstNullAlignmentPar(PositionIndex ip, unsigned int slen, PositionIndex i)
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

double HmmAlignmentModel::calc_lanji_num(PositionIndex i, PositionIndex j, const vector<vector<double>>& alphaMatrix,
                                         const vector<vector<double>>& betaMatrix)
{
  double result = alphaMatrix[i][j] + betaMatrix[i][j];
  if (result < SMALL_LG_NUM)
    result = SMALL_LG_NUM;
  return result;
}

double HmmAlignmentModel::calc_lanjm1ip_anji_num_je1(PositionIndex slen, PositionIndex i,
                                                     const vector<vector<double>>& lexLogProbs,
                                                     const vector<vector<double>>& betaMatrix)
{
  double result = cached_logaProb(0, slen, i) + lexLogProbs[i][1] + betaMatrix[i][1];
  if (result < SMALL_LG_NUM)
    result = SMALL_LG_NUM;
  return result;
}

double HmmAlignmentModel::calc_lanjm1ip_anji_num_jg1(PositionIndex ip, PositionIndex slen, PositionIndex i,
                                                     PositionIndex j, const vector<vector<double>>& lexLogProbs,
                                                     const vector<vector<double>>& alphaMatrix,
                                                     const vector<vector<double>>& betaMatrix)
{
  double result = alphaMatrix[ip][j - 1] + cached_logaProb(ip, slen, i) + lexLogProbs[i][j] + betaMatrix[i][j];
  if (result < SMALL_LG_NUM)
    result = SMALL_LG_NUM;
  return result;
}

double HmmAlignmentModel::unsmoothed_logaProb(PositionIndex prev_i, PositionIndex slen, PositionIndex i)
{
  HmmAligInfo hmmAligInfo;
  getHmmAlignmentInfo(prev_i, slen, i, hmmAligInfo);
  if (!hmmAligInfo.validAlig)
  {
    return SMALL_LG_NUM;
  }
  else
  {
    if (hmmAligInfo.nullAlig)
    {
      if (prev_i == 0)
      {
        return log((double)hmm_p0) - log((double)slen);
      }
      else
        return log((double)hmm_p0);
    }
    else
    {
      bool found;
      double numer = hmmAlignmentTable->getNumerator(hmmAligInfo.modified_ip, slen, i, found);
      if (found)
      {
        // aligNumer for pair asHmm,i exists
        double denom = hmmAlignmentTable->getDenominator(hmmAligInfo.modified_ip, slen, found);
        if (!found)
          return SMALL_LG_NUM;
        else
        {
          LgProb lp = numer - denom;
          return lp + log((double)1.0 - (double)hmm_p0);
        }
      }
      else
      {
        // aligNumer for pair asHmm,i does not exist
        return SMALL_LG_NUM;
      }
    }
  }
}

double HmmAlignmentModel::cached_logaProb(PositionIndex prev_i, PositionIndex slen, PositionIndex i)
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

void HmmAlignmentModel::getHmmAlignmentInfo(PositionIndex ip, PositionIndex slen, PositionIndex i,
                                            HmmAligInfo& hmmAligInfo)
{
  hmmAligInfo.validAlig = isValidAlignment(ip, slen, i);
  if (hmmAligInfo.validAlig)
  {
    hmmAligInfo.nullAlig = isNullAlignment(ip, slen, i);
    hmmAligInfo.modified_ip = getModifiedIp(ip, slen, i);
  }
  else
  {
    hmmAligInfo.nullAlig = false;
    hmmAligInfo.modified_ip = ip;
  }
}

bool HmmAlignmentModel::isValidAlignment(PositionIndex ip, PositionIndex slen, PositionIndex i)
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

bool HmmAlignmentModel::isNullAlignment(PositionIndex ip, PositionIndex slen, PositionIndex i)
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

PositionIndex HmmAlignmentModel::getModifiedIp(PositionIndex ip, PositionIndex slen, PositionIndex i)
{
  if (i <= slen && ip > slen)
  {
    return ip - slen;
  }
  else
    return ip;
}

bool HmmAlignmentModel::loadLexSmIntFactor(const char* lexSmIntFactorFile, int verbose)
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

bool HmmAlignmentModel::printLexSmIntFactor(const char* lexSmIntFactorFile, int verbose)
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

bool HmmAlignmentModel::loadAlSmIntFactor(const char* alSmIntFactorFile, int verbose)
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

bool HmmAlignmentModel::printAlSmIntFactor(const char* alSmIntFactorFile, int verbose)
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

bool HmmAlignmentModel::loadHmmP0(const char* hmmP0FileName, int verbose)
{
  if (verbose)
    std::cerr << "Loading file with hmm p0 value from " << hmmP0FileName << std::endl;

  AwkInputStream awk;

  if (awk.open(hmmP0FileName) == THOT_ERROR)
  {
    if (verbose)
      std::cerr << "Error in file with hmm p0 value, file " << hmmP0FileName
                << " does not exist. Assuming hmm_p0=" << DEFAULT_HMM_P0 << "\n";
    hmm_p0 = DEFAULT_HMM_P0;
    return THOT_OK;
  }
  else
  {
    if (awk.getln())
    {
      if (awk.NF == 1)
      {
        hmm_p0 = (Prob)atof(awk.dollar(1).c_str());
        if (verbose)
          std::cerr << "hmm p0 value has been set to " << hmm_p0 << std::endl;
        return THOT_OK;
      }
      else
      {
        if (verbose)
          std::cerr << "Error: anomalous .hmm_p0 file, " << hmmP0FileName << std::endl;
        return THOT_ERROR;
      }
    }
    else
    {
      if (verbose)
        std::cerr << "Error: anomalous .hmm_p0 file, " << hmmP0FileName << std::endl;
      return THOT_ERROR;
    }
  }
}

bool HmmAlignmentModel::printHmmP0(const char* hmmP0FileName)
{
  std::ofstream outF;
  outF.open(hmmP0FileName, std::ios::out);
  if (!outF)
  {
    std::cerr << "Error while printing file with hmm p0 value." << std::endl;
    return THOT_ERROR;
  }
  else
  {
    outF << hmm_p0 << std::endl;
    return THOT_OK;
  }
}
