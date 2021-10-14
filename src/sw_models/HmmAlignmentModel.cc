#include "sw_models/HmmAlignmentModel.h"

#include "nlp_common/ErrorDefs.h"
#include "sw_models/SwDefs.h"

HmmAlignmentModel::HmmAlignmentModel() : hmmAlignmentTable{std::make_shared<HmmAlignmentTable>()}
{
  lexNumDenFileExtension = ".hmm_lexnd";
  maxSentenceLength = MaxSentenceLength;
}

HmmAlignmentModel::HmmAlignmentModel(Ibm1AlignmentModel& model)
    : Ibm2AlignmentModel{model}, hmmAlignmentTable{std::make_shared<HmmAlignmentTable>()}
{
  lexNumDenFileExtension = ".hmm_lexnd";
  maxSentenceLength = MaxSentenceLength;
}

HmmAlignmentModel::HmmAlignmentModel(HmmAlignmentModel& model)
    : Ibm2AlignmentModel{model}, hmmAlignmentSmoothFactor{model.hmmAlignmentSmoothFactor},
      lexicalSmoothFactor{model.lexicalSmoothFactor}, hmmP0{model.hmmP0}, hmmAlignmentTable{model.hmmAlignmentTable}
{
  lexNumDenFileExtension = ".hmm_lexnd";
  maxSentenceLength = MaxSentenceLength;
}

double HmmAlignmentModel::getLexicalSmoothFactor()
{
  return lexicalSmoothFactor;
}

void HmmAlignmentModel::setLexicalSmoothFactor(double factor)
{
  lexicalSmoothFactor = factor;
}

double HmmAlignmentModel::getHmmAlignmentSmoothFactor()
{
  return hmmAlignmentSmoothFactor;
}

void HmmAlignmentModel::setHmmAlignmentSmoothFactor(double factor)
{
  hmmAlignmentSmoothFactor = factor;
}

Prob HmmAlignmentModel::getHmmP0()
{
  return hmmP0;
}

void HmmAlignmentModel::setHmmP0(Prob p0)
{
  hmmP0 = p0;
}

unsigned int HmmAlignmentModel::startTraining(int verbosity)
{
  clearTempVars();
  std::vector<std::vector<unsigned>> insertBuffer;
  size_t insertBufferItems = 0;
  unsigned int count = 0;
  for (unsigned int n = 0; n < numSentencePairs(); ++n)
  {
    std::vector<WordIndex> src = getSrcSent(n);
    std::vector<WordIndex> trg = getTrgSent(n);

    if (sentenceLengthIsOk(src) && sentenceLengthIsOk(trg))
    {
      std::vector<WordIndex> nsrc = extendWithNullWord(src);

      PositionIndex slen = (PositionIndex)src.size();
      PositionIndex tlen = (PositionIndex)trg.size();

      HmmAlignmentKey asHmm0{0, getCompactedSentenceLength(slen)};
      hmmAlignmentTable->reserveSpace(asHmm0.prev_i, asHmm0.slen);
      HmmAlignmentCountsElem& elem = hmmAlignmentCounts[asHmm0];
      if (elem.size() < src.size())
        elem.resize(src.size(), 0);

      for (PositionIndex i = 1; i <= nsrc.size(); ++i)
      {
        WordIndex s = nsrc[i - 1];
        if (s >= insertBuffer.size())
          insertBuffer.resize((size_t)s + 1);
        for (const WordIndex t : trg)
          insertBuffer[s].push_back(t);
        insertBufferItems += trg.size();

        if (i <= slen)
        {
          HmmAlignmentKey asHmm{i, getCompactedSentenceLength(slen)};
          hmmAlignmentTable->reserveSpace(asHmm.prev_i, asHmm.slen);
          HmmAlignmentCountsElem& elem = hmmAlignmentCounts[asHmm];
          if (elem.size() < src.size())
            elem.resize(src.size(), 0);
        }
      }

      for (PositionIndex j = 1; j <= trg.size(); ++j)
      {
        alignmentTable->reserveSpace(j, slen, getCompactedSentenceLength(tlen));

        AlignmentKey key{j, slen, getCompactedSentenceLength(tlen)};
        AlignmentCountsElem& elem = alignmentCounts[key];
        if (elem.size() < src.size() + 1)
          elem.resize(src.size() + 1, 0);
      }

      if (insertBufferItems > ThreadBufferSize * 100)
      {
        insertBufferItems = 0;
        addTranslationOptions(insertBuffer);
      }
      ++count;
    }
  }
  if (insertBufferItems > 0)
    addTranslationOptions(insertBuffer);

  if (numSentencePairs() > 0)
  {
    // Train sentence length model
    sentLengthModel->trainSentencePairRange(std::make_pair(0, numSentencePairs() - 1), verbosity);
  }
  return count;
}

void HmmAlignmentModel::batchUpdateCounts(
    const std::vector<std::pair<std::vector<WordIndex>, std::vector<WordIndex>>>& pairs)
{
#pragma omp parallel for schedule(dynamic)
  for (int line_idx = 0; line_idx < (int)pairs.size(); ++line_idx)
  {
    std::vector<WordIndex> src = pairs[line_idx].first;
    std::vector<WordIndex> nsrc = extendWithNullWord(src);
    std::vector<WordIndex> trg = pairs[line_idx].second;

    PositionIndex slen = (PositionIndex)src.size();
    PositionIndex tlen = (PositionIndex)trg.size();

    // Calculate alpha and beta matrices
    std::vector<std::vector<double>> lexProbs;
    std::vector<std::vector<double>> alignProbs;
    std::vector<std::vector<double>> alphaMatrix;
    std::vector<std::vector<double>> betaMatrix;
    calcAlphaBetaMatrices(nsrc, trg, slen, lexProbs, alignProbs, alphaMatrix, betaMatrix);

    std::vector<double> lexNums(nsrc.size() + 1);
    std::vector<double> innerAligNums(src.size() + 1);
    std::vector<std::vector<double>> aligNums(src.size() + 1, innerAligNums);
    for (PositionIndex j = 1; j <= trg.size(); ++j)
    {
      double lexSum = 0;
      double aligSum = 0;
      for (PositionIndex i = 1; i <= nsrc.size(); ++i)
      {
        // Obtain numerator
        lexNums[i] = alphaMatrix[i][j] * betaMatrix[i][j];

        // Add contribution to sum
        lexSum += lexNums[i];

        if (i <= slen)
        {
          aligNums[i][0] = 1.0;
          if (j == 1)
          {
            // Obtain numerator
            if (isNullAlignment(0, slen, i))
            {
              if (isFirstNullAlignmentPar(0, slen, i))
                aligNums[i][0] = alignProbs[i][0] * lexProbs[i][1] * betaMatrix[i][1];
              else
                aligNums[i][0] = aligNums[size_t{slen} + 1][0];
            }
            else
            {
              aligNums[i][0] = alignProbs[i][0] * lexProbs[i][1] * betaMatrix[i][1];
            }

            // Add contribution to sum
            aligSum += aligNums[i][0];
          }
          else
          {
            for (PositionIndex ip = 1; ip <= src.size(); ++ip)
            {
              // Obtain numerator
              if (isValidAlignment(ip, slen, i))
                aligNums[i][ip] = alphaMatrix[ip][j - 1] * alignProbs[i][ip] * lexProbs[i][j] * betaMatrix[i][j];
              else
                aligNums[i][ip] = 0.0;

              // Add contribution to sum
              aligSum += aligNums[i][ip];
            }
          }
        }
      }
      for (PositionIndex i = 1; i <= nsrc.size(); ++i)
      {
        // Obtain expected value
        double lexCount = lexSum == 0 ? 0 : lexNums[i] / lexSum;
        if (lexCount > ExpValMax)
          lexCount = ExpValMax;
        if (lexCount < ExpValMin)
          lexCount = ExpValMin;

        // Store expected value
        WordIndex s = nsrc[i - 1];
        WordIndex t = trg[j - 1];

#pragma omp atomic
        lexCounts[s].find(t)->second += lexCount;

        AlignmentKey key{j, slen, getCompactedSentenceLength(tlen)};
        PositionIndex ibm2_i = i > slen ? 0 : i;

#pragma omp atomic
        alignmentCounts[key][ibm2_i] += lexCount;

        if (i <= slen)
        {
          if (j == 1)
          {
            // Obtain expected value
            double aligCount = aligSum == 0 ? 0 : aligNums[i][0] / aligSum;
            if (aligCount > ExpValMax)
              aligCount = ExpValMax;
            if (aligCount < ExpValMin)
              aligCount = ExpValMin;

            // Store expected value
            HmmAlignmentKey asHmm{0, getCompactedSentenceLength(slen)};
#pragma omp atomic
            hmmAlignmentCounts[asHmm][i - 1] += aligCount * slen;
          }
          else
          {
            for (PositionIndex ip = 1; ip <= src.size(); ++ip)
            {
              // Obtain information about alignment
              if (isValidAlignment(ip, slen, i))
              {
                // Obtain expected value
                double aligCount = aligSum == 0 ? 0 : aligNums[i][ip] / aligSum;
                if (aligCount > ExpValMax)
                  aligCount = ExpValMax;
                if (aligCount < ExpValMin)
                  aligCount = ExpValMin;

                // Store expected value
                HmmAlignmentKey asHmm{ip, getCompactedSentenceLength(slen)};
#pragma omp atomic
                hmmAlignmentCounts[asHmm][i - 1] += aligCount * slen;
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
  Ibm2AlignmentModel::batchMaximizeProbs();

#pragma omp parallel for schedule(dynamic)
  for (int asHmmIndex = 0; asHmmIndex < (int)hmmAlignmentCounts.size(); ++asHmmIndex)
  {
    double denom = 0;
    const std::pair<HmmAlignmentKey, HmmAlignmentCountsElem>& p = hmmAlignmentCounts.getAt(asHmmIndex);
    const HmmAlignmentKey& asHmm = p.first;
    HmmAlignmentCountsElem& elem = const_cast<HmmAlignmentCountsElem&>(p.second);
    for (PositionIndex i = 1; i <= elem.size(); ++i)
    {
      double numer = elem[i - 1];
      denom += numer;
      float logNumer = (float)log(numer);
      hmmAlignmentTable->setNumerator(asHmm.prev_i, asHmm.slen, i, logNumer);
      elem[i - 1] = 0.0;
    }
    if (denom == 0)
      denom = 1;
    float logDenom = (float)log(denom);
    hmmAlignmentTable->setDenominator(asHmm.prev_i, asHmm.slen, logDenom);
  }
}

Prob HmmAlignmentModel::translationProb(WordIndex s, WordIndex t)
{
  double uniformProb = 1.0 / getTrgVocabSize();
  double logProb = unsmoothedTranslationLogProb(s, t);
  double prob = (1.0 - lexicalSmoothFactor) * (logProb == SMALL_LG_NUM ? uniformProb : exp(logProb));
  double smoothProb = lexicalSmoothFactor * uniformProb;
  return prob + smoothProb;
}

LgProb HmmAlignmentModel::translationLogProb(WordIndex s, WordIndex t)
{
  double uniformLogProb = log(1.0 / getTrgVocabSize());
  double logProb = unsmoothedTranslationLogProb(s, t);
  logProb = log(1.0 - lexicalSmoothFactor) + (logProb == SMALL_LG_NUM ? uniformLogProb : logProb);
  double smoothLgProb = log(lexicalSmoothFactor) + uniformLogProb;
  return MathFuncs::lns_sumlog(logProb, smoothLgProb);
}

Prob HmmAlignmentModel::hmmAlignmentProb(PositionIndex prev_i, PositionIndex slen, PositionIndex i)
{
  double logProb = unsmoothedHmmAlignmentLogProb(prev_i, slen, i);
  if (isValidAlignment(prev_i, slen, i))
  {
    double uniformProb;
    if (prev_i == 0)
    {
      uniformProb = 1.0 / (2.0 * slen);
    }
    else
    {
      uniformProb = 1.0 / (slen + 1.0);
    }
    double prob = logProb == SMALL_LG_NUM ? uniformProb : exp(logProb);
    double aligProb = (1.0 - hmmAlignmentSmoothFactor) * prob;
    double smoothProb = hmmAlignmentSmoothFactor * uniformProb;
    return aligProb + smoothProb;
  }
  else
  {
    return 0;
  }
}

LgProb HmmAlignmentModel::hmmAlignmentLogProb(PositionIndex prev_i, PositionIndex slen, PositionIndex i)
{
  double logProb = unsmoothedHmmAlignmentLogProb(prev_i, slen, i);
  if (isValidAlignment(prev_i, slen, i))
  {
    double uniformLogProb;
    if (prev_i == 0)
    {
      uniformLogProb = log(1.0 / (2.0 * slen));
    }
    else
    {
      uniformLogProb = log(1.0 / (slen + 1.0));
    }
    if (logProb == SMALL_LG_NUM)
      logProb = uniformLogProb;
    LgProb aligLogProb = (LgProb)log(1.0 - hmmAlignmentSmoothFactor) + logProb;
    double smoothLogProb = log(hmmAlignmentSmoothFactor) + uniformLogProb;
    return MathFuncs::lns_sumlog(aligLogProb, smoothLogProb);
  }
  else
  {
    return logProb;
  }
}

LgProb HmmAlignmentModel::getBestAlignment(const std::vector<WordIndex>& srcSentence,
                                           const std::vector<WordIndex>& trgSentence,
                                           std::vector<PositionIndex>& bestAlignment)
{
  CachedHmmAligLgProb cached_logap;
  return getBestAlignmentCached(srcSentence, trgSentence, cached_logap, bestAlignment);
}

LgProb HmmAlignmentModel::computeLogProb(const std::vector<WordIndex>& srcSentence,
                                         const std::vector<WordIndex>& trgSentence,
                                         const WordAlignmentMatrix& aligMatrix, int verbose)
{
  PositionIndex slen = (PositionIndex)srcSentence.size();
  PositionIndex tlen = (PositionIndex)trgSentence.size();

  std::vector<PositionIndex> aligVec;
  aligMatrix.getAligVec(aligVec);

  if (verbose)
  {
    for (PositionIndex i = 0; i < slen; ++i)
      std::cerr << srcSentence[i] << " ";
    std::cerr << "\n";
    for (PositionIndex j = 0; j < tlen; ++j)
      std::cerr << trgSentence[j] << " ";
    std::cerr << "\n";
    for (PositionIndex j = 0; j < tlen; ++j)
      std::cerr << aligVec[j] << " ";
    std::cerr << "\n";
  }
  if (trgSentence.size() != aligVec.size())
  {
    std::cerr << "Error: the sentence t and the alignment vector have not the same size." << std::endl;
    return THOT_ERROR;
  }
  else
  {
    AlignmentInfo alignment(slen, tlen);
    alignment.setAlignment(aligVec);
    CachedHmmAligLgProb cached_logap;
    return sentenceLengthLogProb(slen, tlen)
         + calcProbOfAlignment(cached_logap, srcSentence, trgSentence, alignment, verbose).get_lp();
  }
}

LgProb HmmAlignmentModel::computeSumLogProb(const std::vector<WordIndex>& srcSentence,
                                            const std::vector<WordIndex>& trgSentence, int verbose)
{
  if (sentenceLengthIsOk(srcSentence) && sentenceLengthIsOk(trgSentence))
  {
    // Calculate sentence length model lgprob
    LgProb slp = sentenceLengthLogProb(srcSentence.size(), trgSentence.size());

    // Obtain extended source vector
    std::vector<WordIndex> nSrcSentIndexVector = extendWithNullWord(srcSentence);

    // Calculate hmm lgprob
    LgProb flp = forwardAlgorithm(nSrcSentIndexVector, trgSentence, verbose);

    if (verbose)
      std::cerr << "lp= " << slp + flp << " ; slm_lp= " << slp << " ; lp-slm_lp= " << flp << std::endl;

    return slp + flp;
  }
  else
  {
    return SMALL_LG_NUM;
  }
}

Prob HmmAlignmentModel::searchForBestAlignment(const std::vector<WordIndex>& src, const std::vector<WordIndex>& trg,
                                               AlignmentInfo& bestAlignment, CachedHmmAligLgProb& cachedAligLogProbs)
{
  PositionIndex slen = (PositionIndex)src.size();

  // Call function to obtain best lgprob and viterbi alignment
  std::vector<std::vector<double>> vitMatrix;
  std::vector<std::vector<PositionIndex>> predMatrix;
  viterbiAlgorithmCached(extendWithNullWord(src), trg, cachedAligLogProbs, vitMatrix, predMatrix);
  std::vector<PositionIndex> aligVec;
  double vit_lp = bestAligGivenVitMatrices(slen, vitMatrix, predMatrix, aligVec);
  bestAlignment.setAlignment(aligVec);

  return exp(vit_lp);
}

void HmmAlignmentModel::populateMoveSwapScores(const std::vector<WordIndex>& src, const std::vector<WordIndex>& trg,
                                               AlignmentInfo& bestAlignment, double alignmentProb,
                                               CachedHmmAligLgProb& cachedAligLogProbs, Matrix<double>& moveScores,
                                               Matrix<double>& swapScores)
{
  PositionIndex slen = (PositionIndex)src.size();
  PositionIndex tlen = (PositionIndex)trg.size();

  moveScores.resize(slen + 1, tlen + 1);
  swapScores.resize(tlen + 1, tlen + 1);

  for (PositionIndex j = 1; j <= tlen; j++)
  {
    PositionIndex iAlig = bestAlignment.get(j);

    // swap alignments
    for (PositionIndex j1 = j + 1; j1 <= tlen; j1++)
    {
      if (iAlig != bestAlignment.get(j1))
      {
        double changeScore = swapScore(cachedAligLogProbs, src, trg, j, j1, bestAlignment, alignmentProb);
        swapScores.set(j, j1, changeScore);
      }
      else
      {
        swapScores.set(j, j1, 1.0);
      }
    }

    // move alignment by one position
    for (PositionIndex i = 0; i <= slen; i++)
    {
      if (i != iAlig)
      {
        double changeScore = moveScore(cachedAligLogProbs, src, trg, i, j, bestAlignment, alignmentProb);
        moveScores.set(i, j, changeScore);
      }
      else
      {
        moveScores.set(i, j, 1.0);
      }
    }
  }
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
      std::cerr << "Loading HMM Model data..." << std::endl;

    // Load file with alignment nd values
    std::string aligNumDenFile = prefFileName;
    aligNumDenFile = aligNumDenFile + ".hmm_alignd";
    retVal = hmmAlignmentTable->load(aligNumDenFile.c_str(), verbose);
    if (retVal == THOT_ERROR)
      return THOT_ERROR;

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
  std::string aligNumDenFile = prefFileName;
  aligNumDenFile = aligNumDenFile + ".hmm_alignd";
  retVal = hmmAlignmentTable->print(aligNumDenFile.c_str());
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  return THOT_OK;
}

void HmmAlignmentModel::clear()
{
  Ibm2AlignmentModel::clear();
  hmmAlignmentTable->clear();
  hmmAlignmentSmoothFactor = DefaultHmmAlignmentSmoothFactor;
  lexicalSmoothFactor = DefaultLexicalSmoothFactor;
  hmmP0 = DefaultHmmP0;
}

void HmmAlignmentModel::clearTempVars()
{
  Ibm2AlignmentModel::clearTempVars();
  hmmAlignmentCounts.clear();
}

LgProb HmmAlignmentModel::getBestAlignmentCached(const std::vector<WordIndex>& srcSentence,
                                                 const std::vector<WordIndex>& trgSentence,
                                                 CachedHmmAligLgProb& cached_logap,
                                                 std::vector<PositionIndex>& bestAlignment)
{
  if (sentenceLengthIsOk(srcSentence) && sentenceLengthIsOk(trgSentence))
  {
    // Obtain extended source vector
    std::vector<WordIndex> nSrcSentIndexVector = extendWithNullWord(srcSentence);
    // Call function to obtain best lgprob and viterbi alignment
    std::vector<std::vector<double>> vitMatrix;
    std::vector<std::vector<PositionIndex>> predMatrix;
    viterbiAlgorithmCached(nSrcSentIndexVector, trgSentence, cached_logap, vitMatrix, predMatrix);
    LgProb vit_lp = bestAligGivenVitMatrices(srcSentence.size(), vitMatrix, predMatrix, bestAlignment);

    // Calculate sentence length model lgprob
    LgProb slm_lp = sentenceLengthLogProb(srcSentence.size(), trgSentence.size());

    return slm_lp + vit_lp;
  }
  else
  {
    bestAlignment.resize(trgSentence.size(), 0);
    return SMALL_LG_NUM;
  }
}

void HmmAlignmentModel::viterbiAlgorithm(const std::vector<WordIndex>& nSrcSentIndexVector,
                                         const std::vector<WordIndex>& trgSentIndexVector,
                                         std::vector<std::vector<double>>& vitMatrix,
                                         std::vector<std::vector<PositionIndex>>& predMatrix)
{
  CachedHmmAligLgProb cached_logap;
  viterbiAlgorithmCached(nSrcSentIndexVector, trgSentIndexVector, cached_logap, vitMatrix, predMatrix);
}

void HmmAlignmentModel::viterbiAlgorithmCached(const std::vector<WordIndex>& nSrcSentIndexVector,
                                               const std::vector<WordIndex>& trgSentIndexVector,
                                               CachedHmmAligLgProb& cached_logap,
                                               std::vector<std::vector<double>>& vitMatrix,
                                               std::vector<std::vector<PositionIndex>>& predMatrix)
{
  // Obtain slen
  PositionIndex slen = getSrcLen(nSrcSentIndexVector);

  // Clear matrices
  vitMatrix.clear();
  predMatrix.clear();

  // Make room for matrices
  std::vector<double> dVec;
  dVec.insert(dVec.begin(), trgSentIndexVector.size() + 1, SMALL_LG_NUM);
  vitMatrix.insert(vitMatrix.begin(), nSrcSentIndexVector.size() + 1, dVec);

  std::vector<PositionIndex> pidxVec;
  pidxVec.insert(pidxVec.begin(), trgSentIndexVector.size() + 1, 0);
  predMatrix.insert(predMatrix.begin(), nSrcSentIndexVector.size() + 1, pidxVec);

  // Fill matrices
  for (PositionIndex j = 1; j <= trgSentIndexVector.size(); ++j)
  {
    for (PositionIndex i = 1; i <= nSrcSentIndexVector.size(); ++i)
    {
      double logPts = translationLogProb(nSrcSentIndexVector[i - 1], trgSentIndexVector[j - 1]);
      if (j == 1)
      {
        // Update cached alignment log-probs if required
        if (!cached_logap.isDefined(0, slen, i))
          cached_logap.set_boundary_check(0, slen, i, hmmAlignmentLogProb(0, slen, i));

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
            cached_logap.set_boundary_check(i_tilde, slen, i, hmmAlignmentLogProb(i_tilde, slen, i));

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

double HmmAlignmentModel::bestAligGivenVitMatricesRaw(const std::vector<std::vector<double>>& vitMatrix,
                                                      const std::vector<std::vector<PositionIndex>>& predMatrix,
                                                      std::vector<PositionIndex>& bestAlig)
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

double HmmAlignmentModel::bestAligGivenVitMatrices(PositionIndex slen,
                                                   const std::vector<std::vector<double>>& vitMatrix,
                                                   const std::vector<std::vector<PositionIndex>>& predMatrix,
                                                   std::vector<PositionIndex>& bestAlig)
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

double HmmAlignmentModel::forwardAlgorithm(const std::vector<WordIndex>& nSrcSentIndexVector,
                                           const std::vector<WordIndex>& trgSentIndexVector, int verbose)
{
  // Obtain slen
  PositionIndex slen = getSrcLen(nSrcSentIndexVector);

  // Make room for matrix
  std::vector<std::vector<double>> forwardMatrix;
  std::vector<double> dVec;
  dVec.insert(dVec.begin(), trgSentIndexVector.size() + 1, 0.0);
  forwardMatrix.insert(forwardMatrix.begin(), nSrcSentIndexVector.size() + 1, dVec);

  // Fill matrix
  for (PositionIndex j = 1; j <= trgSentIndexVector.size(); ++j)
  {
    for (PositionIndex i = 1; i <= nSrcSentIndexVector.size(); ++i)
    {
      double logPts = translationLogProb(nSrcSentIndexVector[i - 1], trgSentIndexVector[j - 1]);
      if (j == 1)
      {
        forwardMatrix[i][j] = hmmAlignmentLogProb(0, slen, i) + logPts;
      }
      else
      {
        for (PositionIndex i_tilde = 1; i_tilde <= nSrcSentIndexVector.size(); ++i_tilde)
        {
          double lp = forwardMatrix[i_tilde][j - 1] + (double)hmmAlignmentLogProb(i_tilde, slen, i) + logPts;
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
        std::cerr << "i=" << i << ",j=" << j << " " << forwardMatrix[i][j];
        if (i < nSrcSentIndexVector.size())
          std::cerr << " ; ";
      }
      std::cerr << std::endl;
    }
  }

  // Return result
  return lp;
}

double HmmAlignmentModel::lgProbGivenForwardMatrix(const std::vector<std::vector<double>>& forwardMatrix)
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

PositionIndex HmmAlignmentModel::getSrcLen(const std::vector<WordIndex>& nsrcWordIndexVec)
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

Prob HmmAlignmentModel::calcProbOfAlignment(CachedHmmAligLgProb& cached_logap, const std::vector<WordIndex>& src,
                                            const std::vector<WordIndex>& trg, AlignmentInfo& alignment, int verbose)
{
  PositionIndex slen = alignment.getSourceLength();

  double logProb = 0;
  PositionIndex prev_i = 0;
  for (PositionIndex j = 1; j <= trg.size(); ++j)
  {
    PositionIndex i = alignment.get(j);
    WordIndex s = i == 0 ? NULL_WORD : src[i - 1];
    WordIndex t = trg[j - 1];
    if (i == 0)
    {
      if (prev_i == 0)
        i = slen + 1;
      else
        i = prev_i <= slen ? prev_i + slen : prev_i;
    }
    if (!cached_logap.isDefined(prev_i, slen, i))
      cached_logap.set_boundary_check(prev_i, slen, i, hmmAlignmentLogProb(prev_i, slen, i));
    logProb += cached_logap.get(prev_i, slen, i) + double{translationLogProb(s, t)};
    prev_i = i;
  }
  return exp(logProb);
}

double HmmAlignmentModel::swapScore(CachedHmmAligLgProb& cached_logap, const std::vector<WordIndex>& src,
                                    const std::vector<WordIndex>& trg, PositionIndex j1, PositionIndex j2,
                                    AlignmentInfo& alignment, double alignmentProb)
{
  PositionIndex i1 = alignment.get(j1);
  PositionIndex i2 = alignment.get(j2);
  if (i1 == i2)
    return 1.0;

  alignment.set(j1, i2);
  alignment.set(j2, i1);
  double newProb = calcProbOfAlignment(cached_logap, src, trg, alignment);
  alignment.set(j1, i1);
  alignment.set(j2, i2);

  if (alignmentProb > 0.0)
    return newProb / alignmentProb;
  else if (newProb > 0.0)
    return 1e20;
  else
    return 1.0;
}

double HmmAlignmentModel::moveScore(CachedHmmAligLgProb& cached_logap, const std::vector<WordIndex>& src,
                                    const std::vector<WordIndex>& trg, PositionIndex iNew, PositionIndex j,
                                    AlignmentInfo& alignment, double alignmentProb)
{
  PositionIndex iOld = alignment.get(j);

  alignment.set(j, iNew);
  double newProb = calcProbOfAlignment(cached_logap, src, trg, alignment);
  alignment.set(j, iOld);

  if (alignmentProb > 0.0)
    return newProb / alignmentProb;
  else if (newProb > 0.0)
    return 1e20;
  else
    return 1.0;
}

std::vector<WordIndex> HmmAlignmentModel::extendWithNullWord(const std::vector<WordIndex>& srcWordIndexVec)
{
  // Initialize result using srcWordIndexVec
  std::vector<WordIndex> result = srcWordIndexVec;

  // Add NULL words
  WordIndex nullWidx = stringToSrcWordIndex(NULL_WORD_STR);
  for (unsigned int i = 0; i < srcWordIndexVec.size(); ++i)
    result.push_back(nullWidx);

  return result;
}

void HmmAlignmentModel::calcAlphaBetaMatrices(const std::vector<WordIndex>& nsrcSent,
                                              const std::vector<WordIndex>& trgSent, PositionIndex slen,
                                              std::vector<std::vector<double>>& lexProbs,
                                              std::vector<std::vector<double>>& alignProbs,
                                              std::vector<std::vector<double>>& alphaMatrix,
                                              std::vector<std::vector<double>>& betaMatrix)
{
  // Create data structure to cache lexical probs
  lexProbs.clear();
  std::vector<double> innerLexProbs(trgSent.size() + 1, 0.0);
  lexProbs.resize(nsrcSent.size() + 1, innerLexProbs);

  // Create data structure to cache alignment probs
  alignProbs.clear();
  std::vector<double> innerAlignProbs(nsrcSent.size() + 1, 0.0);
  alignProbs.resize(nsrcSent.size() + 1, innerAlignProbs);

  // Initialize alphaMatrix
  alphaMatrix.clear();
  std::vector<double> innerMatrix(trgSent.size() + 1, 0.0);
  alphaMatrix.resize(nsrcSent.size() + 1, innerMatrix);

  for (PositionIndex j = 1; j <= trgSent.size(); ++j)
  {
    for (PositionIndex i = 1; i <= nsrcSent.size(); ++i)
      lexProbs[i][j] = translationProb(nsrcSent[i - 1], trgSent[j - 1]);
  }

  for (PositionIndex i = 1; i <= nsrcSent.size(); ++i)
  {
    for (PositionIndex i_tilde = 0; i_tilde <= nsrcSent.size(); ++i_tilde)
      alignProbs[i][i_tilde] = hmmAlignmentProb(i_tilde, slen, i);
  }

  std::vector<double> sums(trgSent.size() + 1, 0.0);
  // Fill alphaMatrix
  for (PositionIndex j = 1; j <= trgSent.size(); ++j)
  {
    for (PositionIndex i = 1; i <= nsrcSent.size(); ++i)
    {
      if (j == 1)
      {
        alphaMatrix[i][j] = alignProbs[i][0] * lexProbs[i][j];
      }
      else
      {
        for (PositionIndex i_tilde = 1; i_tilde <= nsrcSent.size(); ++i_tilde)
          alphaMatrix[i][j] += alphaMatrix[i_tilde][j - 1] * alignProbs[i][i_tilde] * lexProbs[i][j];
      }
      sums[j] += alphaMatrix[i][j];
    }

    if (sums[j] > 0)
    {
      for (PositionIndex i = 1; i <= nsrcSent.size(); ++i)
        alphaMatrix[i][j] /= sums[j];
    }
  }

  // Initialize betaMatrix
  betaMatrix.clear();
  betaMatrix.resize(nsrcSent.size() + 1, innerMatrix);

  // Fill betaMatrix
  for (PositionIndex j = trgSent.size(); j >= 1; --j)
  {
    if (sums[j] > 0)
    {
      for (PositionIndex i = 1; i <= nsrcSent.size(); ++i)
      {
        if (j == trgSent.size())
        {
          betaMatrix[i][j] = 1.0;
        }
        else
        {
          for (PositionIndex i_tilde = 1; i_tilde <= nsrcSent.size(); ++i_tilde)
          {
            betaMatrix[i][j] +=
                betaMatrix[i_tilde][size_t{j} + 1] * alignProbs[i_tilde][i] * lexProbs[i_tilde][size_t{j} + 1];
          }
        }

        betaMatrix[i][j] /= sums[j];
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

double HmmAlignmentModel::unsmoothedHmmAlignmentLogProb(PositionIndex prev_i, PositionIndex slen, PositionIndex i)
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
        return log((double)hmmP0) - log((double)slen);
      }
      else
        return log((double)hmmP0);
    }
    else
    {
      bool found;
      double numer =
          hmmAlignmentTable->getNumerator(hmmAligInfo.modified_ip, getCompactedSentenceLength(slen), i, found);
      if (found)
      {
        // aligNumer for pair asHmm,i exists
        double denom =
            hmmAlignmentTable->getDenominator(hmmAligInfo.modified_ip, getCompactedSentenceLength(slen), found);
        if (!found)
          return SMALL_LG_NUM;
        else
        {
          LgProb lp = numer - denom;
          return lp + log((double)1.0 - (double)hmmP0);
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
    std::cerr << "Loading file with lexical smoothing interpolation factor from " << lexSmIntFactorFile << std::endl;

  AwkInputStream awk;

  if (awk.open(lexSmIntFactorFile) == THOT_ERROR)
  {
    if (verbose)
      std::cerr << "Error in file with lexical smoothing interpolation factor, file " << lexSmIntFactorFile
                << " does not exist. Assuming default value." << std::endl;
    setLexicalSmoothFactor(0.1);
    return THOT_OK;
  }
  else
  {
    if (awk.getln())
    {
      if (awk.NF == 1)
      {
        setLexicalSmoothFactor((Prob)atof(awk.dollar(1).c_str()));
        return THOT_OK;
      }
      else
      {
        if (verbose)
          std::cerr << "Error: anomalous .lsifactor file, " << lexSmIntFactorFile << std::endl;
        return THOT_ERROR;
      }
    }
    else
    {
      if (verbose)
        std::cerr << "Error: anomalous .lsifactor file, " << lexSmIntFactorFile << std::endl;
      return THOT_ERROR;
    }
  }
}

bool HmmAlignmentModel::loadAlSmIntFactor(const char* alSmIntFactorFile, int verbose)
{
  if (verbose)
    std::cerr << "Loading file with alignment smoothing interpolation factor from " << alSmIntFactorFile << std::endl;

  AwkInputStream awk;

  if (awk.open(alSmIntFactorFile) == THOT_ERROR)
  {
    if (verbose)
      std::cerr << "Error in file with alignment smoothing interpolation factor, file " << alSmIntFactorFile
                << " does not exist. Assuming default value." << std::endl;
    setHmmAlignmentSmoothFactor(0.3);
    return THOT_OK;
  }
  else
  {
    if (awk.getln())
    {
      if (awk.NF == 1)
      {
        setHmmAlignmentSmoothFactor((Prob)atof(awk.dollar(1).c_str()));
        return THOT_OK;
      }
      else
      {
        if (verbose)
          std::cerr << "Error: anomalous .asifactor file, " << alSmIntFactorFile << std::endl;
        return THOT_ERROR;
      }
    }
    else
    {
      if (verbose)
        std::cerr << "Error: anomalous .asifactor file, " << alSmIntFactorFile << std::endl;
      return THOT_ERROR;
    }
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
                << " does not exist. Assuming hmm_p0=" << DefaultHmmP0 << "\n";
    setHmmP0(0.1);
    return THOT_OK;
  }
  else
  {
    if (awk.getln())
    {
      if (awk.NF == 1)
      {
        hmmP0 = (Prob)atof(awk.dollar(1).c_str());
        if (verbose)
          std::cerr << "hmm p0 value has been set to " << hmmP0 << std::endl;
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

void HmmAlignmentModel::loadConfig(const YAML::Node& config)
{
  Ibm2AlignmentModel::loadConfig(config);

  hmmP0 = config["hmmP0"].as<double>();
  hmmAlignmentSmoothFactor = config["hmmAlignmentSmoothFactor"].as<double>();
  lexicalSmoothFactor = config["lexicalSmoothFactor"].as<double>();
}

bool HmmAlignmentModel::loadOldConfig(const char* prefFileName, int verbose)
{
  Ibm2AlignmentModel::loadOldConfig(prefFileName, verbose);

  // Load file with with lexical smoothing interpolation factor
  std::string lsifFile = prefFileName;
  lsifFile = lsifFile + ".lsifactor";
  bool retVal = loadLexSmIntFactor(lsifFile.c_str(), verbose);
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  // Load file with with alignment smoothing interpolation factor
  std::string asifFile = prefFileName;
  asifFile = asifFile + ".asifactor";
  retVal = loadAlSmIntFactor(asifFile.c_str(), verbose);
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  // Load file with hmm p0 value
  std::string hmmP0File = prefFileName;
  hmmP0File = hmmP0File + ".hmm_p0";
  return loadHmmP0(hmmP0File.c_str(), verbose);
}

void HmmAlignmentModel::createConfig(YAML::Emitter& out)
{
  Ibm2AlignmentModel::createConfig(out);

  out << YAML::Key << "hmmP0" << YAML::Value << double{hmmP0};
  out << YAML::Key << "hmmAlignmentSmoothFactor" << YAML::Value << hmmAlignmentSmoothFactor;
  out << YAML::Key << "lexicalSmoothFactor" << YAML::Value << lexicalSmoothFactor;
}
