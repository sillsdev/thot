#include "sw_models/Ibm3AlignmentModel.h"

#include "nlp_common/MathFuncs.h"
#include "sw_models/SwDefs.h"

Ibm3AlignmentModel::Ibm3AlignmentModel()
    : p1{std::make_shared<Prob>(DefaultP1)}, distortionTable{std::make_shared<DistortionTable>()},
      fertilityTable{std::make_shared<FertilityTable>()}
{
  maxSentenceLength = MaxSentenceLength;
}

Ibm3AlignmentModel::Ibm3AlignmentModel(Ibm2AlignmentModel& model)
    : Ibm2AlignmentModel{model}, p1{std::make_shared<Prob>(DefaultP1)},
      distortionTable{std::make_shared<DistortionTable>()}, fertilityTable{std::make_shared<FertilityTable>()},
      performIbm2Transfer{true}
{
  maxSentenceLength = MaxSentenceLength;
}

Ibm3AlignmentModel::Ibm3AlignmentModel(HmmAlignmentModel& model)
    : Ibm2AlignmentModel{model}, p1{std::make_shared<Prob>(DefaultP1)},
      distortionTable{std::make_shared<DistortionTable>()},
      fertilityTable{std::make_shared<FertilityTable>()}, hmmModel{new HmmAlignmentModel{model}}
{
  maxSentenceLength = MaxSentenceLength;
}

Ibm3AlignmentModel::Ibm3AlignmentModel(Ibm3AlignmentModel& model)
    : Ibm2AlignmentModel{model}, countThreshold{model.countThreshold},
      fertilitySmoothFactor{model.fertilitySmoothFactor}, p1{model.p1}, distortionTable{model.distortionTable},
      fertilityTable{model.fertilityTable}
{
  maxSentenceLength = MaxSentenceLength;
}

double Ibm3AlignmentModel::getCountThreshold() const
{
  return countThreshold;
}

void Ibm3AlignmentModel::setCountThreshold(double threshold)
{
  countThreshold = threshold;
}

double Ibm3AlignmentModel::getFertilitySmoothFactor() const
{
  return fertilitySmoothFactor;
}

void Ibm3AlignmentModel::setFertilitySmoothFactor(double factor)
{
  fertilitySmoothFactor = factor;
}

unsigned int Ibm3AlignmentModel::startTraining(int verbosity)
{
  unsigned int count = Ibm2AlignmentModel::startTraining(verbosity);

  maxSrcWordLen = 0;
  for (WordIndex s = 3; s < getSrcVocabSize(); ++s)
    maxSrcWordLen = std::max(maxSrcWordLen, wordIndexToSrcString(s).length());

  if (performIbm2Transfer)
  {
    ibm2Transfer();
    performIbm2Transfer = false;
  }
  return count;
}

void Ibm3AlignmentModel::ibm2Transfer()
{
  std::vector<std::pair<std::vector<WordIndex>, std::vector<WordIndex>>> buffer;
  for (unsigned int n = 0; n < numSentencePairs(); ++n)
  {
    std::vector<WordIndex> src = getSrcSent(n);
    std::vector<WordIndex> trg = getTrgSent(n);
    if (sentenceLengthIsOk(src) && sentenceLengthIsOk(trg))
      buffer.push_back(make_pair(src, trg));

    if (buffer.size() >= ThreadBufferSize)
    {
      ibm2TransferUpdateCounts(buffer);
      buffer.clear();
    }
  }
  if (buffer.size() > 0)
  {
    ibm2TransferUpdateCounts(buffer);
    buffer.clear();
  }

  batchMaximizeProbs();
}

void Ibm3AlignmentModel::ibm2TransferUpdateCounts(
    const std::vector<std::pair<std::vector<WordIndex>, std::vector<WordIndex>>>& pairs)
{
#pragma omp parallel for schedule(dynamic)
  for (int line_idx = 0; line_idx < (int)pairs.size(); ++line_idx)
  {
    std::vector<WordIndex> src = pairs[line_idx].first;
    std::vector<WordIndex> nsrc = extendWithNullWord(src);
    std::vector<WordIndex> trg = pairs[line_idx].second;

    PositionIndex slen = PositionIndex(src.size());
    PositionIndex tlen = PositionIndex(trg.size());

    Matrix<double> probs{slen + 1, tlen + 1};
    for (PositionIndex j = 1; j <= tlen; ++j)
    {
      double sum = 0;
      for (PositionIndex i = 0; i <= slen; ++i)
      {
        probs(i, j) = getCountNumerator(nsrc, trg, i, j);
        sum += probs(i, j);
      }
      if (sum > 0)
      {
        for (PositionIndex i = 0; i <= slen; ++i)
        {
          probs(i, j) /= sum;
          if (probs(i, j) == 1.0)
            probs(i, j) = 0.99;
          else if (probs(i, j) == 0)
            probs(i, j) = SW_PROB_SMOOTH;
          double count = probs(i, j);
          if (count > SW_PROB_SMOOTH)
          {
            Ibm2AlignmentModel::incrementWordPairCounts(nsrc, trg, i, j, count);
            if (i > 0)
            {
              DistortionKey key{i, getCompactedSentenceLength(slen), tlen};

#pragma omp atomic
              distortionCounts[key][j - 1] += count;
            }
          }
        }
      }
    }

    PositionIndex maxFertility = std::min(tlen + 1, MaxFertility);
    Matrix<double> alpha{maxFertility, slen + 1};
    for (PositionIndex i = 1; i <= slen; ++i)
    {
      for (PositionIndex phi = 1; phi < maxFertility; ++phi)
      {
        double beta = 0;
        alpha(phi, i) = 0;
        for (PositionIndex j = 1; j <= tlen; ++j)
        {
          double prob = probs(i, j);
          if (prob > 0.95)
            prob = 0.95;
          else if (prob < 0.05)
            prob = 0.05;
          beta += pow(prob / (1.0 - prob), double(phi));
        }
        alpha(phi, i) = beta * pow(-1.0, double(phi) + 1.0) / double(phi);
      }
    }

    for (PositionIndex i = 1; i <= slen; ++i)
    {
      WordIndex s = nsrc[i];
      double r = 1;
      for (PositionIndex j = 1; j <= tlen; ++j)
        r *= 1 - probs(i, j);
      for (PositionIndex phi = 0; phi < maxFertility; ++phi)
      {
        double sum = getSumOfPartitions(phi, i, alpha);
        double count = r * sum;

#pragma omp atomic
        fertilityCounts[s][phi] += count;
      }
    }
  }
}

void Ibm3AlignmentModel::train(int verbosity)
{
  if (hmmModel)
  {
    hmmTransfer();
    hmmModel.reset(nullptr);
    cachedHmmAligLogProbs.clear();
  }
  else
  {
    Ibm2AlignmentModel::train(verbosity);
  }
}

void Ibm3AlignmentModel::hmmTransfer()
{
  auto search = [this](const std::vector<WordIndex>& src, const std::vector<WordIndex>& trg,
                       AlignmentInfo& bestAlignment, Matrix<double>& moveScores, Matrix<double>& swapScores) {
    Prob prob = hmmModel->searchForBestAlignment(src, trg, bestAlignment, cachedHmmAligLogProbs);
    if (!bestAlignment.isValid(MaxFertility))
    {
      std::vector<WordIndex> nsrc = extendWithNullWord(src);
      getInitialAlignmentForSearch(nsrc, trg, bestAlignment);
      prob = hmmModel->calcProbOfAlignment(cachedHmmAligLogProbs, src, trg, bestAlignment);
    }
    hmmModel->populateMoveSwapScores(src, trg, bestAlignment, prob, cachedHmmAligLogProbs, moveScores, swapScores);
    return prob;
  };

  std::vector<std::pair<std::vector<WordIndex>, std::vector<WordIndex>>> buffer;
  for (unsigned int n = 0; n < numSentencePairs(); ++n)
  {
    std::vector<WordIndex> src = getSrcSent(n);
    std::vector<WordIndex> trg = getTrgSent(n);
    if (sentenceLengthIsOk(src) && sentenceLengthIsOk(trg))
      buffer.push_back(make_pair(src, trg));

    if (buffer.size() >= ThreadBufferSize)
    {
      batchUpdateCounts(buffer, search);
      buffer.clear();
    }
  }
  if (buffer.size() > 0)
  {
    batchUpdateCounts(buffer, search);
    buffer.clear();
  }

  batchMaximizeProbs();
}

double Ibm3AlignmentModel::getSumOfPartitions(PositionIndex phi, PositionIndex srcPos, const Matrix<double>& alpha)
{
  std::vector<PositionIndex> partitions(MaxFertility, 0);
  std::vector<PositionIndex> mult(MaxFertility, 0);
  PositionIndex numPartitions = 0;
  double sum = 0;

  bool done = false;
  bool init = true;
  while (!done)
  {
    if (init)
    {
      partitions[1] = phi;
      mult[1] = 1;
      numPartitions = 1;
      init = false;
    }
    else
    {
      if ((partitions[numPartitions] > 1) || (numPartitions > 1))
      {
        int s;
        PositionIndex k;
        if (partitions[numPartitions] == 1)
        {
          s = partitions[numPartitions - 1] + mult[numPartitions];
          k = numPartitions - 1;
        }
        else
        {
          s = partitions[numPartitions];
          k = numPartitions;
        }
        int w = partitions[k] - 1;
        int u = s / w;
        int v = s % w;
        mult[k] -= 1;
        PositionIndex k1;
        if (mult[k] == 0)
          k1 = k;
        else
          k1 = k + 1;
        mult[k1] = u;
        partitions[k1] = w;
        if (v == 0)
        {
          numPartitions = k1;
        }
        else
        {
          mult[(size_t)k1 + 1] = 1;
          partitions[(size_t)k1 + 1] = v;
          numPartitions = k1 + 1;
        }
      }
      else
      {
        done = true;
      }
    }

    if (!done)
    {
      double prod = 1.0;
      if (phi != 0)
      {
        for (PositionIndex i = 1; i <= numPartitions; ++i)
          prod *= pow(alpha(partitions[i], srcPos), mult[i]) / MathFuncs::factorial(mult[i]);
      }
      sum += prod;
    }
  }
  return sum < 0 ? 0 : sum;
}

void Ibm3AlignmentModel::initSentencePair(const std::vector<WordIndex>& src, const std::vector<WordIndex>& trg)
{
  if (hmmModel)
  {
    // Make room for data structure to cache alignment log-probs
    cachedHmmAligLogProbs.makeRoomGivenSrcSentLen(src.size());
  }
}

void Ibm3AlignmentModel::initSourceWord(const std::vector<WordIndex>& nsrc, const std::vector<WordIndex>& trg,
                                        PositionIndex i)
{
  Ibm2AlignmentModel::initSourceWord(nsrc, trg, i);

  PositionIndex slen = (PositionIndex)nsrc.size() - 1;
  PositionIndex tlen = (PositionIndex)trg.size();

  distortionTable->reserveSpace(i, getCompactedSentenceLength(slen), tlen);

  DistortionKey key{i, getCompactedSentenceLength(slen), tlen};
  DistortionCountsElem& distortionEntry = distortionCounts[key];
  if (distortionEntry.size() < trg.size())
    distortionEntry.resize(trg.size(), 0);
}

void Ibm3AlignmentModel::addTranslationOptions(std::vector<std::vector<WordIndex>>& insertBuffer)
{
  WordIndex maxSrcWordIndex = (WordIndex)insertBuffer.size() - 1;

  if (maxSrcWordIndex >= lexCounts.size())
    lexCounts.resize((size_t)maxSrcWordIndex + 1);
  lexTable->reserveSpace(maxSrcWordIndex);

  if (maxSrcWordIndex >= fertilityCounts.size())
    fertilityCounts.resize((size_t)maxSrcWordIndex + 1);
  fertilityTable->reserveSpace(maxSrcWordIndex);

#pragma omp parallel for schedule(dynamic)
  for (int s = 0; s < (int)insertBuffer.size(); ++s)
  {
    for (WordIndex t : insertBuffer[s])
      lexCounts[s][t] = 0;

    FertilityCountsElem& fertilityEntry = fertilityCounts[s];
    fertilityEntry.resize(MaxFertility, 0);

    insertBuffer[s].clear();
  }
}

void Ibm3AlignmentModel::batchUpdateCounts(
    const std::vector<std::pair<std::vector<WordIndex>, std::vector<WordIndex>>>& pairs)
{
  auto search = [this](const std::vector<WordIndex>& src, const std::vector<WordIndex>& trg,
                       AlignmentInfo& bestAlignment, Matrix<double>& moveScores, Matrix<double>& swapScores) {
    return searchForBestAlignment(src, trg, bestAlignment, &moveScores, &swapScores);
  };
  batchUpdateCounts(pairs, search);
}

void Ibm3AlignmentModel::batchUpdateCounts(
    const std::vector<std::pair<std::vector<WordIndex>, std::vector<WordIndex>>>& pairs,
    SearchForBestAlignmentFunc search)
{
#pragma omp parallel for schedule(dynamic)
  for (int line_idx = 0; line_idx < (int)pairs.size(); ++line_idx)
  {
    std::vector<WordIndex> src = pairs[line_idx].first;
    std::vector<WordIndex> nsrc = extendWithNullWord(src);
    std::vector<WordIndex> trg = pairs[line_idx].second;

    AlignmentInfo alignment(nsrc.size() - 1, trg.size());
    Matrix<double> moveScores, swapScores;
    double aligProb = search(src, trg, alignment, moveScores, swapScores);
    if (aligProb <= 0)
      continue;

    updateCounts(nsrc, trg, alignment, aligProb, moveScores, swapScores);
  }
}

void Ibm3AlignmentModel::incrementWordPairCounts(const std::vector<WordIndex>& nsrc, const std::vector<WordIndex>& trg,
                                                 PositionIndex i, PositionIndex j, double count)
{
  Ibm2AlignmentModel::incrementWordPairCounts(nsrc, trg, i, j, count);

  DistortionKey key{i, getCompactedSentenceLength(nsrc.size() - 1), (PositionIndex)trg.size()};

#pragma omp atomic
  distortionCounts[key][j - 1] += count;
}

double Ibm3AlignmentModel::updateCounts(const std::vector<WordIndex>& nsrc, const std::vector<WordIndex>& trg,
                                        AlignmentInfo& alignment, double aligProb, const Matrix<double>& moveScores,
                                        const Matrix<double>& swapScores)
{
  PositionIndex slen = (PositionIndex)nsrc.size() - 1;
  PositionIndex tlen = (PositionIndex)trg.size();

  Matrix<double> moveCounts(slen + 1, tlen + 1), swapCounts(slen + 1, tlen + 1);
  std::vector<double> negMove((size_t)tlen + 1), negSwap((size_t)tlen + 1), plus1Fert((size_t)slen + 1),
      minus1Fert((size_t)slen + 1);
  double totalMove = aligProb;
  double totalSwap = 0;

  for (PositionIndex j = 1; j <= tlen; ++j)
  {
    for (PositionIndex i = 0; i <= slen; ++i)
    {
      if (alignment.get(j) != i)
      {
        double prob = aligProb * moveScores(i, j);
        totalMove += prob;
        moveCounts(i, j) += prob;
        negMove[j] += prob;
        plus1Fert[i] += prob;
        minus1Fert[alignment.get(j)] += prob;
      }
    }

    for (PositionIndex j1 = j + 1; j1 <= tlen; ++j1)
    {
      if (alignment.get(j) != alignment.get(j1))
      {
        double prob = aligProb * swapScores(j, j1);
        totalSwap += prob;
        swapCounts(alignment.get(j), j1) += prob;
        swapCounts(alignment.get(j1), j) += prob;
        negSwap[j] += prob;
        negSwap[j1] += prob;
      }
    }
  }

  double totalCount = totalMove + totalSwap;
  Matrix<double> fertCounts(slen + 1, MaxFertility + 1);
  for (PositionIndex i = 0; i <= slen; ++i)
  {
    for (PositionIndex j = 1; j <= tlen; ++j)
    {
      double count =
          i == alignment.get(j) ? totalCount - (negMove[j] + negSwap[j]) : moveCounts(i, j) + swapCounts(i, j);
      count /= totalCount;
      if (count > countThreshold)
        incrementWordPairCounts(nsrc, trg, i, j, count);
    }

    if (i > 0)
    {
      double temp = minus1Fert[i] + plus1Fert[i];
      PositionIndex phi = alignment.getFertility(i);
      if (phi < MaxFertility)
        fertCounts(i, phi) += totalCount - temp;
      if (phi > 0 && phi - 1 < MaxFertility)
        fertCounts(i, phi - 1) += minus1Fert[i];
      if (phi + 1 < MaxFertility)
        fertCounts(i, phi + 1) += plus1Fert[i];
    }
  }

  for (PositionIndex i = 1; i <= slen; ++i)
  {
    WordIndex s = nsrc[i];
    for (PositionIndex phi = 0; phi < MaxFertility; ++phi)
    {
      double count = fertCounts(i, phi) / totalCount;

#pragma omp atomic
      fertilityCounts[s][phi] += count;
    }
  }

  PositionIndex phi0 = alignment.getFertility(0);
  double temp = minus1Fert[0] + plus1Fert[0];
  double p1c = (totalCount - temp) * phi0;
  double p0c = (totalCount - temp) * (tlen - 2 * phi0);
  if (phi0 > 0)
  {
    p1c += minus1Fert[0] * (phi0 - 1);
    p0c += minus1Fert[0] * (tlen - 2 * (phi0 - 1));
  }
  if (tlen - 2 * (phi0 + 1) >= 0)
  {
    p1c += plus1Fert[0] * (phi0 + 1.0);
    p0c += plus1Fert[0] * (tlen - 2 * (phi0 + 1));
  }

#pragma omp atomic
  p1Count += p1c / totalCount;
#pragma omp atomic
  p0Count += p0c / totalCount;

  return totalCount;
}

void Ibm3AlignmentModel::batchMaximizeProbs()
{
  Ibm2AlignmentModel::batchMaximizeProbs();

#pragma omp parallel for schedule(dynamic)
  for (int asIndex = 0; asIndex < (int)distortionCounts.size(); ++asIndex)
  {
    double denom = 0;
    const std::pair<DistortionKey, DistortionCountsElem>& p = distortionCounts.getAt(asIndex);
    const DistortionKey& key = p.first;
    DistortionCountsElem& elem = const_cast<DistortionCountsElem&>(p.second);
    for (PositionIndex j = 1; j <= key.tlen; ++j)
    {
      double numer = elem[j - 1];
      denom += numer;
      float logNumer = (float)log(numer);
      distortionTable->setNumerator(key.i, key.slen, key.tlen, j, logNumer);
      elem[j - 1] = 0.0;
    }
    if (denom == 0)
      denom = 1;
    float logDenom = (float)log(denom);
    distortionTable->setDenominator(key.i, key.slen, key.tlen, logDenom);
  }

  Matrix<double> counts(maxSrcWordLen + 1, MaxFertility + 1, 0.0);
#pragma omp parallel for schedule(dynamic)
  for (int s = 3; s < (int)fertilityCounts.size(); ++s)
  {
    size_t len = wordIndexToSrcString(s).length();
    FertilityCountsElem& elem = fertilityCounts[s];
    for (PositionIndex phi = 0; phi < MaxFertility; ++phi)
    {
#pragma omp atomic
      counts(len, phi) += std::max(elem[phi], SW_PROB_SMOOTH);
    }
  }

  for (size_t i = 1; i < maxSrcWordLen + 1; ++i)
  {
    double sum = 0.0;
    for (PositionIndex phi = 0; phi < MaxFertility; ++phi)
      sum += counts(i, phi);
    if (sum > 0)
    {
      for (PositionIndex phi = 0; phi < MaxFertility; ++phi)
        counts(i, phi) /= sum;
    }
  }

#pragma omp parallel for schedule(dynamic)
  for (int s = 3; s < (int)fertilityCounts.size(); ++s)
  {
    size_t len = wordIndexToSrcString(s).length();
    double denom = 0;
    FertilityCountsElem& elem = fertilityCounts[s];
    for (PositionIndex phi = 0; phi < MaxFertility; ++phi)
    {
      double numer = std::max(elem[phi], SW_PROB_SMOOTH) + (counts(len, phi) * fertilitySmoothFactor);
      denom += numer;
      fertilityTable->setNumerator(s, phi, (float)log(numer));
      elem[phi] = 0.0;
    }
    if (denom == 0)
      denom = 1;
    fertilityTable->setDenominator(s, (float)log(denom));
  }

  if (p1Count + p0Count > 0)
    *p1 = p1Count / (p1Count + p0Count);
  else
    *p1 = DefaultP1;

  p1Count = 0;
  p0Count = 0;
}

Prob Ibm3AlignmentModel::distortionProb(PositionIndex i, PositionIndex slen, PositionIndex tlen, PositionIndex j)
{
  double logProb = unsmoothedDistortionLogProb(i, slen, tlen, j);
  double prob = logProb == SMALL_LG_NUM ? 1.0 / tlen : exp(logProb);
  return std::max(prob, SW_PROB_SMOOTH);
}

LgProb Ibm3AlignmentModel::distortionLogProb(PositionIndex i, PositionIndex slen, PositionIndex tlen, PositionIndex j)
{
  double logProb = unsmoothedDistortionLogProb(i, slen, tlen, j);
  if (logProb == SMALL_LG_NUM)
    logProb = log(1.0 / tlen);
  return std::max(logProb, SW_LOG_PROB_SMOOTH);
}

double Ibm3AlignmentModel::unsmoothedDistortionLogProb(PositionIndex i, PositionIndex slen, PositionIndex tlen,
                                                       PositionIndex j)
{
  bool found;
  double numer = distortionTable->getNumerator(i, getCompactedSentenceLength(slen), tlen, j, found);
  if (found)
  {
    // numerator for pair ds,j exists
    double denom = distortionTable->getDenominator(i, getCompactedSentenceLength(slen), tlen, found);
    if (found)
      return numer - denom;
  }
  return SMALL_LG_NUM;
}

Prob Ibm3AlignmentModel::fertilityProb(WordIndex s, PositionIndex phi)
{
  double logProb = unsmoothedFertilityLogProb(s, phi);
  double prob = 0;
  if (logProb == SMALL_LG_NUM)
  {
    if (phi == 0)
      prob = 0.2;
    else if (phi == 1)
      prob = 0.65;
    else if (phi == 2)
      prob = 0.1;
    else if (phi == 3)
      prob = 0.04;
    else if (phi >= 4 && phi < MaxFertility)
      prob = 0.01 / (MaxFertility - 4);
  }
  else
  {
    prob = exp(logProb);
  }
  return std::max(prob, SW_PROB_SMOOTH);
}

LgProb Ibm3AlignmentModel::fertilityLogProb(WordIndex s, PositionIndex phi)
{
  double logProb = unsmoothedFertilityLogProb(s, phi);
  if (logProb == SMALL_LG_NUM)
  {
    if (phi == 0)
      logProb = log(0.2);
    else if (phi == 1)
      logProb = log(0.65);
    else if (phi == 2)
      logProb = log(0.1);
    else if (phi == 3)
      logProb = log(0.04);
    else if (phi >= 4 && phi < MaxFertility)
      logProb = log(0.01 / (MaxFertility - 4));
  }
  return std::max(logProb, SW_LOG_PROB_SMOOTH);
}

double Ibm3AlignmentModel::unsmoothedFertilityLogProb(WordIndex s, PositionIndex phi)
{
  if (phi >= MaxFertility)
    return SMALL_LG_NUM;
  bool found;
  double numer = fertilityTable->getNumerator(s, phi, found);
  if (found)
  {
    // numerator for pair s,phi exists
    double denom = fertilityTable->getDenominator(s, found);
    if (found)
      return numer - denom;
  }
  return SMALL_LG_NUM;
}

LgProb Ibm3AlignmentModel::getBestAlignment(const std::vector<WordIndex>& srcSentence,
                                            const std::vector<WordIndex>& trgSentence,
                                            std::vector<PositionIndex>& bestAlignment)
{
  if (sentenceLengthIsOk(srcSentence) && sentenceLengthIsOk(trgSentence))
  {
    PositionIndex slen = (PositionIndex)srcSentence.size();
    PositionIndex tlen = (PositionIndex)trgSentence.size();

    AlignmentInfo bestAlignmentInfo(slen, tlen);
    LgProb lgProb = sentenceLengthLogProb(slen, tlen);
    lgProb += searchForBestAlignment(srcSentence, trgSentence, bestAlignmentInfo).get_lp();

    bestAlignment = bestAlignmentInfo.getAlignment();

    return lgProb;
  }
  else
  {
    bestAlignment.resize(trgSentence.size(), 0);
    return SMALL_LG_NUM;
  }
}

LgProb Ibm3AlignmentModel::computeLogProb(const std::vector<WordIndex>& srcSentence,
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
    return sentenceLengthLogProb(slen, tlen)
         + calcProbOfAlignment(addNullWordToWidxVec(srcSentence), trgSentence, alignment, verbose).get_lp();
  }
}

Prob Ibm3AlignmentModel::calcProbOfAlignment(const std::vector<WordIndex>& nsrc, const std::vector<WordIndex>& trg,
                                             AlignmentInfo& alignment, int verbose)
{
  PositionIndex slen = PositionIndex(nsrc.size() - 1);
  PositionIndex tlen = PositionIndex(trg.size());

  if (verbose)
    std::cerr << "Obtaining IBM Model 3 prob..." << std::endl;

  Prob p0 = Prob(1.0) - *p1;

  PositionIndex phi0 = alignment.getFertility(0);
  Prob prob = pow(p0, double(tlen - 2 * phi0)) * pow(*p1, double(phi0));

  for (PositionIndex phi = 1; phi <= phi0; ++phi)
    prob *= double(tlen - phi0 - phi + 1.0) / phi;

  for (PositionIndex i = 1; i <= slen; ++i)
  {
    WordIndex s = nsrc[i];
    PositionIndex phi = alignment.getFertility(i);
    prob *= Prob(MathFuncs::factorial(phi)) * fertilityProb(s, phi);
  }

  for (PositionIndex j = 1; j <= tlen; ++j)
  {
    PositionIndex i = alignment.get(j);
    WordIndex s = nsrc[i];
    WordIndex t = trg[j - 1];

    prob *= translationProb(s, t);
    if (i > 0)
      prob *= distortionProb(i, slen, tlen, j);
  }
  return prob;
}

LgProb Ibm3AlignmentModel::computeSumLogProb(const std::vector<WordIndex>& srcSentence,
                                             const std::vector<WordIndex>& trgSentence, int verbose)
{
  PositionIndex slen = (PositionIndex)srcSentence.size();
  PositionIndex tlen = (PositionIndex)trgSentence.size();
  std::vector<PositionIndex> nsrc = addNullWordToWidxVec(srcSentence);

  if (verbose)
    std::cerr << "Obtaining Sum IBM Model 3 logprob..." << std::endl;

  Prob p0 = 1.0 - (double)*p1;

  LgProb lgProb = sentenceLengthLogProb(slen, tlen);
  LgProb fertilityContrib = 0;
  for (PositionIndex fertility = 0; fertility < std::min(tlen, MaxFertility); ++fertility)
  {
    Prob sump = 0;
    Prob prob = 1.0;
    PositionIndex phi0 = fertility;
    prob *= pow(*p1, double(phi0)) * pow(p0, double(tlen - 2 * phi0));

    for (PositionIndex phi = 1; phi <= phi0; phi++)
      prob *= double(tlen - phi0 - phi + 1.0) / phi;
    sump += prob;

    for (PositionIndex i = 1; i <= slen; ++i)
    {
      PositionIndex phi = fertility;
      sump += Prob(MathFuncs::factorial(phi)) * fertilityProb(nsrc[i], phi);
    }
    fertilityContrib += sump.get_lp();
  }

  if (verbose)
    std::cerr << "- Fertility contribution= " << fertilityContrib << std::endl;
  lgProb += fertilityContrib;

  LgProb lexDistorionContrib = 0;
  for (PositionIndex j = 1; j <= tlen; ++j)
  {
    Prob sump = 0;
    for (PositionIndex i = 0; i <= slen; ++i)
    {
      WordIndex s = nsrc[i];
      WordIndex t = trgSentence[j - 1];

      sump += translationProb(s, t) * distortionProb(i, slen, tlen, j);
    }
    lexDistorionContrib += sump.get_lp();
  }

  if (verbose)
    std::cerr << "- Lexical plus distortion contribution= " << lexDistorionContrib << std::endl;
  lgProb += lexDistorionContrib;

  return lgProb;
}

bool Ibm3AlignmentModel::load(const char* prefFileName, int verbose)
{
  // Load IBM 2 Model data
  bool retVal = Ibm2AlignmentModel::load(prefFileName, verbose);
  if (retVal == THOT_ERROR)
    return retVal;

  if (verbose)
    std::cerr << "Loading IBM 3 Model data..." << std::endl;

  std::string p1File = prefFileName;
  p1File = p1File + ".p1";
  retVal = loadP1(p1File);
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  // Load file with distortion nd values
  std::string distortionNumDenFile = prefFileName;
  distortionNumDenFile = distortionNumDenFile + ".distnd";
  retVal = distortionTable->load(distortionNumDenFile.c_str(), verbose);
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  // Load file with fertility nd values
  std::string fertilityNumDenFile = prefFileName;
  fertilityNumDenFile = fertilityNumDenFile + ".fertnd";
  return fertilityTable->load(fertilityNumDenFile.c_str(), verbose);
}

bool Ibm3AlignmentModel::loadP1(const std::string& filename)
{
  std::ifstream in(filename);
  if (!in)
    return THOT_ERROR;
  in >> *p1;

  return THOT_OK;
}

bool Ibm3AlignmentModel::print(const char* prefFileName, int verbose)
{
  // Print IBM 2 Model data
  bool retVal = Ibm2AlignmentModel::print(prefFileName);
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  std::string p1File = prefFileName;
  p1File = p1File + ".p1";
  retVal = printP1(p1File);
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  // Print file with distortion nd values
  std::string distortionNumDenFile = prefFileName;
  distortionNumDenFile = distortionNumDenFile + ".distnd";
  retVal = distortionTable->print(distortionNumDenFile.c_str());
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  // Load file with fertility nd values
  std::string fertilityNumDenFile = prefFileName;
  fertilityNumDenFile = fertilityNumDenFile + ".fertnd";
  return fertilityTable->print(fertilityNumDenFile.c_str());
}

bool Ibm3AlignmentModel::printP1(const std::string& filename)
{
  std::ofstream out(filename);
  if (!out)
    return THOT_ERROR;
  out << std::setprecision(std::numeric_limits<double>::max_digits10) << *p1;
  return THOT_OK;
}

void Ibm3AlignmentModel::loadConfig(const YAML::Node& config)
{
  Ibm2AlignmentModel::loadConfig(config);

  countThreshold = config["countThreshold"].as<double>();
  fertilitySmoothFactor = config["fertilitySmoothFactor"].as<double>();
}

void Ibm3AlignmentModel::createConfig(YAML::Emitter& out)
{
  Ibm2AlignmentModel::createConfig(out);

  out << YAML::Key << "countThreshold" << YAML::Value << countThreshold;
  out << YAML::Key << "fertilitySmoothFactor" << YAML::Value << fertilitySmoothFactor;
}

Prob Ibm3AlignmentModel::searchForBestAlignment(const std::vector<WordIndex>& src, const std::vector<WordIndex>& trg,
                                                AlignmentInfo& bestAlignment, Matrix<double>* moveScores,
                                                Matrix<double>* swapScores)
{
  PositionIndex slen = (PositionIndex)src.size();
  PositionIndex tlen = (PositionIndex)trg.size();

  std::vector<WordIndex> nsrc = extendWithNullWord(src);

  // start with IBM-2 alignment
  getInitialAlignmentForSearch(nsrc, trg, bestAlignment);

  if (moveScores != nullptr)
    moveScores->resize(slen + 1, tlen + 1);
  if (swapScores != nullptr)
    swapScores->resize(tlen + 1, tlen + 1);

  // hillclimbing search
  int bestChangeType = -1;
  int changes = 0;
  while (bestChangeType != 0)
  {
    double cachedAlignmentValue = -1;
    bestChangeType = 0;
    PositionIndex bestChangeArg1 = 0;
    PositionIndex bestChangeArg2 = 0;
    double bestChangeScore = 1.00001;
    for (PositionIndex j = 1; j <= tlen; j++)
    {
      PositionIndex iAlig = bestAlignment.get(j);

      // swap alignments
      for (PositionIndex j1 = j + 1; j1 <= tlen; j1++)
      {
        if (iAlig != bestAlignment.get(j1))
        {
          double changeScore = swapScore(nsrc, trg, j, j1, bestAlignment, cachedAlignmentValue);
          if (swapScores != nullptr)
            swapScores->set(j, j1, changeScore);
          if (changeScore > bestChangeScore)
          {
            bestChangeScore = changeScore;
            bestChangeType = 1;
            bestChangeArg1 = j;
            bestChangeArg2 = j1;
          }
        }
        else if (swapScores != nullptr)
        {
          swapScores->set(j, j1, 1.0);
        }
      }

      // move alignment by one position
      for (PositionIndex i = 0; i <= slen; i++)
      {
        if (i != iAlig)
        {
          double changeScore = moveScore(nsrc, trg, i, j, bestAlignment, cachedAlignmentValue);
          if (moveScores != nullptr)
            moveScores->set(i, j, changeScore);
          if ((i != 0 || (tlen >= 2 * (bestAlignment.getFertility(0) + 1)))
              && bestAlignment.getFertility(i) + 1 < MaxFertility && changeScore > bestChangeScore)
          {
            bestChangeScore = changeScore;
            bestChangeType = 2;
            bestChangeArg1 = j;
            bestChangeArg2 = i;
          }
        }
        else if (moveScores != nullptr)
        {
          moveScores->set(i, j, 1.0);
        }
      }
    }
    if (bestChangeType == 1)
    {
      // swap
      PositionIndex j = bestChangeArg1;
      PositionIndex j1 = bestChangeArg2;
      PositionIndex i = bestAlignment.get(j);
      PositionIndex i1 = bestAlignment.get(j1);
      bestAlignment.set(j, i1);
      bestAlignment.set(j1, i);
    }
    else if (bestChangeType == 2)
    {
      // move
      PositionIndex j = bestChangeArg1;
      PositionIndex i = bestChangeArg2;
      bestAlignment.set(j, i);
    }
    ++changes;
    if (changes > 60)
      break;
  }
  return calcProbOfAlignment(nsrc, trg, bestAlignment);
}

void Ibm3AlignmentModel::getInitialAlignmentForSearch(const std::vector<WordIndex>& nsrc,
                                                      const std::vector<WordIndex>& trg, AlignmentInfo& alignment)
{
  PositionIndex slen = (PositionIndex)nsrc.size() - 1;
  PositionIndex tlen = (PositionIndex)trg.size();

  std::vector<PositionIndex> fertility((size_t)slen + 1, 0);

  for (PositionIndex j = 1; j <= tlen; ++j)
  {
    PositionIndex iBest = 0;
    double bestProb = 0;
    for (PositionIndex i = 0; i <= slen; ++i)
    {
      if (fertility[i] + 1 < MaxFertility && (i != 0 || tlen >= (2 * (fertility[0] + 1))))
      {
        WordIndex s = nsrc[i];
        WordIndex t = trg[j - 1];
        double prob = translationProb(s, t) * alignmentProb(j, slen, tlen, i);
        if (prob > bestProb)
        {
          iBest = i;
          bestProb = prob;
        }
      }
    }

    if (bestProb > 0)
    {
      alignment.set(j, iBest);
      fertility[iBest]++;
    }
  }
}

double Ibm3AlignmentModel::swapScore(const std::vector<WordIndex>& nsrc, const std::vector<WordIndex>& trg,
                                     PositionIndex j1, PositionIndex j2, AlignmentInfo& alignment,
                                     double& cachedAlignmentValue)
{
  PositionIndex i1 = alignment.get(j1);
  PositionIndex i2 = alignment.get(j2);
  if (i1 == i2)
    return 1.0;

  PositionIndex slen = (PositionIndex)nsrc.size() - 1;
  PositionIndex tlen = (PositionIndex)trg.size();
  WordIndex s1 = nsrc[i1];
  WordIndex s2 = nsrc[i2];
  WordIndex t1 = trg[j1 - 1];
  WordIndex t2 = trg[j2 - 1];
  Prob change =
      (translationProb(s2, t1) / translationProb(s1, t1)) * (translationProb(s1, t2) / translationProb(s2, t2));
  if (i1 > 0)
    change *= distortionProb(i1, slen, tlen, j2) / distortionProb(i1, slen, tlen, j1);
  if (i2 > 0)
    change *= distortionProb(i2, slen, tlen, j1) / distortionProb(i2, slen, tlen, j2);
  return change;
}

double Ibm3AlignmentModel::moveScore(const std::vector<WordIndex>& nsrc, const std::vector<WordIndex>& trg,
                                     PositionIndex iNew, PositionIndex j, AlignmentInfo& alignment,
                                     double& cachedAlignmentValue)
{
  PositionIndex iOld = alignment.get(j);
  if (iOld == iNew)
    return 1.0;

  PositionIndex slen = (PositionIndex)nsrc.size() - 1;
  PositionIndex tlen = (PositionIndex)trg.size();
  WordIndex sOld = nsrc[iOld];
  WordIndex sNew = nsrc[iNew];
  WordIndex t = trg[j - 1];
  PositionIndex phi0 = alignment.getFertility(0);
  PositionIndex phiOld = alignment.getFertility(iOld);
  PositionIndex phiNew = alignment.getFertility(iNew);
  Prob p0 = Prob(1.0) - *p1;
  Prob change;
  if (iOld == 0)
  {
    Prob phi0Change =
        (p0 * p0 / *p1) * ((phi0 * (tlen - phi0 + 1.0)) / ((tlen - 2 * phi0 + 1.0) * (tlen - 2 * phi0 + 2.0)));
    Prob phiChange = phiNew + 1.0;
    Prob plus1FertChange = fertilityProb(sNew, phiNew + 1) / fertilityProb(sNew, phiNew);
    Prob ptsChange = translationProb(sNew, t) / translationProb(sOld, t);
    Prob distortionChange = distortionProb(iNew, slen, tlen, j);
    change = phi0Change * phiChange * plus1FertChange * ptsChange * distortionChange;
  }
  else if (iNew == 0)
  {
    Prob phi0Change =
        (*p1 / (p0 * p0)) * (double((tlen - 2.0 * phi0) * (tlen - 2 * phi0 - 1)) / ((1.0 + phi0) * (tlen - phi0)));
    Prob phiChange = 1.0 / phiOld;
    Prob minus1FertChange = fertilityProb(sOld, phiOld - 1) / fertilityProb(sOld, phiOld);
    Prob ptsChange = translationProb(sNew, t) / translationProb(sOld, t);
    Prob distortionChange = Prob(1.0) / distortionProb(iOld, slen, tlen, j);
    change = phi0Change * phiChange * minus1FertChange * ptsChange * distortionChange;
  }
  else
  {
    Prob phiChange = Prob((phiNew + 1.0) / phiOld);
    Prob minus1FertChange = fertilityProb(sOld, phiOld - 1) / fertilityProb(sOld, phiOld);
    Prob plus1FertChange = fertilityProb(sNew, phiNew + 1) / fertilityProb(sNew, phiNew);
    Prob ptsChange = translationProb(sNew, t) / translationProb(sOld, t);
    Prob distortionChange = distortionProb(iNew, slen, tlen, j) / distortionProb(iOld, slen, tlen, j);
    change = phiChange * minus1FertChange * plus1FertChange * ptsChange * distortionChange;
  }
  return change;
}

void Ibm3AlignmentModel::clear()
{
  Ibm2AlignmentModel::clear();
  distortionTable->clear();
  fertilityTable->clear();
  countThreshold = DefaultCountThreshold;
  fertilitySmoothFactor = DefaultFertilitySmoothFactor;
  *p1 = DefaultP1;
  performIbm2Transfer = false;
  hmmModel.reset(nullptr);
}

void Ibm3AlignmentModel::clearTempVars()
{
  Ibm2AlignmentModel::clearTempVars();
  distortionCounts.clear();
  fertilityCounts.clear();
  p0Count = 0;
  p1Count = 0;
  maxSrcWordLen = 0;
  cachedHmmAligLogProbs.clear();
}
