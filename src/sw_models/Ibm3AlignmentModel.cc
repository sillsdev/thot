#include "sw_models/Ibm3AlignmentModel.h"

#include "nlp_common/MathFuncs.h"
#include "sw_models/SwDefs.h"

using namespace std;

Ibm3AlignmentModel::Ibm3AlignmentModel()
    : distortionTable{make_shared<DistortionTable>()}, fertilityTable{make_shared<FertilityTable>()}
{
}

Ibm3AlignmentModel::Ibm3AlignmentModel(Ibm2AlignmentModel& model)
    : Ibm2AlignmentModel{model}, distortionTable{make_shared<DistortionTable>()},
      fertilityTable{make_shared<FertilityTable>()}, performIbm2Transfer{true}
{
}

Ibm3AlignmentModel::Ibm3AlignmentModel(HmmAlignmentModel& model)
    : Ibm2AlignmentModel{model}, distortionTable{make_shared<DistortionTable>()},
      fertilityTable{make_shared<FertilityTable>()}, hmmModel{new HmmAlignmentModel{model}}
{
}

Ibm3AlignmentModel::Ibm3AlignmentModel(Ibm3AlignmentModel& model)
    : Ibm2AlignmentModel{model}, p1{model.p1}, distortionTable{model.distortionTable}, fertilityTable{
                                                                                           model.fertilityTable}
{
}

void Ibm3AlignmentModel::startTraining(int verbosity)
{
  Ibm2AlignmentModel::startTraining(verbosity);

  if (performIbm2Transfer)
  {
    ibm2Transfer();
    performIbm2Transfer = false;
  }
  else if (hmmModel)
  {
    hmmTransfer();
    hmmModel.reset(nullptr);
  }
}

void Ibm3AlignmentModel::ibm2Transfer()
{
  vector<pair<vector<WordIndex>, vector<WordIndex>>> buffer;
  for (unsigned int n = 0; n < numSentencePairs(); ++n)
  {
    vector<WordIndex> src = getSrcSent(n);
    vector<WordIndex> trg = getTrgSent(n);
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

  p0Count = 0.95;
  p1Count = 0.05;

  batchMaximizeProbs();
}

void Ibm3AlignmentModel::ibm2TransferUpdateCounts(const vector<pair<vector<WordIndex>, vector<WordIndex>>>& pairs)
{
#pragma omp parallel for schedule(dynamic)
  for (int line_idx = 0; line_idx < (int)pairs.size(); ++line_idx)
  {
    vector<WordIndex> src = pairs[line_idx].first;
    vector<WordIndex> nsrc = extendWithNullWord(src);
    vector<WordIndex> trg = pairs[line_idx].second;

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
      for (PositionIndex i = 0; i <= slen; ++i)
      {
        double count = probs(i, j) / sum;
        incrementWordPairCounts(nsrc, trg, i, j, count);
      }
    }

    PositionIndex maxFertility = min(tlen + 1, MaxFertility);
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
          beta += pow(prob / (1.0 - prob), phi);
        }
        alpha(phi, i) = beta * pow(-1.0, phi + 1) / phi;
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

void Ibm3AlignmentModel::hmmTransfer()
{
  auto search = [this](const std::vector<WordIndex>& src, const std::vector<WordIndex>& trg,
                       AlignmentInfo& bestAlignment, Matrix<double>& moveScores, Matrix<double>& swapScores) {
    return hmmModel->searchForBestAlignment(MaxFertility, src, trg, bestAlignment, &moveScores, &swapScores);
  };

  vector<pair<vector<WordIndex>, vector<WordIndex>>> buffer;
  for (unsigned int n = 0; n < numSentencePairs(); ++n)
  {
    vector<WordIndex> src = getSrcSent(n);
    vector<WordIndex> trg = getTrgSent(n);
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
  vector<PositionIndex> partitions(MaxFertility, 0);
  vector<PositionIndex> mult(MaxFertility, 0);
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
    hmmModel->cachedAligLogProbs.makeRoomGivenSrcSentLen(src.size());
  }
}

void Ibm3AlignmentModel::initSourceWord(const vector<WordIndex>& nsrc, const vector<WordIndex>& trg, PositionIndex i)
{
  Ibm2AlignmentModel::initSourceWord(nsrc, trg, i);

  PositionIndex slen = (PositionIndex)nsrc.size() - 1;
  PositionIndex tlen = (PositionIndex)trg.size();

  distortionTable->reserveSpace(i, slen, tlen);

  DistortionKey key{i, slen, tlen};
  DistortionCountsElem& distortionEntry = distortionCounts[key];
  if (distortionEntry.size() < trg.size())
    distortionEntry.resize(trg.size(), 0);
}

void Ibm3AlignmentModel::addTranslationOptions(vector<vector<WordIndex>>& insertBuffer)
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

void Ibm3AlignmentModel::batchUpdateCounts(const vector<pair<vector<WordIndex>, vector<WordIndex>>>& pairs)
{
  auto search = [this](const vector<WordIndex>& src, const vector<WordIndex>& trg, AlignmentInfo& bestAlignment,
                       Matrix<double>& moveScores, Matrix<double>& swapScores) {
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
    vector<WordIndex> src = pairs[line_idx].first;
    vector<WordIndex> nsrc = extendWithNullWord(src);
    vector<WordIndex> trg = pairs[line_idx].second;

    PositionIndex slen = (PositionIndex)nsrc.size() - 1;
    PositionIndex tlen = (PositionIndex)trg.size();

    AlignmentInfo alignment(slen, tlen);
    Matrix<double> moveScores, swapScores;
    Prob aligProb = search(src, trg, alignment, moveScores, swapScores);
    Matrix<double> moveCounts(slen + 1, tlen + 1), swapCounts(slen + 1, tlen + 1);
    vector<double> negMove((size_t)tlen + 1), negSwap((size_t)tlen + 1), plus1Fert((size_t)slen + 1),
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
        incrementWordPairCounts(nsrc, trg, i, j, count);

        if (i == 0)
          incrementTargetWordCounts(nsrc, trg, alignment, j, aligProb / totalCount);
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
  }
}

void Ibm3AlignmentModel::incrementWordPairCounts(const vector<WordIndex>& nsrc, const vector<WordIndex>& trg,
                                                 PositionIndex i, PositionIndex j, double count)
{
  Ibm2AlignmentModel::incrementWordPairCounts(nsrc, trg, i, j, count);

  DistortionKey key{i, (PositionIndex)nsrc.size() - 1, (PositionIndex)trg.size()};

#pragma omp atomic
  distortionCounts[key][j - 1] += count;
}

void Ibm3AlignmentModel::incrementTargetWordCounts(const vector<WordIndex>& nsrc, const vector<WordIndex>& trg,
                                                   const AlignmentInfo& alignment, PositionIndex j, double count)
{
}

void Ibm3AlignmentModel::batchMaximizeProbs()
{
  Ibm2AlignmentModel::batchMaximizeProbs();

#pragma omp parallel for schedule(dynamic)
  for (int asIndex = 0; asIndex < (int)distortionCounts.size(); ++asIndex)
  {
    double denom = 0;
    const pair<DistortionKey, DistortionCountsElem>& p = distortionCounts.getAt(asIndex);
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

#pragma omp parallel for schedule(dynamic)
  for (int s = 0; s < (int)fertilityCounts.size(); ++s)
  {
    double denom = 0;
    FertilityCountsElem& elem = fertilityCounts[s];
    for (PositionIndex phi = 0; phi < (PositionIndex)elem.size(); ++phi)
    {
      double numer = elem[phi];
      denom += numer;
      fertilityTable->setNumerator(s, phi, (float)log(numer));
      elem[phi] = 0.0;
    }
    if (denom == 0)
      denom = 1;
    fertilityTable->setDenominator(s, (float)log(denom));
  }

  p1 = p1Count / (p1Count + p0Count);
}

Prob Ibm3AlignmentModel::distortionProb(PositionIndex i, PositionIndex slen, PositionIndex tlen, PositionIndex j)
{
  double logProb = unsmoothedLogDistortionProb(i, slen, tlen, j);
  double prob = logProb == SMALL_LG_NUM ? 1.0 / tlen : exp(logProb);
  return max(prob, SW_PROB_SMOOTH);
}

LgProb Ibm3AlignmentModel::logDistortionProb(PositionIndex i, PositionIndex slen, PositionIndex tlen, PositionIndex j)
{
  double logProb = unsmoothedLogDistortionProb(i, slen, tlen, j);
  if (logProb == SMALL_LG_NUM)
    logProb = log(1.0 / tlen);
  return max(logProb, SW_LOG_PROB_SMOOTH);
}

double Ibm3AlignmentModel::unsmoothedDistortionProb(PositionIndex i, PositionIndex slen, PositionIndex tlen,
                                                    PositionIndex j)
{
  return exp(unsmoothedLogDistortionProb(i, slen, tlen, j));
}

double Ibm3AlignmentModel::unsmoothedLogDistortionProb(PositionIndex i, PositionIndex slen, PositionIndex tlen,
                                                       PositionIndex j)
{
  bool found;
  double numer = distortionTable->getNumerator(i, slen, tlen, j, found);
  if (found)
  {
    // numerator for pair ds,j exists
    double denom = distortionTable->getDenominator(i, slen, tlen, found);
    if (found)
      return numer - denom;
  }
  return SMALL_LG_NUM;
}

Prob Ibm3AlignmentModel::fertilityProb(WordIndex s, PositionIndex phi)
{
  double logProb = unsmoothedLogFertilityProb(s, phi);
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
  return max(prob, SW_PROB_SMOOTH);
}

LgProb Ibm3AlignmentModel::logFertilityProb(WordIndex s, PositionIndex phi)
{
  double logProb = unsmoothedLogFertilityProb(s, phi);
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
  return max(logProb, SW_LOG_PROB_SMOOTH);
}

double Ibm3AlignmentModel::unsmoothedFertilityProb(WordIndex s, PositionIndex phi)
{
  return exp(unsmoothedLogFertilityProb(s, phi));
}

double Ibm3AlignmentModel::unsmoothedLogFertilityProb(WordIndex s, PositionIndex phi)
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

LgProb Ibm3AlignmentModel::getBestAlignment(const vector<WordIndex>& srcSentence, const vector<WordIndex>& trgSentence,
                                            vector<PositionIndex>& bestAlignment)
{
  PositionIndex slen = (PositionIndex)srcSentence.size();
  PositionIndex tlen = (PositionIndex)trgSentence.size();

  AlignmentInfo bestAlignmentInfo(slen, tlen);
  LgProb lgProb = getSentenceLengthLgProb(slen, tlen);
  lgProb += searchForBestAlignment(srcSentence, trgSentence, bestAlignmentInfo).get_lp();

  bestAlignment = bestAlignmentInfo.getAlignment();

  return lgProb;
}

LgProb Ibm3AlignmentModel::getAlignmentLgProb(const vector<WordIndex>& srcSentence,
                                              const vector<WordIndex>& trgSentence,
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
    return getSentenceLengthLgProb(slen, tlen)
         + calcProbOfAlignment(addNullWordToWidxVec(srcSentence), trgSentence, alignment, verbose).get_lp();
  }
}

Prob Ibm3AlignmentModel::calcProbOfAlignment(const vector<WordIndex>& nsrc, const vector<WordIndex>& trg,
                                             AlignmentInfo& alignment, int verbose)
{
  if (alignment.getProb() >= 0.0)
    return alignment.getProb();

  PositionIndex slen = PositionIndex(nsrc.size() - 1);
  PositionIndex tlen = PositionIndex(trg.size());

  if (verbose)
    cerr << "Obtaining IBM Model 3 prob..." << endl;

  Prob p0 = Prob(1.0) - p1;

  PositionIndex phi0 = alignment.getFertility(0);
  Prob prob = pow(p0, double(tlen - 2 * phi0)) * pow(p1, double(phi0));

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

    prob *= pts(s, t) * distortionProb(i, slen, tlen, j);
  }
  alignment.setProb(prob);
  return prob;
}

LgProb Ibm3AlignmentModel::getSumLgProb(const vector<WordIndex>& srcSentence, const vector<WordIndex>& trgSentence,
                                        int verbose)
{
  PositionIndex slen = (PositionIndex)srcSentence.size();
  PositionIndex tlen = (PositionIndex)trgSentence.size();
  vector<PositionIndex> nsrc = addNullWordToWidxVec(srcSentence);

  if (verbose)
    cerr << "Obtaining Sum IBM Model 3 logprob..." << endl;

  Prob p0 = 1.0 - (double)p1;

  LgProb lgProb = getSentenceLengthLgProb(slen, tlen);
  LgProb fertilityContrib = 0;
  for (PositionIndex fertility = 0; fertility < min(tlen, MaxFertility); ++fertility)
  {
    Prob sump = 0;
    Prob prob = 1.0;
    PositionIndex phi0 = fertility;
    prob *= pow(p1, double(phi0)) * pow(p0, double(tlen - 2 * phi0));

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
    cerr << "- Fertility contribution= " << fertilityContrib << endl;
  lgProb += fertilityContrib;

  LgProb lexDistorionContrib = 0;
  for (PositionIndex j = 1; j <= tlen; ++j)
  {
    Prob sump = 0;
    for (PositionIndex i = 0; i <= slen; ++i)
    {
      WordIndex s = nsrc[i];
      WordIndex t = trgSentence[j - 1];

      sump += pts(s, t) * distortionProb(i, slen, tlen, j);
    }
    lexDistorionContrib += sump.get_lp();
  }

  if (verbose)
    cerr << "- Lexical plus distortion contribution= " << lexDistorionContrib << endl;
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
    cerr << "Loading IBM 3 Model data..." << endl;

  // Load file with distortion nd values
  string distortionNumDenFile = prefFileName;
  distortionNumDenFile = distortionNumDenFile + ".distnd";
  retVal = distortionTable->load(distortionNumDenFile.c_str(), verbose);
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  // Load file with fertility nd values
  string fertilityNumDenFile = prefFileName;
  fertilityNumDenFile = distortionNumDenFile + ".fertnd";
  return fertilityTable->load(fertilityNumDenFile.c_str(), verbose);
}

bool Ibm3AlignmentModel::print(const char* prefFileName, int verbose)
{
  // Print IBM 2 Model data
  bool retVal = Ibm2AlignmentModel::print(prefFileName);
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  // Print file with distortion nd values
  string distortionNumDenFile = prefFileName;
  distortionNumDenFile = distortionNumDenFile + ".distnd";
  retVal = distortionTable->print(distortionNumDenFile.c_str());
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  // Load file with fertility nd values
  string fertilityNumDenFile = prefFileName;
  fertilityNumDenFile = distortionNumDenFile + ".fertnd";
  return fertilityTable->print(fertilityNumDenFile.c_str());
}

Prob Ibm3AlignmentModel::searchForBestAlignment(const vector<WordIndex>& src, const vector<WordIndex>& trg,
                                                AlignmentInfo& bestAlignment, Matrix<double>* moveScores,
                                                Matrix<double>* swapScores)
{
  PositionIndex slen = (PositionIndex)src.size();
  PositionIndex tlen = (PositionIndex)trg.size();

  vector<WordIndex> nsrc = extendWithNullWord(src);

  // start with IBM-2 alignment
  getInitialAlignmentForSearch(nsrc, trg, bestAlignment);

  if (moveScores != nullptr)
    moveScores->resize(slen + 1, tlen + 1);
  if (swapScores != nullptr)
    swapScores->resize(tlen + 1, tlen + 1);

  // hillclimbing search
  int bestChangeType = -1;
  while (bestChangeType != 0)
  {
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
          double changeScore = swapScore(nsrc, trg, j, j1, bestAlignment);
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
        if (i != iAlig && (i != 0 || (tlen >= 2 * (bestAlignment.getFertility(0) + 1)))
            && bestAlignment.getFertility(i) + 1 < MaxFertility)
        {
          double changeScore = moveScore(nsrc, trg, i, j, bestAlignment);
          if (moveScores != nullptr)
            moveScores->set(i, j, changeScore);
          if (changeScore > bestChangeScore)
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
  }
  return calcProbOfAlignment(nsrc, trg, bestAlignment);
}

void Ibm3AlignmentModel::getInitialAlignmentForSearch(const vector<WordIndex>& nsrc, const vector<WordIndex>& trg,
                                                      AlignmentInfo& alignment)
{
  PositionIndex slen = (PositionIndex)nsrc.size() - 1;
  PositionIndex tlen = (PositionIndex)trg.size();

  vector<PositionIndex> fertility((size_t)slen + 1, 0);

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
        double prob = pts(s, t) * aProb(j, slen, tlen, i);
        if (prob > bestProb)
        {
          iBest = i;
          bestProb = prob;
        }
      }
    }
    alignment.set(j, iBest);
    fertility[iBest]++;
  }
}

double Ibm3AlignmentModel::swapScore(const vector<WordIndex>& nsrc, const vector<WordIndex>& trg, PositionIndex j1,
                                     PositionIndex j2, AlignmentInfo& alignment)
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
  Prob score = (pts(s2, t1) / pts(s1, t1)) * (pts(s1, t2) / pts(s2, t2));
  if (i1 > 0)
    score *= distortionProb(i1, slen, tlen, j2) / distortionProb(i1, slen, tlen, j1);
  if (i2 > 0)
    score *= distortionProb(i2, slen, tlen, j1) / distortionProb(i2, slen, tlen, j2);
  return score;
}

double Ibm3AlignmentModel::moveScore(const vector<WordIndex>& nsrc, const vector<WordIndex>& trg, PositionIndex iNew,
                                     PositionIndex j, AlignmentInfo& alignment)
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
  Prob p0 = Prob(1.0) - p1;
  Prob score;
  if (iOld == 0)
  {
    score = (p0 * p0 / p1) * ((phi0 * (tlen - phi0 + 1.0)) / ((tlen - 2 * phi0 + 1.0) * (tlen - 2 * phi0 + 2.0)))
          * (phiNew + 1.0) * (fertilityProb(sNew, phiNew + 1) / fertilityProb(sNew, phiNew))
          * (pts(sNew, t) / pts(sOld, t)) * distortionProb(iNew, slen, tlen, j);
  }
  else if (iNew == 0)
  {
    score = (p1 / (p0 * p0)) * (double((tlen - 2.0 * phi0) * (tlen - 2 * phi0 - 1)) / ((1.0 + phi0) * (tlen - phi0)))
          * (1.0 / phiOld) * (fertilityProb(sOld, phiOld - 1) / fertilityProb(sOld, phiOld))
          * (pts(sNew, t) / pts(sOld, t)) * (Prob(1.0) / distortionProb(iOld, slen, tlen, j));
  }
  else
  {
    score = Prob((phiNew + 1.0) / phiOld) * (fertilityProb(sOld, phiOld - 1) / fertilityProb(sOld, phiOld))
          * (fertilityProb(sNew, phiNew + 1) / fertilityProb(sNew, phiNew)) * (pts(sNew, t) / pts(sOld, t))
          * (distortionProb(iNew, slen, tlen, j) / distortionProb(iOld, slen, tlen, j));
  }
  return score;
}

void Ibm3AlignmentModel::clear()
{
  Ibm2AlignmentModel::clear();
  distortionTable->clear();
  fertilityTable->clear();
  p1 = 0.5;
  p0Count = 0;
  p1Count = 0;
  performIbm2Transfer = false;
}

void Ibm3AlignmentModel::clearTempVars()
{
  Ibm2AlignmentModel::clearTempVars();
  distortionCounts.clear();
  fertilityCounts.clear();
  p0Count = 0;
  p1Count = 0;
}
