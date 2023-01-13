#include "sw_models/Ibm4AlignmentModel.h"

#include "nlp_common/Exceptions.h"
#include "nlp_common/MathFuncs.h"
#include "sw_models/SwDefs.h"

Ibm4AlignmentModel::Ibm4AlignmentModel()
    : headDistortionTable{std::make_shared<HeadDistortionTable>()}, nonheadDistortionTable{
                                                                        std::make_shared<NonheadDistortionTable>()}
{
}

Ibm4AlignmentModel::Ibm4AlignmentModel(HmmAlignmentModel& model)
    : Ibm3AlignmentModel{model}, headDistortionTable{std::make_shared<HeadDistortionTable>()},
      nonheadDistortionTable{std::make_shared<NonheadDistortionTable>()}
{
}

Ibm4AlignmentModel::Ibm4AlignmentModel(Ibm3AlignmentModel& model)
    : Ibm3AlignmentModel{model}, headDistortionTable{std::make_shared<HeadDistortionTable>()},
      nonheadDistortionTable{std::make_shared<NonheadDistortionTable>()}, ibm3Model{new Ibm3AlignmentModel{model}}
{
}

Ibm4AlignmentModel::Ibm4AlignmentModel(Ibm4AlignmentModel& model)
    : Ibm3AlignmentModel{model}, distortionSmoothFactor{model.distortionSmoothFactor},
      headDistortionTable{model.headDistortionTable}, nonheadDistortionTable{model.nonheadDistortionTable}
{
}

unsigned int Ibm4AlignmentModel::startTraining(int verbosity)
{
  unsigned int count = Ibm3AlignmentModel::startTraining(verbosity);

  nonheadDistortionCounts.resize(wordClasses->getTrgWordClassCount());
  nonheadDistortionTable->reserveSpace(wordClasses->getTrgWordClassCount() - 1);
  return count;
}

void Ibm4AlignmentModel::initWordPair(const std::vector<WordIndex>& nsrc, const std::vector<WordIndex>& trg,
                                      PositionIndex i, PositionIndex j)
{
  WordIndex s = nsrc[i];
  WordIndex t = trg[j - 1];
  WordClassIndex srcWordClass = wordClasses->getSrcWordClass(s);
  WordClassIndex trgWordClass = wordClasses->getTrgWordClass(t);
  headDistortionTable->reserveSpace(srcWordClass, trgWordClass);
}

double Ibm4AlignmentModel::updateCounts(const std::vector<WordIndex>& nsrc, const std::vector<WordIndex>& trg,
                                        AlignmentInfo& alignment, double aligProb, const Matrix<double>& moveScores,
                                        const Matrix<double>& swapScores)
{
  double totalCount = Ibm3AlignmentModel::updateCounts(nsrc, trg, alignment, aligProb, moveScores, swapScores);

  PositionIndex slen = (PositionIndex)nsrc.size() - 1;
  PositionIndex tlen = (PositionIndex)trg.size();

  double sum = 0;
  // mark this as unused
  (void)sum;
  double normalizedAligProb = aligProb / totalCount;

  incrementDistortionCounts(nsrc, trg, alignment, normalizedAligProb);
  sum += normalizedAligProb;

  for (PositionIndex j = 1; j <= tlen; ++j)
  {
    PositionIndex iOld = alignment.get(j);
    for (PositionIndex i = 0; i <= slen; ++i)
    {
      if (i != iOld)
      {
        double count = normalizedAligProb * moveScores(i, j);
        if (count > countThreshold)
        {
          alignment.set(j, i);
          incrementDistortionCounts(nsrc, trg, alignment, count);
          alignment.set(j, iOld);
          sum += count;
        }
      }
    }

    for (PositionIndex j1 = j + 1; j1 <= tlen; ++j1)
    {
      PositionIndex iOld1 = alignment.get(j1);
      if (iOld != iOld1)
      {
        double count = normalizedAligProb * swapScores(j, j1);
        if (count > countThreshold)
        {
          alignment.set(j, iOld1);
          alignment.set(j1, iOld);
          incrementDistortionCounts(nsrc, trg, alignment, count);
          alignment.set(j, iOld);
          alignment.set(j1, iOld1);
          sum += count;
        }
      }
    }
  }
  assert(fabs(1.0 - sum) < 0.01);
  return totalCount;
}

void Ibm4AlignmentModel::incrementDistortionCounts(const std::vector<WordIndex>& nsrc,
                                                   const std::vector<WordIndex>& trg, const AlignmentInfo& alignment,
                                                   double count)
{
  for (PositionIndex j = 1; j <= trg.size(); ++j)
  {
    PositionIndex i = alignment.get(j);
    if (i == 0)
      continue;

    WordIndex t = trg[j - 1];
    WordClassIndex trgWordClass = wordClasses->getTrgWordClass(t);
    if (alignment.isHead(j))
    {
      PositionIndex prevCept = alignment.getPrevCept(i);
      WordIndex sPrev = nsrc[prevCept];
      WordClassIndex srcWordClass = wordClasses->getSrcWordClass(sPrev);
      HeadDistortionKey key{srcWordClass, trgWordClass};
      int dj = j - alignment.getCenter(prevCept);

#pragma omp critical(headDistortionCounts)
      headDistortionCounts[key][dj] += count;
    }
    else
    {
      PositionIndex prevInCept = alignment.getPrevInCept(j);
      int dj = j - prevInCept;

#pragma omp critical(nonheadDistortionCounts)
      nonheadDistortionCounts[trgWordClass][dj] += count;
    }
  }
}

void Ibm4AlignmentModel::train(int verbosity)
{
  if (ibm3Model)
  {
    ibm3Transfer();
    ibm3Model.reset(nullptr);
  }
  else
  {
    Ibm3AlignmentModel::train(verbosity);
  }
}

void Ibm4AlignmentModel::batchMaximizeProbs()
{
  Ibm3AlignmentModel::batchMaximizeProbs();

#pragma omp parallel for schedule(dynamic)
  for (int index = 0; index < (int)headDistortionCounts.size(); ++index)
  {
    double denom = 0;
    const std::pair<HeadDistortionKey, HeadDistortionCountsElem>& p = headDistortionCounts.getAt(index);
    const HeadDistortionKey& key = p.first;
    HeadDistortionCountsElem& elem = const_cast<HeadDistortionCountsElem&>(p.second);
    for (auto& pair : elem)
    {
      double numer = pair.second;
      denom += numer;
      float logNumer = numer == 0 ? SMALL_LG_NUM : (float)log(numer);
      headDistortionTable->setNumerator(key.srcWordClass, key.trgWordClass, pair.first, logNumer);
      pair.second = 0.0;
    }
    if (denom == 0)
      denom = 1;
    float logDenom = (float)log(denom);
    headDistortionTable->setDenominator(key.srcWordClass, key.trgWordClass, logDenom);
  }

#pragma omp parallel for schedule(dynamic)
  for (int targetWordClass = 0; targetWordClass < (int)nonheadDistortionCounts.size(); ++targetWordClass)
  {
    double denom = 0;
    NonheadDistortionCountsElem& elem = nonheadDistortionCounts[targetWordClass];
    for (auto& pair : elem)
    {
      double numer = pair.second;
      denom += numer;
      float logNumer = numer == 0 ? SMALL_LG_NUM : (float)log(numer);
      nonheadDistortionTable->setNumerator(targetWordClass, pair.first, logNumer);
      pair.second = 0.0;
    }
    if (denom == 0)
      denom = 1;
    float logDenom = (float)log(denom);
    nonheadDistortionTable->setDenominator(targetWordClass, logDenom);
  }
}

void Ibm4AlignmentModel::loadConfig(const YAML::Node& config)
{
  Ibm3AlignmentModel::loadConfig(config);

  distortionSmoothFactor = config["distortionSmoothFactor"].as<double>();
}

void Ibm4AlignmentModel::createConfig(YAML::Emitter& out)
{
  Ibm3AlignmentModel::createConfig(out);

  out << YAML::Key << "distortionSmoothFactor" << YAML::Value << distortionSmoothFactor;
}

LgProb Ibm4AlignmentModel::computeSumLogProb(const std::vector<WordIndex>& srcSentence,
                                             const std::vector<WordIndex>& trgSentence, int verbose)
{
  throw NotImplemented();
}

double Ibm4AlignmentModel::getDistortionSmoothFactor()
{
  return distortionSmoothFactor;
}

void Ibm4AlignmentModel::setDistortionSmoothFactor(double distortionSmoothFactor)
{
  this->distortionSmoothFactor = distortionSmoothFactor;
}

Prob Ibm4AlignmentModel::headDistortionProb(WordClassIndex srcWordClass, WordClassIndex trgWordClass,
                                            PositionIndex tlen, int dj)
{
  bool found;
  double logProb = unsmoothedHeadDistortionLogProb(srcWordClass, trgWordClass, dj, found);
  if (!found)
    return SW_PROB_SMOOTH;
  double prob = exp(logProb);
  prob = (distortionSmoothFactor / (2.0 * tlen - 1)) + ((1.0 - distortionSmoothFactor) * prob);
  return std::max(prob, SW_PROB_SMOOTH);
}

LgProb Ibm4AlignmentModel::headDistortionLogProb(WordClassIndex srcWordClass, WordClassIndex trgWordClass,
                                                 PositionIndex tlen, int dj)
{
  bool found;
  double logProb = unsmoothedHeadDistortionLogProb(srcWordClass, trgWordClass, dj, found);
  if (!found)
    return SW_LOG_PROB_SMOOTH;
  logProb = MathFuncs::lns_sumlog(log(distortionSmoothFactor / (2.0 * tlen - 1)),
                                  (log(1.0 - distortionSmoothFactor) + logProb));
  return std::max(logProb, SW_LOG_PROB_SMOOTH);
}

double Ibm4AlignmentModel::unsmoothedHeadDistortionLogProb(WordClassIndex srcWordClass, WordClassIndex trgWordClass,
                                                           int dj, bool& found)
{
  double denom = headDistortionTable->getDenominator(srcWordClass, trgWordClass, found);
  if (found)
  {
    double numer = headDistortionTable->getNumerator(srcWordClass, trgWordClass, dj, found);
    if (found)
      return numer - denom;
    found = true;
  }
  return SMALL_LG_NUM;
}

Prob Ibm4AlignmentModel::nonheadDistortionProb(WordClassIndex trgWordClass, PositionIndex tlen, int dj)
{
  bool found;
  double logProb = unsmoothedNonheadDistortionLogProb(trgWordClass, dj, found);
  if (!found)
    return SW_PROB_SMOOTH;
  double prob = exp(logProb);
  prob = (distortionSmoothFactor / (tlen - 1)) + ((1.0 - distortionSmoothFactor) * prob);
  return std::max(prob, SW_PROB_SMOOTH);
}

LgProb Ibm4AlignmentModel::nonheadDistortionLogProb(WordClassIndex trgWordClass, PositionIndex tlen, int dj)
{
  bool found;
  double logProb = unsmoothedNonheadDistortionLogProb(trgWordClass, dj, found);
  if (!found)
    return SW_LOG_PROB_SMOOTH;
  logProb =
      MathFuncs::lns_sumlog(log(distortionSmoothFactor / (tlen - 1)), (log(1.0 - distortionSmoothFactor) + logProb));
  return std::max(logProb, SW_LOG_PROB_SMOOTH);
}

double Ibm4AlignmentModel::unsmoothedNonheadDistortionLogProb(WordClassIndex targetWordClass, int dj, bool& found)
{
  double denom = nonheadDistortionTable->getDenominator(targetWordClass, found);
  if (found)
  {
    double numer = nonheadDistortionTable->getNumerator(targetWordClass, dj, found);
    if (found)
      return numer - denom;
    found = true;
  }
  return SMALL_LG_NUM;
}

Prob Ibm4AlignmentModel::calcProbOfAlignment(const std::vector<WordIndex>& nsrc, const std::vector<WordIndex>& trg,
                                             AlignmentInfo& alignment, int verbose)
{
  PositionIndex slen = (PositionIndex)nsrc.size() - 1;
  PositionIndex tlen = (PositionIndex)trg.size();

  if (verbose)
    std::cerr << "Obtaining IBM Model 4 prob..." << std::endl;

  Prob p0 = Prob(1.0) - *p1;

  PositionIndex phi0 = alignment.getFertility(0);
  Prob prob = pow(p0, double(tlen - 2 * phi0)) * pow(*p1, double(phi0));

  for (PositionIndex phi = 1; phi <= phi0; ++phi)
    prob *= double(size_t{tlen - phi0 - phi} + 1) / phi;

  for (PositionIndex i = 1; i <= slen; ++i)
  {
    WordIndex s = nsrc[i];
    PositionIndex phi = alignment.getFertility(i);
    prob *= fertilityProb(s, phi);
  }

  for (PositionIndex j = 1; j <= tlen; ++j)
  {
    PositionIndex i = alignment.get(j);
    WordIndex s = nsrc[i];
    WordIndex t = trg[j - 1];

    prob *= translationProb(s, t);
  }

  prob *= calcDistortionProbOfAlignment(nsrc, trg, alignment);

  return prob;
}

Prob Ibm4AlignmentModel::calcDistortionProbOfAlignment(const std::vector<WordIndex>& nsrc,
                                                       const std::vector<WordIndex>& trg, AlignmentInfo& alignment)
{
  PositionIndex tlen = (PositionIndex)trg.size();

  Prob prob = 1.0;
  for (PositionIndex j = 1; j <= tlen; ++j)
  {
    PositionIndex i = alignment.get(j);
    if (i > 0)
    {
      WordIndex t = trg[j - 1];
      WordClassIndex trgWordClass = wordClasses->getTrgWordClass(t);
      if (alignment.isHead(j))
      {
        PositionIndex prevCept = alignment.getPrevCept(i);
        WordIndex sPrev = nsrc[prevCept];
        WordClassIndex srcWordClass = wordClasses->getSrcWordClass(sPrev);
        int dj = j - alignment.getCenter(prevCept);
        prob *= headDistortionProb(srcWordClass, trgWordClass, tlen, dj);
      }
      else
      {
        PositionIndex prevInCept = alignment.getPrevInCept(j);
        int dj = j - prevInCept;
        prob *= nonheadDistortionProb(trgWordClass, tlen, dj);
      }
    }
  }
  return prob;
}

bool Ibm4AlignmentModel::load(const char* prefFileName, int verbose)
{
  // Load IBM 3 Model data
  bool retVal = Ibm3AlignmentModel::load(prefFileName, verbose);
  if (retVal == THOT_ERROR)
    return retVal;

  if (verbose)
    std::cerr << "Loading IBM 4 Model data..." << std::endl;

  // Load file with head distortion nd values
  std::string headDistortionNumDenFile = prefFileName;
  headDistortionNumDenFile = headDistortionNumDenFile + ".h_distnd";
  retVal = headDistortionTable->load(headDistortionNumDenFile.c_str(), verbose);
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  // Load file with nonhead distortion nd values
  std::string nonheadDistortionNumDenFile = prefFileName;
  nonheadDistortionNumDenFile = nonheadDistortionNumDenFile + ".nh_distnd";
  retVal = nonheadDistortionTable->load(nonheadDistortionNumDenFile.c_str(), verbose);
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  return THOT_OK;
}

bool Ibm4AlignmentModel::print(const char* prefFileName, int verbose)
{
  // Print IBM 3 Model data
  bool retVal = Ibm3AlignmentModel::print(prefFileName);
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  // Print file with head distortion nd values
  std::string headDistortionNumDenFile = prefFileName;
  headDistortionNumDenFile = headDistortionNumDenFile + ".h_distnd";
  retVal = headDistortionTable->print(headDistortionNumDenFile.c_str());
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  // Print file with nonhead distortion nd values
  std::string nonheadDistortionNumDenFile = prefFileName;
  nonheadDistortionNumDenFile = nonheadDistortionNumDenFile + ".nh_distnd";
  retVal = nonheadDistortionTable->print(nonheadDistortionNumDenFile.c_str());
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  return THOT_OK;
}

double Ibm4AlignmentModel::swapScore(const std::vector<WordIndex>& nsrc, const std::vector<WordIndex>& trg,
                                     PositionIndex j1, PositionIndex j2, AlignmentInfo& alignment,
                                     double& cachedAlignmentValue)
{
  PositionIndex i1 = alignment.get(j1);
  PositionIndex i2 = alignment.get(j2);
  if (i1 == i2)
    return 1.0;

  WordIndex s1 = nsrc[i1];
  WordIndex s2 = nsrc[i2];
  WordIndex t1 = trg[j1 - 1];
  WordIndex t2 = trg[j2 - 1];

  Prob change =
      (translationProb(s2, t1) / translationProb(s1, t1)) * (translationProb(s1, t2) / translationProb(s2, t2));

  if (cachedAlignmentValue < 0)
    cachedAlignmentValue = calcDistortionProbOfAlignment(nsrc, trg, alignment);
  Prob oldDistortionProb = cachedAlignmentValue;

  alignment.set(j1, i2);
  alignment.set(j2, i1);
  Prob newDistortionProb = calcDistortionProbOfAlignment(nsrc, trg, alignment);
  alignment.set(j1, i1);
  alignment.set(j2, i2);

  change *= newDistortionProb / oldDistortionProb;

  return change;
}

double Ibm4AlignmentModel::moveScore(const std::vector<WordIndex>& nsrc, const std::vector<WordIndex>& trg,
                                     PositionIndex iNew, PositionIndex j, AlignmentInfo& alignment,
                                     double& cachedAlignmentValue)
{
  PositionIndex iOld = alignment.get(j);
  if (iOld == iNew)
    return 1.0;

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
    Prob plus1FertChange = fertilityProb(sNew, phiNew + 1) / fertilityProb(sNew, phiNew);
    Prob ptsChange = translationProb(sNew, t) / translationProb(sOld, t);
    change = phi0Change * plus1FertChange * ptsChange;
  }
  else if (iNew == 0)
  {
    Prob phi0Change =
        (*p1 / (p0 * p0)) * (double((tlen - 2.0 * phi0) * (tlen - 2 * phi0 - 1)) / ((1.0 + phi0) * (tlen - phi0)));
    Prob minus1FertChange = fertilityProb(sOld, phiOld - 1) / fertilityProb(sOld, phiOld);
    Prob ptsChange = translationProb(sNew, t) / translationProb(sOld, t);
    change = phi0Change * minus1FertChange * ptsChange;
  }
  else
  {
    Prob minus1FertChange = fertilityProb(sOld, phiOld - 1) / fertilityProb(sOld, phiOld);
    Prob plus1FertChange = fertilityProb(sNew, phiNew + 1) / fertilityProb(sNew, phiNew);
    Prob ptsChange = translationProb(sNew, t) / translationProb(sOld, t);
    change = minus1FertChange * plus1FertChange * ptsChange;
  }

  if (cachedAlignmentValue < 0)
    cachedAlignmentValue = calcDistortionProbOfAlignment(nsrc, trg, alignment);
  Prob oldDistortionProb = cachedAlignmentValue;

  alignment.set(j, iNew);
  Prob newDistortionProb = calcDistortionProbOfAlignment(nsrc, trg, alignment);
  alignment.set(j, iOld);

  change *= newDistortionProb / oldDistortionProb;

  return change;
}

void Ibm4AlignmentModel::ibm3Transfer()
{
  auto search = [this](const std::vector<WordIndex>& src, const std::vector<WordIndex>& trg,
                       AlignmentInfo& bestAlignment, Matrix<double>& moveScores, Matrix<double>& swapScores) {
    return ibm3Model->searchForBestAlignment(src, trg, bestAlignment, &moveScores, &swapScores);
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

void Ibm4AlignmentModel::clear()
{
  Ibm3AlignmentModel::clear();
  headDistortionTable->clear();
  nonheadDistortionTable->clear();
  distortionSmoothFactor = DefaultDistortionSmoothFactor;
}

void Ibm4AlignmentModel::clearTempVars()
{
  Ibm3AlignmentModel::clearTempVars();
  headDistortionCounts.clear();
  nonheadDistortionCounts.clear();
}
