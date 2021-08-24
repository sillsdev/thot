#include "sw_models/Ibm4AlignmentModel.h"

#include "nlp_common/Exceptions.h"
#include "nlp_common/MathFuncs.h"
#include "sw_models/SwDefs.h"

using namespace std;

Ibm4AlignmentModel::Ibm4AlignmentModel()
    : distortionSmoothFactor{0.2}, wordClasses{make_shared<WordClasses>()},
      headDistortionTable{make_shared<HeadDistortionTable>()}, nonheadDistortionTable{
                                                                   make_shared<NonheadDistortionTable>()}
{
}

Ibm4AlignmentModel::Ibm4AlignmentModel(Ibm3AlignmentModel& model)
    : Ibm3AlignmentModel{model}, distortionSmoothFactor{0.2}, wordClasses{make_shared<WordClasses>()},
      headDistortionTable{make_shared<HeadDistortionTable>()}, nonheadDistortionTable{
                                                                   make_shared<NonheadDistortionTable>()}
{
}

Ibm4AlignmentModel::Ibm4AlignmentModel(Ibm4AlignmentModel& model)
    : Ibm3AlignmentModel{model}, distortionSmoothFactor{model.distortionSmoothFactor}, wordClasses{model.wordClasses},
      headDistortionTable{model.headDistortionTable}, nonheadDistortionTable{model.nonheadDistortionTable}
{
}

void Ibm4AlignmentModel::startTraining(int verbosity)
{
  Ibm3AlignmentModel::startTraining(verbosity);

  nonheadDistortionCounts.resize(wordClasses->getTrgWordClassCount());
  nonheadDistortionTable->reserveSpace(wordClasses->getTrgWordClassCount() - 1);
}

void Ibm4AlignmentModel::initWordPair(const vector<WordIndex>& nsrc, const vector<WordIndex>& trg, PositionIndex i,
                                      PositionIndex j)
{
  WordIndex s = nsrc[i];
  WordIndex t = trg[j - 1];
  WordClassIndex srcWordClass = wordClasses->getSrcWordClass(s);
  WordClassIndex trgWordClass = wordClasses->getTrgWordClass(t);
  headDistortionTable->reserveSpace(srcWordClass, trgWordClass);
}

void Ibm4AlignmentModel::incrementTargetWordCounts(const vector<WordIndex>& nsrc, const vector<WordIndex>& trg,
                                                   const AlignmentInfo& alignment, PositionIndex j, double count)
{
  PositionIndex i = alignment.get(j);
  if (i == 0)
    return;

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

void Ibm4AlignmentModel::batchMaximizeProbs()
{
  Ibm3AlignmentModel::batchMaximizeProbs();

#pragma omp parallel for schedule(dynamic)
  for (int index = 0; index < (int)headDistortionCounts.size(); ++index)
  {
    double denom = 0;
    const pair<HeadDistortionKey, HeadDistortionCountsElem>& p = headDistortionCounts.getAt(index);
    const HeadDistortionKey& key = p.first;
    HeadDistortionCountsElem& elem = const_cast<HeadDistortionCountsElem&>(p.second);
    for (auto& pair : elem)
    {
      double numer = pair.second;
      if (numer == 0)
        continue;
      denom += numer;
      float logNumer = (float)log(numer);
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
      if (numer == 0)
        continue;
      denom += numer;
      float logNumer = (float)log(numer);
      nonheadDistortionTable->setNumerator(targetWordClass, pair.first, logNumer);
      pair.second = 0.0;
    }
    if (denom == 0)
      denom = 1;
    float logDenom = (float)log(denom);
    nonheadDistortionTable->setDenominator(targetWordClass, logDenom);
  }
}

bool Ibm4AlignmentModel::loadDistortionSmoothFactor(const char* distortionSmoothFactorFile, int verbose)
{
  if (verbose)
    cerr << "Loading file with distortion smoothing interpolation factor from " << distortionSmoothFactorFile << endl;

  AwkInputStream awk;

  if (awk.open(distortionSmoothFactorFile) == THOT_ERROR)
  {
    if (verbose)
      cerr << "Error in file with distortion smoothing interpolation factor, file " << distortionSmoothFactorFile
           << " does not exist. Assuming default value." << endl;
    setDistortionSmoothFactor(0.2, verbose);
    return THOT_OK;
  }
  else
  {
    if (awk.getln())
    {
      if (awk.NF == 1)
      {
        setDistortionSmoothFactor((Prob)atof(awk.dollar(1).c_str()), verbose);
        return THOT_OK;
      }
      else
      {
        if (verbose)
          cerr << "Error: anomalous .dsifactor file, " << distortionSmoothFactorFile << endl;
        return THOT_ERROR;
      }
    }
    else
    {
      if (verbose)
        cerr << "Error: anomalous .dsifactor file, " << distortionSmoothFactorFile << endl;
      return THOT_ERROR;
    }
  }
}

bool Ibm4AlignmentModel::printDistortionSmoothFactor(const char* distortionSmoothFactorFile, int verbose)
{
  ofstream outF;
  outF.open(distortionSmoothFactorFile, ios::out);
  if (!outF)
  {
    if (verbose)
      cerr << "Error while printing file with alignment smoothing interpolation factor." << endl;
    return THOT_ERROR;
  }
  else
  {
    outF << distortionSmoothFactor << endl;
    return THOT_OK;
  }
}

Prob Ibm4AlignmentModel::headDistortionProb(WordClassIndex srcWordClass, WordClassIndex trgWordClass,
                                            PositionIndex tlen, int dj)
{
  double logProb = unsmoothedLogHeadDistortionProb(srcWordClass, trgWordClass, dj);
  double prob = exp(logProb);
  prob = (distortionSmoothFactor / (tlen - 1)) + ((1.0 - distortionSmoothFactor) * prob);
  return max(prob, SW_PROB_SMOOTH);
}

LgProb Ibm4AlignmentModel::logHeadDistortionProb(WordClassIndex srcWordClass, WordClassIndex trgWordClass,
                                                 PositionIndex tlen, int dj)
{
  double logProb = unsmoothedLogHeadDistortionProb(srcWordClass, trgWordClass, dj);
  logProb =
      MathFuncs::lns_sumlog(log(distortionSmoothFactor / (tlen - 1)), (log(1.0 - distortionSmoothFactor) + logProb));
  return max(logProb, SW_LOG_PROB_SMOOTH);
}

LgProb Ibm4AlignmentModel::getSumLgProb(const vector<WordIndex>& srcSentence, const vector<WordIndex>& trgSentence,
                                        int verbose)
{
  throw NotImplemented();
}

void Ibm4AlignmentModel::setDistortionSmoothFactor(double distortionSmoothFactor, int verbose)
{
  this->distortionSmoothFactor = distortionSmoothFactor;
  if (verbose)
    cerr << "Distortion smoothing interpolation factor has been set to " << distortionSmoothFactor << endl;
}

void Ibm4AlignmentModel::addSrcWordClass(WordIndex s, WordClassIndex c)
{
  wordClasses->addSrcWordClass(s, c);
}

void Ibm4AlignmentModel::addTrgWordClass(WordIndex t, WordClassIndex c)
{
  wordClasses->addTrgWordClass(t, c);
}

bool Ibm4AlignmentModel::sentenceLengthIsOk(const vector<WordIndex> sentence)
{
  return !sentence.empty() && sentence.size() <= IBM4_SWM_MAX_SENT_LENGTH;
}

double Ibm4AlignmentModel::unsmoothedHeadDistortionProb(WordClassIndex srcWordClass, WordClassIndex trgWordClass,
                                                        int dj)
{
  return exp(unsmoothedLogHeadDistortionProb(srcWordClass, trgWordClass, dj));
}

double Ibm4AlignmentModel::unsmoothedLogHeadDistortionProb(WordClassIndex srcWordClass, WordClassIndex trgWordClass,
                                                           int dj)
{
  bool found;
  double numer = headDistortionTable->getNumerator(srcWordClass, trgWordClass, dj, found);
  if (found)
  {
    double denom = headDistortionTable->getDenominator(srcWordClass, trgWordClass, found);
    if (found)
      return numer - denom;
  }
  return SMALL_LG_NUM;
}

Prob Ibm4AlignmentModel::nonheadDistortionProb(WordClassIndex trgWordClass, PositionIndex tlen, int dj)
{
  double logProb = unsmoothedLogNonheadDistortionProb(trgWordClass, dj);
  double prob = exp(logProb);
  prob = (distortionSmoothFactor / (tlen - 1)) + ((1.0 - distortionSmoothFactor) * prob);
  return max(prob, SW_PROB_SMOOTH);
}

LgProb Ibm4AlignmentModel::logNonheadDistortionProb(WordClassIndex trgWordClass, PositionIndex tlen, int dj)
{
  double logProb = unsmoothedLogNonheadDistortionProb(trgWordClass, dj);
  logProb =
      MathFuncs::lns_sumlog(log(distortionSmoothFactor / (tlen - 1)), (log(1.0 - distortionSmoothFactor) + logProb));
  return max(logProb, SW_LOG_PROB_SMOOTH);
}

double Ibm4AlignmentModel::unsmoothedNonheadDistortionProb(WordClassIndex trgWordClass, int dj)
{
  return exp(unsmoothedLogNonheadDistortionProb(trgWordClass, dj));
}

double Ibm4AlignmentModel::unsmoothedLogNonheadDistortionProb(WordClassIndex targetWordClass, int dj)
{
  bool found;
  double numer = nonheadDistortionTable->getNumerator(targetWordClass, dj, found);
  if (found)
  {
    double denom = nonheadDistortionTable->getDenominator(targetWordClass, found);
    if (found)
      return numer - denom;
  }
  return SMALL_LG_NUM;
}

Prob Ibm4AlignmentModel::calcProbOfAlignment(const vector<WordIndex>& nsrc, const vector<WordIndex>& trg,
                                             AlignmentInfo& alignment, int verbose)
{
  if (alignment.getProb() >= 0.0)
    return alignment.getProb();

  PositionIndex slen = (PositionIndex)nsrc.size() - 1;
  PositionIndex tlen = (PositionIndex)trg.size();

  if (verbose)
    cerr << "Obtaining IBM Model 4 prob..." << endl;

  Prob p0 = Prob(1.0) - p1;

  PositionIndex phi0 = alignment.getFertility(0);
  Prob prob = pow(p0, double(tlen - 2 * phi0)) * pow(p1, double(phi0));

  for (PositionIndex phi = 1; phi <= phi0; ++phi)
    prob *= double(tlen - phi0 - phi + 1) / phi;

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

    prob *= pts(s, t);
    if (i > 0)
    {
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
  alignment.setProb(prob);
  return prob;
}

bool Ibm4AlignmentModel::load(const char* prefFileName, int verbose)
{
  // Load IBM 3 Model data
  bool retVal = Ibm3AlignmentModel::load(prefFileName, verbose);
  if (retVal == THOT_ERROR)
    return retVal;

  if (verbose)
    cerr << "Loading IBM 4 Model data..." << endl;

  // Load file with source word classes
  string srcWordClassesFile = prefFileName;
  srcWordClassesFile = srcWordClassesFile + ".src_classes";
  retVal = wordClasses->loadSrcWordClasses(srcWordClassesFile.c_str(), verbose);
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  // Load file with target word classes
  string trgWordClassesFile = prefFileName;
  trgWordClassesFile = trgWordClassesFile + ".trg_classes";
  retVal = wordClasses->loadTrgWordClasses(trgWordClassesFile.c_str(), verbose);
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  // Load file with head distortion nd values
  string headDistortionNumDenFile = prefFileName;
  headDistortionNumDenFile = headDistortionNumDenFile + ".h_distnd";
  retVal = headDistortionTable->load(headDistortionNumDenFile.c_str(), verbose);
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  // Load file with nonhead distortion nd values
  string nonheadDistortionNumDenFile = prefFileName;
  nonheadDistortionNumDenFile = headDistortionNumDenFile + ".nh_distnd";
  retVal = nonheadDistortionTable->load(nonheadDistortionNumDenFile.c_str(), verbose);
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  // Print file with with alignment smoothing interpolation factor
  string dsifFile = prefFileName;
  dsifFile = dsifFile + ".dsifactor";
  return printDistortionSmoothFactor(dsifFile.c_str(), verbose);
}

bool Ibm4AlignmentModel::print(const char* prefFileName, int verbose)
{
  // Print IBM 3 Model data
  bool retVal = Ibm3AlignmentModel::print(prefFileName);
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  // Print file with source word classes
  string srcWordClassesFile = prefFileName;
  srcWordClassesFile = srcWordClassesFile + ".src_classes";
  retVal = wordClasses->printSrcWordClasses(srcWordClassesFile.c_str(), verbose);
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  // Print file with target word classes
  string trgWordClassesFile = prefFileName;
  trgWordClassesFile = trgWordClassesFile + ".trg_classes";
  retVal = wordClasses->printTrgWordClasses(trgWordClassesFile.c_str(), verbose);
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  // Print file with head distortion nd values
  string headDistortionNumDenFile = prefFileName;
  headDistortionNumDenFile = headDistortionNumDenFile + ".h_distnd";
  retVal = headDistortionTable->print(headDistortionNumDenFile.c_str());
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  // Print file with nonhead distortion nd values
  string nonheadDistortionNumDenFile = prefFileName;
  nonheadDistortionNumDenFile = headDistortionNumDenFile + ".nh_distnd";
  retVal = nonheadDistortionTable->print(nonheadDistortionNumDenFile.c_str());
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  // Load file with with distortion smoothing interpolation factor
  string dsifFile = prefFileName;
  dsifFile = dsifFile + ".dsifactor";
  return loadDistortionSmoothFactor(dsifFile.c_str(), verbose);
}

double Ibm4AlignmentModel::swapScore(const vector<WordIndex>& nsrc, const vector<WordIndex>& trg, PositionIndex j1,
                                     PositionIndex j2, AlignmentInfo& alignment)
{
  PositionIndex i1 = alignment.get(j1);
  PositionIndex i2 = alignment.get(j2);
  if (i1 == i2)
    return 1.0;

  WordIndex s1 = nsrc[i1];
  WordIndex s2 = nsrc[i2];
  WordIndex t1 = trg[j1 - 1];
  WordIndex t2 = trg[j2 - 1];

  Prob score = (pts(s2, t1) / pts(s1, t1)) * (pts(s1, t2) / pts(s2, t2));

  Prob oldProb = calcProbOfAlignment(nsrc, trg, alignment);

  alignment.set(j1, i2);
  alignment.set(j2, i1);
  Prob newProb = calcProbOfAlignment(nsrc, trg, alignment);
  alignment.set(j1, i1);
  alignment.set(j2, i2);
  alignment.setProb(oldProb);

  score *= newProb / oldProb;

  return score;
}

double Ibm4AlignmentModel::moveScore(const vector<WordIndex>& nsrc, const vector<WordIndex>& trg, PositionIndex iNew,
                                     PositionIndex j, AlignmentInfo& alignment)
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
  Prob p0 = Prob(1.0) - p1;
  Prob score;
  if (iOld == 0)
  {
    score = (p0 * p0 / p1) * ((phi0 * (tlen - phi0 + 1.0)) / ((tlen - 2 * phi0 + 1) * (tlen - 2 * phi0 + 2.0)))
          * (fertilityProb(sNew, phiNew + 1) / fertilityProb(sNew, phiNew)) * (pts(sNew, t) / pts(sOld, t));
  }
  else if (iNew == 0)
  {
    score = (p1 / (p0 * p0)) * (double((tlen - 2 * phi0) * (tlen - 2 * phi0 - 1)) / ((1 + phi0) * (tlen - phi0)))
          * (fertilityProb(sOld, phiOld - 1) / fertilityProb(sOld, phiOld)) * (pts(sNew, t) / pts(sOld, t));
  }
  else
  {
    score = (fertilityProb(sOld, phiOld - 1) / fertilityProb(sOld, phiOld))
          * (fertilityProb(sNew, phiNew + 1) / fertilityProb(sNew, phiNew)) * (pts(sNew, t) / pts(sOld, t));
  }

  Prob oldProb = calcProbOfAlignment(nsrc, trg, alignment);

  alignment.set(j, iNew);
  Prob newProb = calcProbOfAlignment(nsrc, trg, alignment);
  alignment.set(j, iOld);
  alignment.setProb(oldProb);

  score *= newProb / oldProb;

  return score;
}

void Ibm4AlignmentModel::clear()
{
  Ibm3AlignmentModel::clear();
  headDistortionTable->clear();
  nonheadDistortionTable->clear();
  wordClasses->clear();
  distortionSmoothFactor = 0.2;
}

void Ibm4AlignmentModel::clearTempVars()
{
  Ibm3AlignmentModel::clearTempVars();
  headDistortionCounts.clear();
  nonheadDistortionCounts.clear();
}
