#include "sw_models/Ibm4AligModel.h"

#include "nlp_common/Exceptions.h"
#include "nlp_common/MathFuncs.h"

using namespace std;

Ibm4AligModel::Ibm4AligModel() : distortionSmoothFactor{0.2}
{
}

void Ibm4AligModel::initialBatchPass(pair<unsigned int, unsigned int> sentPairRange)
{
  Ibm3AligModel::initialBatchPass(sentPairRange);
  nonheadDistortionCounts.resize(wordClasses.getTrgWordClassCount());
  nonheadDistortionTable.reserveSpace(wordClasses.getTrgWordClassCount() - 1);
}

void Ibm4AligModel::initWordPair(const Sentence& nsrc, const Sentence& trg, PositionIndex i, PositionIndex j)
{
  WordIndex s = nsrc[i];
  WordIndex t = trg[j - 1];
  WordClassIndex srcWordClass = wordClasses.getSrcWordClass(s);
  WordClassIndex trgWordClass = wordClasses.getTrgWordClass(t);
  headDistortionTable.reserveSpace(srcWordClass, trgWordClass);
}

void Ibm4AligModel::incrementTargetWordCounts(const Sentence& nsrc, const Sentence& trg, const AlignmentInfo& alignment,
                                              PositionIndex j, double count)
{
  PositionIndex i = alignment.get(j);
  if (i == 0)
    return;

  WordIndex t = trg[j - 1];
  WordClassIndex trgWordClass = wordClasses.getTrgWordClass(t);
  if (alignment.isHead(j))
  {
    PositionIndex prevCept = alignment.getPrevCept(i);
    WordIndex sPrev = nsrc[prevCept];
    WordClassIndex srcWordClass = wordClasses.getSrcWordClass(sPrev);
    HeadDistortionKey key{srcWordClass, trgWordClass};
    PositionIndex dj = j - alignment.getCenter(prevCept);

#pragma omp critical(headDistortionCounts)
    headDistortionCounts[key][dj] += count;
  }
  else
  {
    PositionIndex prevInCept = alignment.getPrevInCept(j);
    PositionIndex dj = j - prevInCept;

#pragma omp critical(nonheadDistortionCounts)
    nonheadDistortionCounts[trgWordClass][dj] += count;
  }
}

void Ibm4AligModel::batchMaximizeProbs()
{
  Ibm3AligModel::batchMaximizeProbs();

#pragma omp parallel for schedule(dynamic)
  for (int index = 0; index < (int)headDistortionCounts.size(); ++index)
  {
    double denom = 0;
    const pair<HeadDistortionKey, HeadDistortionCountsElem>& p = headDistortionCounts.getAt(index);
    const HeadDistortionKey& key = p.first;
    HeadDistortionCountsElem& elem = const_cast<HeadDistortionCountsElem&>(p.second);
    for (PositionIndex dj = 1; dj <= elem.size(); ++dj)
    {
      double numer = elem[dj - 1];
      if (numer == 0)
        continue;
      denom += numer;
      float logNumer = (float)log(numer);
      headDistortionTable.setNumerator(key.srcWordClass, key.trgWordClass, dj, logNumer);
      elem[dj - 1] = 0.0;
    }
    if (denom == 0)
      denom = 1;
    float logDenom = (float)log(denom);
    headDistortionTable.setDenominator(key.srcWordClass, key.trgWordClass, logDenom);
  }

#pragma omp parallel for schedule(dynamic)
  for (int targetWordClass = 0; targetWordClass < (int)nonheadDistortionCounts.size(); ++targetWordClass)
  {
    double denom = 0;
    NonheadDistortionCountsElem& elem = nonheadDistortionCounts[targetWordClass];
    for (PositionIndex dj = 1; dj <= elem.size(); ++dj)
    {
      double numer = elem[dj - 1];
      if (numer == 0)
        continue;
      denom += numer;
      float logNumer = (float)log(numer);
      nonheadDistortionTable.setNumerator(targetWordClass, dj, logNumer);
      elem[dj - 1] = 0.0;
    }
    if (denom == 0)
      denom = 1;
    float logDenom = (float)log(denom);
    nonheadDistortionTable.setDenominator(targetWordClass, logDenom);
  }
}

bool Ibm4AligModel::loadDistortionSmoothFactor(const char* distortionSmoothFactorFile, int verbose)
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

bool Ibm4AligModel::printDistortionSmoothFactor(const char* distortionSmoothFactorFile, int verbose)
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

Prob Ibm4AligModel::headDistortionProb(WordClassIndex srcWordClass, WordClassIndex trgWordClass, PositionIndex tlen,
                                       int dj)
{
  double logProb = unsmoothedLogHeadDistortionProb(srcWordClass, trgWordClass, dj);
  double prob = exp(logProb);
  prob = (distortionSmoothFactor / (tlen - 1)) + ((1.0 - distortionSmoothFactor) * prob);
  return max(prob, SW_PROB_SMOOTH);
}

LgProb Ibm4AligModel::logHeadDistortionProb(WordClassIndex srcWordClass, WordClassIndex trgWordClass,
                                            PositionIndex tlen, int dj)
{
  double logProb = unsmoothedLogHeadDistortionProb(srcWordClass, trgWordClass, dj);
  logProb =
      MathFuncs::lns_sumlog(log(distortionSmoothFactor / (tlen - 1)), (log(1.0 - distortionSmoothFactor) + logProb));
  return max(logProb, SW_LOG_PROB_SMOOTH);
}

LgProb Ibm4AligModel::calcLgProb(const vector<WordIndex>& src, const vector<WordIndex>& trg, int verbose)
{
  throw NotImplemented();
}

void Ibm4AligModel::setDistortionSmoothFactor(double distortionSmoothFactor, int verbose)
{
  this->distortionSmoothFactor = distortionSmoothFactor;
  if (verbose)
    cerr << "Distortion smoothing interpolation factor has been set to " << distortionSmoothFactor << endl;
}

void Ibm4AligModel::addSrcWordClass(WordIndex s, WordClassIndex c)
{
  wordClasses.addSrcWordClass(s, c);
}

void Ibm4AligModel::addTrgWordClass(WordIndex t, WordClassIndex c)
{
  wordClasses.addTrgWordClass(t, c);
}

bool Ibm4AligModel::sentenceLengthIsOk(const vector<WordIndex> sentence)
{
  return !sentence.empty() && sentence.size() <= IBM4_SWM_MAX_SENT_LENGTH;
}

double Ibm4AligModel::unsmoothedHeadDistortionProb(WordClassIndex srcWordClass, WordClassIndex trgWordClass, int dj)
{
  return exp(unsmoothedLogHeadDistortionProb(srcWordClass, trgWordClass, dj));
}

double Ibm4AligModel::unsmoothedLogHeadDistortionProb(WordClassIndex srcWordClass, WordClassIndex trgWordClass, int dj)
{
  bool found;
  double numer = headDistortionTable.getNumerator(srcWordClass, trgWordClass, dj, found);
  if (found)
  {
    double denom = headDistortionTable.getDenominator(srcWordClass, trgWordClass, found);
    if (found)
      return numer - denom;
  }
  return SMALL_LG_NUM;
}

Prob Ibm4AligModel::nonheadDistortionProb(WordClassIndex trgWordClass, PositionIndex tlen, int dj)
{
  double logProb = unsmoothedLogNonheadDistortionProb(trgWordClass, dj);
  double prob = exp(logProb);
  prob = (distortionSmoothFactor / (tlen - 1)) + ((1.0 - distortionSmoothFactor) * prob);
  return max(prob, SW_PROB_SMOOTH);
}

LgProb Ibm4AligModel::logNonheadDistortionProb(WordClassIndex trgWordClass, PositionIndex tlen, int dj)
{
  double logProb = unsmoothedLogNonheadDistortionProb(trgWordClass, dj);
  logProb =
      MathFuncs::lns_sumlog(log(distortionSmoothFactor / (tlen - 1)), (log(1.0 - distortionSmoothFactor) + logProb));
  return max(logProb, SW_LOG_PROB_SMOOTH);
}

double Ibm4AligModel::unsmoothedNonheadDistortionProb(WordClassIndex trgWordClass, int dj)
{
  return exp(unsmoothedLogNonheadDistortionProb(trgWordClass, dj));
}

double Ibm4AligModel::unsmoothedLogNonheadDistortionProb(WordClassIndex targetWordClass, int dj)
{
  bool found;
  double numer = nonheadDistortionTable.getNumerator(targetWordClass, dj, found);
  if (found)
  {
    double denom = nonheadDistortionTable.getDenominator(targetWordClass, found);
    if (found)
      return numer - denom;
  }
  return SMALL_LG_NUM;
}

Prob Ibm4AligModel::calcProbOfAlignment(const vector<WordIndex>& nsrc, const vector<WordIndex>& trg,
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
      WordClassIndex trgWordClass = wordClasses.getTrgWordClass(t);
      if (alignment.isHead(j))
      {
        PositionIndex prevCept = alignment.getPrevCept(i);
        WordIndex sPrev = nsrc[prevCept];
        WordClassIndex srcWordClass = wordClasses.getSrcWordClass(sPrev);
        PositionIndex dj = j - alignment.getCenter(prevCept);
        prob *= headDistortionProb(srcWordClass, trgWordClass, tlen, dj);
      }
      else
      {
        PositionIndex prevInCept = alignment.getPrevInCept(j);
        PositionIndex dj = j - prevInCept;
        prob *= nonheadDistortionProb(trgWordClass, tlen, dj);
      }
    }
  }
  alignment.setProb(prob);
  return prob;
}

bool Ibm4AligModel::load(const char* prefFileName, int verbose)
{
  // Load IBM 3 Model data
  bool retVal = Ibm3AligModel::load(prefFileName, verbose);
  if (retVal == THOT_ERROR)
    return retVal;

  if (verbose)
    cerr << "Loading IBM 4 Model data..." << endl;

  // Load file with source word classes
  string srcWordClassesFile = prefFileName;
  srcWordClassesFile = srcWordClassesFile + ".src_classes";
  retVal = wordClasses.loadSrcWordClasses(srcWordClassesFile.c_str(), verbose);
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  // Load file with target word classes
  string trgWordClassesFile = prefFileName;
  trgWordClassesFile = trgWordClassesFile + ".trg_classes";
  retVal = wordClasses.loadTrgWordClasses(trgWordClassesFile.c_str(), verbose);
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  // Load file with head distortion nd values
  string headDistortionNumDenFile = prefFileName;
  headDistortionNumDenFile = headDistortionNumDenFile + ".h_distnd";
  retVal = headDistortionTable.load(headDistortionNumDenFile.c_str(), verbose);
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  // Load file with nonhead distortion nd values
  string nonheadDistortionNumDenFile = prefFileName;
  nonheadDistortionNumDenFile = headDistortionNumDenFile + ".nh_distnd";
  retVal = nonheadDistortionTable.load(nonheadDistortionNumDenFile.c_str(), verbose);
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  // Print file with with alignment smoothing interpolation factor
  string dsifFile = prefFileName;
  dsifFile = dsifFile + ".dsifactor";
  return printDistortionSmoothFactor(dsifFile.c_str(), verbose);
}

bool Ibm4AligModel::print(const char* prefFileName, int verbose)
{
  // Print IBM 3 Model data
  bool retVal = Ibm3AligModel::print(prefFileName);
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  // Print file with source word classes
  string srcWordClassesFile = prefFileName;
  srcWordClassesFile = srcWordClassesFile + ".src_classes";
  retVal = wordClasses.printSrcWordClasses(srcWordClassesFile.c_str(), verbose);
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  // Print file with target word classes
  string trgWordClassesFile = prefFileName;
  trgWordClassesFile = trgWordClassesFile + ".trg_classes";
  retVal = wordClasses.printTrgWordClasses(trgWordClassesFile.c_str(), verbose);
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  // Print file with head distortion nd values
  string headDistortionNumDenFile = prefFileName;
  headDistortionNumDenFile = headDistortionNumDenFile + ".h_distnd";
  retVal = headDistortionTable.print(headDistortionNumDenFile.c_str());
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  // Print file with nonhead distortion nd values
  string nonheadDistortionNumDenFile = prefFileName;
  nonheadDistortionNumDenFile = headDistortionNumDenFile + ".nh_distnd";
  retVal = nonheadDistortionTable.print(nonheadDistortionNumDenFile.c_str());
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  // Load file with with distortion smoothing interpolation factor
  string dsifFile = prefFileName;
  dsifFile = dsifFile + ".dsifactor";
  return loadDistortionSmoothFactor(dsifFile.c_str(), verbose);
}

double Ibm4AligModel::swapScore(const Sentence& nsrc, const Sentence& trg, PositionIndex j1, PositionIndex j2,
                                AlignmentInfo& alignment)
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

double Ibm4AligModel::moveScore(const Sentence& nsrc, const Sentence& trg, PositionIndex iNew, PositionIndex j,
                                AlignmentInfo& alignment)
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

void Ibm4AligModel::clear()
{
  Ibm3AligModel::clear();
  headDistortionTable.clear();
  nonheadDistortionTable.clear();
  wordClasses.clear();
  distortionSmoothFactor = 0.2;
}

void Ibm4AligModel::clearInfoAboutSentRange()
{
  Ibm3AligModel::clearInfoAboutSentRange();
  headDistortionCounts.clear();
  nonheadDistortionCounts.clear();
}

void Ibm4AligModel::clearTempVars()
{
  Ibm3AligModel::clearTempVars();
  headDistortionCounts.clear();
  nonheadDistortionCounts.clear();
}
