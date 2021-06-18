#include "sw_models/Ibm4AligModel.h"

#include "nlp_common/MathFuncs.h"

using namespace std;

void Ibm4AligModel::initialBatchPass(pair<unsigned int, unsigned int> sentPairRange)
{
  Ibm3AligModel::initialBatchPass(sentPairRange);
  nonheadDistortionCounts.resize(trgWordClasses.size());
  nonheadDistortionTable.reserveSpace(trgWordClasses.size() - 1);
}

void Ibm4AligModel::initWordPair(const Sentence& nsrc, const Sentence& trg, PositionIndex i, PositionIndex j)
{
  WordIndex s = nsrc[i];
  WordIndex t = trg[j - 1];
  WordClassIndex srcWordClass = srcWordClasses[s];
  WordClassIndex trgWordClass = trgWordClasses[t];
  headDistortionTable.reserveSpace(srcWordClass, trgWordClass);
}

void Ibm4AligModel::incrementTargetWordCounts(const Sentence& nsrc, const Sentence& trg, const AlignmentInfo& alignment,
                                              PositionIndex j, double count)
{
  PositionIndex i = alignment.get(j);
  if (i == 0)
    return;

  WordIndex t = trg[j - 1];
  WordClassIndex trgWordClass = trgWordClasses[t];
  if (alignment.isHead(j))
  {
    PositionIndex prevCept = alignment.getPrevCept(i);
    WordIndex sPrev = nsrc[prevCept];
    WordClassIndex srcWordClass = srcWordClasses[sPrev];
    HeadDistortionTableKey key{srcWordClass, trgWordClass};
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
    const pair<HeadDistortionTableKey, HeadDistortionCountsElem>& p = headDistortionCounts.getAt(index);
    const HeadDistortionTableKey& key = p.first;
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

Prob Ibm4AligModel::headDistortionProb(WordClassIndex srcWordClass, WordClassIndex trgWordClass, int dj)
{
  return unsmoothedHeadDistortionProb(srcWordClass, trgWordClass, dj);
}

LgProb Ibm4AligModel::logHeadDistortionProb(WordClassIndex srcWordClass, WordClassIndex trgWordClass, int dj)
{
  return unsmoothedLogHeadDistortionProb(srcWordClass, trgWordClass, dj);
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

double Ibm4AligModel::headDistortionProb(WordClassIndex srcWordClass, WordClassIndex trgWordClass, int dj,
                                         bool training)
{
  if (training)
  {
    double logProb = unsmoothedLogHeadDistortionProb(srcWordClass, trgWordClass, dj);
    if (logProb != SMALL_LG_NUM)
      return exp(logProb);
    return 1.0 / (2 * (IBM4_SWM_MAX_SENT_LENGTH - 1));
  }
  return headDistortionProb(srcWordClass, trgWordClass, dj);
}

Prob Ibm4AligModel::nonheadDistortionProb(WordClassIndex trgWordClass, int dj)
{
  return unsmoothedNonheadDistortionProb(trgWordClass, dj);
}

LgProb Ibm4AligModel::logNonheadDistortionProb(WordClassIndex trgWordClass, int dj)
{
  return unsmoothedLogNonheadDistortionProb(trgWordClass, dj);
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

double Ibm4AligModel::nonheadDistortionProb(WordClassIndex trgWordClass, int dj, bool training)
{
  if (training)
  {
    double logProb = unsmoothedLogNonheadDistortionProb(trgWordClass, dj);
    if (logProb != SMALL_LG_NUM)
      return exp(logProb);
    return 1.0 / (2 * (IBM4_SWM_MAX_SENT_LENGTH - 1));
  }
  return nonheadDistortionProb(trgWordClass, dj);
}

Prob Ibm4AligModel::calcProbOfAlignment(const vector<WordIndex>& nsrc, const vector<WordIndex>& trg, bool training,
                                        AlignmentInfo& alignment, int verbose)
{
  if (alignment.getProb() >= 0.0)
    return alignment.getProb();

  PositionIndex slen = (PositionIndex)nsrc.size() - 1;
  PositionIndex tlen = (PositionIndex)trg.size();

  if (verbose)
    cerr << "Obtaining IBM Model 4 prob...\n";

  Prob p0 = Prob(1.0) - p1;

  PositionIndex phi0 = alignment.getFertility(0);
  Prob prob = pow(p0, double(tlen - 2 * phi0)) * pow(p1, double(phi0));

  for (PositionIndex phi = 1; phi <= phi0; ++phi)
    prob *= double(tlen - phi0 - phi + 1) / phi;

  for (PositionIndex i = 1; i <= slen; ++i)
  {
    WordIndex s = nsrc[i];
    PositionIndex phi = alignment.getFertility(i);
    prob *= fertilityProb(s, phi, training);
  }

  for (PositionIndex j = 1; j <= tlen; ++j)
  {
    PositionIndex i = alignment.get(j);
    WordIndex s = nsrc[i];
    WordIndex t = trg[j - 1];

    prob *= pts(s, t, training);
    if (i > 0)
    {
      WordClassIndex trgWordClass = trgWordClasses[t];
      if (alignment.isHead(j))
      {
        PositionIndex prevCept = alignment.getPrevCept(i);
        WordIndex sPrev = nsrc[prevCept];
        WordClassIndex srcWordClass = srcWordClasses[sPrev];
        PositionIndex dj = j - alignment.getCenter(prevCept);
        prob *= headDistortionProb(srcWordClass, trgWordClass, dj, training);
      }
      else
      {
        PositionIndex prevInCept = alignment.getPrevInCept(j);
        PositionIndex dj = j - prevInCept;
        prob *= nonheadDistortionProb(trgWordClass, dj, training);
      }
    }
  }
  alignment.setProb(prob);
  return prob;
}

LgProb Ibm4AligModel::calcLgProb(const vector<WordIndex>& src, const vector<WordIndex>& trg, int verbose)
{
  return 0.0;
}

bool Ibm4AligModel::load(const char* prefFileName, int verbose)
{
  // Load IBM 3 Model data
  bool retVal = Ibm3AligModel::load(prefFileName, verbose);
  if (retVal == THOT_ERROR)
    return retVal;

  if (verbose)
    cerr << "Loading IBM 4 Model data..." << endl;

  // Load file with distortion nd values
  string distortionNumDenFile = prefFileName;
  distortionNumDenFile = distortionNumDenFile + ".distnd";
  retVal = distortionTable.load(distortionNumDenFile.c_str(), verbose);
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  // Load file with fertility nd values
  string fertilityNumDenFile = prefFileName;
  fertilityNumDenFile = distortionNumDenFile + ".fertnd";
  return fertilityTable.load(fertilityNumDenFile.c_str(), verbose);
}

bool Ibm4AligModel::print(const char* prefFileName, int verbose)
{
  // Print IBM 3 Model data
  bool retVal = Ibm3AligModel::print(prefFileName);
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  // Print file with distortion nd values
  string distortionNumDenFile = prefFileName;
  distortionNumDenFile = distortionNumDenFile + ".distnd";
  retVal = distortionTable.print(distortionNumDenFile.c_str());
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  // Load file with fertility nd values
  string fertilityNumDenFile = prefFileName;
  fertilityNumDenFile = distortionNumDenFile + ".fertnd";
  return fertilityTable.print(fertilityNumDenFile.c_str());
}

double Ibm4AligModel::swapScore(const Sentence& nsrc, const Sentence& trg, PositionIndex j1, PositionIndex j2,
                                bool training, AlignmentInfo& alignment)
{
  PositionIndex i1 = alignment.get(j1);
  PositionIndex i2 = alignment.get(j2);
  if (i1 == i2)
    return 1.0;

  WordIndex s1 = nsrc[i1];
  WordIndex s2 = nsrc[i2];
  WordIndex t1 = trg[j1 - 1];
  WordIndex t2 = trg[j2 - 1];

  Prob score = (pts(s2, t1, training) / pts(s1, t1, training)) * (pts(s1, t2, training) / pts(s2, t2, training));

  Prob oldProb = calcProbOfAlignment(nsrc, trg, training, alignment);

  alignment.set(j1, i2);
  alignment.set(j2, i1);
  Prob newProb = calcProbOfAlignment(nsrc, trg, training, alignment);
  alignment.set(j1, i1);
  alignment.set(j2, i2);
  alignment.setProb(oldProb);

  score *= newProb / oldProb;

  return score;
}

double Ibm4AligModel::moveScore(const Sentence& nsrc, const Sentence& trg, PositionIndex iNew, PositionIndex j,
                                bool training, AlignmentInfo& alignment)
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
          * (fertilityProb(sNew, phiNew + 1, training) / fertilityProb(sNew, phiNew, training))
          * (pts(sNew, t, training) / pts(sOld, t, training));
  }
  else if (iNew == 0)
  {
    score = (p1 / (p0 * p0)) * (double((tlen - 2 * phi0) * (tlen - 2 * phi0 - 1)) / ((1 + phi0) * (tlen - phi0)))
          * (fertilityProb(sOld, phiOld - 1, training) / fertilityProb(sOld, phiOld, training))
          * (pts(sNew, t, training) / pts(sOld, t, training));
  }
  else
  {
    score = (fertilityProb(sOld, phiOld - 1, training) / fertilityProb(sOld, phiOld, training))
          * (fertilityProb(sNew, phiNew + 1, training) / fertilityProb(sNew, phiNew, training))
          * (pts(sNew, t, training) / pts(sOld, t, training));
  }

  Prob oldProb = calcProbOfAlignment(nsrc, trg, training, alignment);

  alignment.set(j, iNew);
  Prob newProb = calcProbOfAlignment(nsrc, trg, training, alignment);
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

Ibm4AligModel::~Ibm4AligModel()
{
}
