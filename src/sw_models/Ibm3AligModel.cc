/*
thot package for statistical machine translation
Copyright (C) 2013 Daniel Ortiz-Mart\'inez

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public License
as published by the Free Software Foundation; either version 3
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program; If not, see <http://www.gnu.org/licenses/>.
*/

/**
 * @file Ibm3AligModel.cc
 *
 * @brief Definitions file for Ibm3AligModel.h
 */

#include "Ibm3AligModel.h"
#include "MathFuncs.h"

using namespace std;


Ibm3AligModel::Ibm3AligModel() : IncrIbm2AligModel(), p0Count(0), p1Count(0), p1(0.5)
{
}

void Ibm3AligModel::initSourceWord(const Sentence& nsrc, const Sentence& trg, PositionIndex i)
{
  IncrIbm2AligModel::initSourceWord(nsrc, trg, i);

  dSource ds;
  ds.i = i;
  ds.slen = (PositionIndex)nsrc.size() - 1;
  ds.tlen = (PositionIndex)trg.size();
  distortionTable.setDistortionDenom(ds, (float)log(trg.size()));

  DistortionCountsEntry& distortionEntry = distortionCounts[ds];
  if (distortionEntry.size() < trg.size())
    distortionEntry.resize(trg.size(), 0);
}

void Ibm3AligModel::initWordPair(const Sentence& nsrc, const Sentence& trg, PositionIndex i, PositionIndex j)
{
  IncrIbm2AligModel::initWordPair(nsrc, trg, i, j);

  dSource ds;
  ds.i = i;
  ds.slen = (PositionIndex)nsrc.size() - 1;
  ds.tlen = (PositionIndex)trg.size();
  distortionTable.setDistortionNumer(ds, j, 0);
}

void Ibm3AligModel::addTranslationOptions(vector<vector<WordIndex>>& insertBuffer) {
  WordIndex maxSrcWordIndex = (WordIndex)insertBuffer.size() - 1;

  if (maxSrcWordIndex >= lexCounts.size())
    lexCounts.resize((size_t)maxSrcWordIndex + 1);
  lexTable.reserveSpace(maxSrcWordIndex);

  if (maxSrcWordIndex >= fertilityCounts.size())
    fertilityCounts.resize((size_t)maxSrcWordIndex + 1);
  fertilityTable.reserveSpace(maxSrcWordIndex, MaxFertility);

  #pragma omp parallel for schedule(dynamic)
  for (int s = 0; s < (int)insertBuffer.size(); ++s)
  {
    for (WordIndex t : insertBuffer[s])
    {
      lexCounts[s][t] = 0;
      lexTable.setLexNumer(s, t, 0);
    }

    FertilityCountsEntry& fertilityEntry = fertilityCounts[s];
    fertilityEntry.resize(MaxFertility, 0);
    fertilityTable.setFertilityNumer(s, 0, (float)log(0.2));
    fertilityTable.setFertilityNumer(s, 1, (float)log(0.65));
    fertilityTable.setFertilityNumer(s, 2, (float)log(0.1));
    fertilityTable.setFertilityNumer(s, 2, (float)log(0.04));
    const float initialProb = (float)log(0.01 / (MaxFertility - 4));
    for (PositionIndex phi = 4; phi < MaxFertility; phi++)
      fertilityTable.setFertilityNumer(s, phi, initialProb);

    insertBuffer[s].clear();
  }
}

void Ibm3AligModel::batchUpdateCounts(const SentPairCont& pairs)
{
  #pragma omp parallel for schedule(dynamic)
  for (int line_idx = 0; line_idx < (int)pairs.size(); ++line_idx)
  {
    Sentence src = pairs[line_idx].first;
    Sentence nsrc = extendWithNullWord(src);
    Sentence trg = pairs[line_idx].second;

    PositionIndex slen = (PositionIndex)nsrc.size() - 1;
    PositionIndex tlen = (PositionIndex)trg.size();

    vector<PositionIndex> alig, fertility;
    Matrix<double> moveScores, swapScores;
    Prob aligProb = lexAligM3ProbForBestAlig(nsrc, trg, alig, fertility, moveScores, swapScores);
    Matrix<double> moveCounts(slen + 1, tlen + 1), swapCounts(slen + 1, tlen + 1);
    vector<double> negMove(tlen + 1), negSwap(tlen + 1), plus1Fert(slen + 1), minus1Fert(slen + 1);
    double totalMove = aligProb;
    double totalSwap = 0;

    for (PositionIndex j = 1; j <= trg.size(); j++)
    {
      for (PositionIndex i = 0; i < nsrc.size(); i++)
      {
        if (alig[j - 1] != i)
        {
          double prob = aligProb * moveScores(i, j);
          if (prob < SmoothingAnjiNum) prob = SmoothingAnjiNum;
          totalMove += prob;
          moveCounts(i, j) += prob;
          negMove[j] += prob;
          plus1Fert[i] += prob;
          minus1Fert[alig[j - 1]] += prob;
        }
      }

      for (PositionIndex j1 = j + 1; j1 <= trg.size(); j1++)
      {
        if (alig[j - 1] != alig[j1 - 1])
        {
          double prob = aligProb * swapScores(j, j1);
          if (prob < SmoothingAnjiNum) prob = SmoothingAnjiNum;
          totalSwap += prob;
          swapCounts(alig[j - 1], j1) += prob;
          swapCounts(alig[j1 - 1], j) += prob;
          negSwap[j] += prob;
          negSwap[j1] += prob;
        }
      }
    }

    double totalCount = totalMove + totalSwap;
    Matrix<double> fertCounts(slen + 1, MaxFertility + 1);
    for (PositionIndex i = 0; i < nsrc.size(); i++)
    {
      for (PositionIndex j = 1; j <= trg.size(); j++)
      {
        double count = i == alig[j - 1] ? totalCount - (negMove[j] + negSwap[j]) : moveCounts(i, j) + swapCounts(i, j);
        count /= totalCount;
        incrementWordPairCounts(nsrc, trg, i, j, count);
      }

      if (i > 0)
      {
        double temp = minus1Fert[i] + plus1Fert[i];
        PositionIndex phi = fertility[i];
        if (phi < MaxFertility)
          fertCounts(i, phi) += totalCount - temp;
        if (phi > 0 && phi - 1 < MaxFertility)
          fertCounts(i, phi - 1) += minus1Fert[i];
        if (phi + 1 < MaxFertility)
          fertCounts(i, phi + 1) += plus1Fert[i];
      }
    }

    for (PositionIndex i = 1; i < nsrc.size(); i++)
    {
      WordIndex s = nsrc[i];
      for (PositionIndex phi = 0; phi < MaxFertility; phi++)
      {
        double count = fertCounts(i, phi) / totalCount;

        #pragma omp atomic
        fertilityCounts[s][phi] += count;
      }
    }

    PositionIndex phi0 = fertility[0];
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
      p1c += plus1Fert[0] * (phi0 + 1);
      p0c += plus1Fert[0] * (tlen - 2 * (phi0 + 1));
    }

    #pragma omp atomic
    p1Count += p1c / totalCount;
    #pragma omp atomic
    p0Count += p0c / totalCount;
  }
}

void Ibm3AligModel::incrementWordPairCounts(const Sentence& nsrc, const Sentence& trg, PositionIndex i, PositionIndex j,
  double count)
{
  IncrIbm2AligModel::incrementWordPairCounts(nsrc, trg, i, j, count);

  dSource ds;
  ds.i = i;
  ds.slen = (PositionIndex)nsrc.size() - 1;
  ds.tlen = (PositionIndex)trg.size();

  #pragma omp atomic
  distortionCounts[ds][j] += count;
}

void Ibm3AligModel::batchMaximizeProbs()
{
  IncrIbm2AligModel::batchMaximizeProbs();

  #pragma omp parallel for schedule(dynamic)
  for (int asIndex = 0; asIndex < (int)distortionCounts.size(); ++asIndex)
  {
    double denom = 0;
    const pair<dSource, DistortionCountsEntry>& p = distortionCounts.getAt(asIndex);
    const dSource& ds = p.first;
    DistortionCountsEntry& elem = const_cast<DistortionCountsEntry&>(p.second);
    for (PositionIndex j = 0; j < elem.size(); ++j)
    {
      double numer = elem[j];
      denom += numer;
      float logNumer = (float)log(numer);
      distortionTable.setDistortionNumer(ds, j, logNumer);
      elem[j] = 0.0;
    }
    if (denom == 0)
      denom = 1;
    float logDenom = (float)log(denom);
    distortionTable.setDistortionDenom(ds, logDenom);
  }

  #pragma omp parallel for schedule(dynamic)
  for (int s = 0; s < (int)fertilityCounts.size(); ++s)
  {
    double denom = 0;
    FertilityCountsEntry& elem = fertilityCounts[s];
    for (PositionIndex phi = 0; phi < elem.size(); phi++)
    {
      double numer = elem[phi];
      denom += numer;
      fertilityTable.setFertilityNumer(s, phi, (float)log(numer));
      elem[phi] = 0.0;
    }
    if (denom == 0)
      denom = 1;
    fertilityTable.setFertilityDenom(s, (float)log(denom));
  }

  p1 = p1Count / (p1Count + p0Count);
}

Prob Ibm3AligModel::distortionProb(PositionIndex i, PositionIndex slen, PositionIndex tlen, PositionIndex j)
{
  return unsmoothedDistortionProb(j, slen, tlen, i);
}

LgProb Ibm3AligModel::logDistortionProb(PositionIndex i, PositionIndex slen, PositionIndex tlen, PositionIndex j)
{
  return unsmoothedLogDistortionProb(i, slen, tlen, j);
}

double Ibm3AligModel::unsmoothedDistortionProb(PositionIndex i, PositionIndex slen, PositionIndex tlen, PositionIndex j)
{
  return exp(unsmoothedLogDistortionProb(i, slen, tlen, j));
}

double Ibm3AligModel::unsmoothedLogDistortionProb(PositionIndex i, PositionIndex slen, PositionIndex tlen,
  PositionIndex j)
{
  bool found;
  double numer;
  dSource ds;

  ds.i = i;
  ds.slen = slen;
  ds.tlen = tlen;

  numer = distortionTable.getDistortionNumer(ds, j, found);
  if (found)
  {
    // numerator for pair ds,j exists
    double denom;
    denom = distortionTable.getDistortionDenom(ds, found);
    if (!found) return SMALL_LG_NUM;
    else
    {
      return numer - denom;
    }
  }
  else
  {
    // numerator for pair ds,j does not exist
    return SMALL_LG_NUM;
  }
}

Prob Ibm3AligModel::fertilityProb(WordIndex s, PositionIndex phi)
{
  return unsmoothedLogFertilityProb(s, phi);
}

LgProb Ibm3AligModel::logFertilityProb(WordIndex s, PositionIndex phi)
{
  return unsmoothedFertilityProb(s, phi);
}

double Ibm3AligModel::unsmoothedFertilityProb(WordIndex s, PositionIndex phi)
{
  return exp(unsmoothedLogFertilityProb(s, phi));
}

double Ibm3AligModel::unsmoothedLogFertilityProb(WordIndex s, PositionIndex phi)
{
  if (phi >= MaxFertility)
    return SMALL_LG_NUM;
  bool found;
  double numer = fertilityTable.getFertilityNumer(s, phi, found);
  if (found)
  {
    // numerator for pair s,phi exists
    double denom = fertilityTable.getFertilityDenom(s, found);
    if (!found) return SMALL_LG_NUM;
    else
    {
      return numer - denom;
    }
  }
  else
  {
    // numerator for pair s,phi does not exist
    return SMALL_LG_NUM;
  }
}

LgProb Ibm3AligModel::obtainBestAlignment(const vector<WordIndex>& srcSentIndexVector,
  const vector<WordIndex>& trgSentIndexVector, WordAligMatrix& bestWaMatrix)
{
  vector<PositionIndex> bestAlig, bestFertility;
  Matrix<double> moveScores, swapScores;
  LgProb lgProb = sentLenLgProb((PositionIndex)srcSentIndexVector.size(), (PositionIndex)trgSentIndexVector.size());
  lgProb += lexAligM3ProbForBestAlig(addNullWordToWidxVec(srcSentIndexVector), trgSentIndexVector, bestAlig,
    bestFertility, moveScores, swapScores).get_lp();

  bestWaMatrix.init((PositionIndex)srcSentIndexVector.size(), (PositionIndex)trgSentIndexVector.size());
  bestWaMatrix.putAligVec(bestAlig);

  return lgProb;
}

LgProb Ibm3AligModel::calcLgProbForAlig(const vector<WordIndex>& sSent, const vector<WordIndex>& tSent,
  const WordAligMatrix& aligMatrix, int verbose)
{
  PositionIndex i;

  vector<PositionIndex> alig;
  aligMatrix.getAligVec(alig);

  if (verbose)
  {
    for (i = 0; i < sSent.size(); ++i) cerr << sSent[i] << " ";
    cerr << "\n";
    for (i = 0; i < tSent.size(); ++i) cerr << tSent[i] << " ";
    cerr << "\n";
    for (i = 0; i < alig.size(); ++i) cerr << alig[i] << " ";
    cerr << "\n";
  }
  if (tSent.size() != alig.size())
  {
    cerr << "Error: the sentence t and the alignment vector have not the same size." << endl;
    return THOT_ERROR;
  }
  else
  {
    vector<PositionIndex> fertility(sSent.size() + 1, 0);
    for (PositionIndex j = 0; j < alig.size(); j++)
      fertility[alig[j]]++;
    return calcIbm3ProbFromAlig(addNullWordToWidxVec(sSent), tSent, alig, fertility).get_lp();
  }
}

Prob Ibm3AligModel::calcIbm3ProbFromAlig(const vector<WordIndex>& nsSent, const vector<WordIndex>& tSent,
  const vector<PositionIndex>& alig, const vector<PositionIndex>& fertility, int verbose)
{
  PositionIndex slen = (PositionIndex)nsSent.size() - 1;
  PositionIndex tlen = (PositionIndex)tSent.size();

  if (verbose) cerr << "Obtaining IBM Model 3 logprob...\n";

  Prob p0 = Prob(1.0) - p1;

  PositionIndex phi0 = fertility[0];
  Prob prob = pow(p0, double(tlen - 2 * phi0)) * pow(p1, double(phi0));

  for (PositionIndex phi = 1; phi <= phi0; phi++)
    prob *= double(tlen - phi0 - phi + 1) / phi;

  for (PositionIndex i = 1; i <= slen; i++)
  {
    PositionIndex phi = fertility[i];
    prob *= Prob(MathFuncs::factorial(phi)) * fertilityProb(nsSent[i], phi);
  }

  for (PositionIndex j = 1; j <= alig.size(); ++j)
  {
    PositionIndex i = alig[j - 1];
    WordIndex s = nsSent[i];
    WordIndex t = tSent[j - 1];

    prob *= pts(s, t) * distortionProb(i, slen, tlen, j);
  }
  return prob;
}

LgProb Ibm3AligModel::calcLgProb(const vector<WordIndex>& sSent, const vector<WordIndex>& tSent, int verbose)
{
  return calcSumIbm3LgProb(addNullWordToWidxVec(sSent), tSent, verbose);
}

LgProb Ibm3AligModel::calcSumIbm3LgProb(const vector<WordIndex>& nsSent, const vector<WordIndex>& tSent, int verbose)
{
  PositionIndex slen = (PositionIndex)nsSent.size() - 1;
  PositionIndex tlen = (PositionIndex)tSent.size();

  if (verbose) cerr << "Obtaining Sum IBM Model 3 logprob...\n";

  Prob p0 = 1.0 - (double)p1;

  LgProb lgProb = sentLenLgProb(slen, tlen);
  LgProb fertilityContrib = 0;
  for (PositionIndex fertility = 0; fertility < min(tlen, MaxFertility); fertility++)
  {
    Prob sump = 0;
    Prob prob = 1.0;
    PositionIndex phi0 = fertility;
    prob *= pow(p1, double(phi0)) * pow(p0, double(tlen - 2 * phi0));

    for (PositionIndex phi = 1; phi <= phi0; phi++)
      prob *= double(tlen - phi0 - phi + 1) / phi;
    sump += prob;

    for (PositionIndex i = 1; i < nsSent.size(); i++)
    {
      PositionIndex phi = fertility;
      sump += Prob(MathFuncs::factorial(phi)) * fertilityProb(nsSent[i], phi);
    }
    fertilityContrib += sump.get_lp();
  }

  if (verbose) cerr << "- Fertility contribution= " << fertilityContrib << endl;
  lgProb += fertilityContrib;

  LgProb lexDistorionContrib = 0;
  for (PositionIndex j = 1; j <= tSent.size(); j++)
  {
    Prob sump = 0;
    for (PositionIndex i = 0; i < nsSent.size(); i++)
    {
      WordIndex s = nsSent[i];
      WordIndex t = tSent[j - 1];

      sump += pts(s, t) * distortionProb(i, slen, tlen, j);
    }
    lexDistorionContrib += sump.get_lp();
  }

  if (verbose) cerr << "- Lexical plus distortion contribution= " << lexDistorionContrib << endl;
  lgProb += lexDistorionContrib;

  return lgProb;
}

bool Ibm3AligModel::load(const char* prefFileName, int verbose)
{
  if (prefFileName[0] != 0)
  {
    bool retVal;

    // Load IBM 2 Model data
    retVal = IncrIbm2AligModel::load(prefFileName, verbose);
    if (retVal == THOT_ERROR) return THOT_ERROR;

    if (verbose)
      cerr << "Loading IBM 3 Model data..." << endl;

    // Load file with distortion nd values
    string distortionNumDenFile = prefFileName;
    distortionNumDenFile = distortionNumDenFile + ".distnd";
    retVal = distortionTable.load(distortionNumDenFile.c_str(), verbose);
    if (retVal == THOT_ERROR) return THOT_ERROR;

    // Load file with fertility nd values
    string fertilityNumDenFile = prefFileName;
    fertilityNumDenFile = distortionNumDenFile + ".fertnd";
    retVal = fertilityTable.load(fertilityNumDenFile.c_str(), verbose);
    if (retVal == THOT_ERROR) return THOT_ERROR;

    return THOT_OK;
  }
  else return THOT_ERROR;
}

bool Ibm3AligModel::print(const char* prefFileName, int verbose)
{
  bool retVal;

  // Print IBM 2 Model data
  retVal = IncrIbm2AligModel::print(prefFileName);
  if (retVal == THOT_ERROR) return THOT_ERROR;

  // Print file with distortion nd values
  string distortionNumDenFile = prefFileName;
  distortionNumDenFile = distortionNumDenFile + ".distnd";
  retVal = distortionTable.print(distortionNumDenFile.c_str());
  if (retVal == THOT_ERROR) return THOT_ERROR;

  // Load file with fertility nd values
  string fertilityNumDenFile = prefFileName;
  fertilityNumDenFile = distortionNumDenFile + ".fertnd";
  retVal = fertilityTable.print(fertilityNumDenFile.c_str());
  if (retVal == THOT_ERROR) return THOT_ERROR;

  return THOT_OK;
}

Prob Ibm3AligModel::lexAligM3ProbForBestAlig(const vector<WordIndex>& nSrcSentIndexVector,
  const vector<WordIndex>& trgSentIndexVector, vector<PositionIndex>& bestAlig, vector<PositionIndex>& bestFertility,
  Matrix<double>& moveScores, Matrix<double>& swapScores)
{
  // Initialize variables
  PositionIndex slen = (PositionIndex)nSrcSentIndexVector.size() - 1;
  PositionIndex tlen = (PositionIndex)trgSentIndexVector.size();

  // start with IBM-2 alignment
  lexAligM2LpForBestAlig(nSrcSentIndexVector, trgSentIndexVector, bestAlig, bestFertility);

  moveScores.resize(slen + 1, tlen + 1);
  swapScores.resize(tlen + 1, tlen + 1);

  // hillclimbing search
  int bestChangeType = -1;
  while (bestChangeType != 0)
  {
    bestChangeType = 0;
    PositionIndex bestChangeArg1;
    PositionIndex bestChangeArg2;
    double bestChangeScore = 1.00001;
    for (PositionIndex j = 1; j <= trgSentIndexVector.size(); j++)
    {
      PositionIndex iAlig = bestAlig[j - 1];

      // swap alignments
      for (PositionIndex j1 = j + 1; j1 <= trgSentIndexVector.size(); j1++)
      {
        if (iAlig != bestAlig[j1])
        {
          double changeScore = swapScore(nSrcSentIndexVector, trgSentIndexVector, bestAlig, j, j1);
          swapScores(j, j1) = changeScore;
          if (changeScore > bestChangeScore)
          {
            bestChangeScore = changeScore;
            bestChangeType = 1;
            bestChangeArg1 = j;
            bestChangeArg2 = j1;
          }
        }
        else
        {
          swapScores(j, j1) = 1.0;
        }
      }

      // move alignment by one position
      for (PositionIndex i = 0; i < nSrcSentIndexVector.size(); i++)
      {
        if (i != iAlig && (i != 0 || (tlen >= 2 * (bestFertility[0] + 1))) && bestFertility[i] + 1 < MaxFertility)
        {
          double changeScore = moveScore(nSrcSentIndexVector, trgSentIndexVector, bestAlig, bestFertility, i, j);
          moveScores(i, j) = changeScore;
          if (changeScore > bestChangeScore)
          {
            bestChangeScore = changeScore;
            bestChangeType = 2;
            bestChangeArg1 = j;
            bestChangeArg2 = i;
          }
        }
        else
        {
          moveScores(i, j) = 1.0;
        }
      }

      if (bestChangeType == 1)
      {
        // swap
        PositionIndex j = bestChangeArg1;
        PositionIndex j1 = bestChangeArg2;
        PositionIndex i = bestAlig[j - 1];
        PositionIndex i1 = bestAlig[j1 - 1];
        bestAlig[j - 1] = i1;
        bestAlig[j1 - 1] = i;
      }
      else if (bestChangeType == 2)
      {
        // move
        PositionIndex j = bestChangeArg1;
        PositionIndex i = bestChangeArg2;
        PositionIndex iOld = bestAlig[j - 1];
        bestAlig[j - 1] = i;
        bestFertility[i]++;
        bestFertility[iOld]--;
      }
    }
  }
  return calcIbm3ProbFromAlig(nSrcSentIndexVector, trgSentIndexVector, bestAlig, bestFertility);
}

double Ibm3AligModel::swapScore(const Sentence& nsrc, const Sentence& trg, const vector<PositionIndex>& alig,
  PositionIndex j1, PositionIndex j2)
{
  PositionIndex slen = (PositionIndex)nsrc.size() - 1;
  PositionIndex tlen = (PositionIndex)trg.size();

  Prob score = 1.0;
  PositionIndex i1 = alig[j1 - 1];
  PositionIndex i2 = alig[j2 - 1];
  if (i1 != i2)
  {
    WordIndex s1 = nsrc[i1];
    WordIndex s2 = nsrc[i2];
    WordIndex t1 = trg[j1 - 1];
    WordIndex t2 = trg[j2 - 1];
    score = (pts(s2, t1) / pts(s1, t1)) * (pts(s1, t2) / pts(s2, t2));
    if (i1 > 0)
      score *= distortionProb(i1, slen, tlen, j2) / distortionProb(i1, slen, tlen, j1);
    if (i2 > 0)
      score *= distortionProb(i2, slen, tlen, j1) / distortionProb(i2, slen, tlen, j2);
  }
  return score;
}

double Ibm3AligModel::moveScore(const Sentence& nsrc, const Sentence& trg, const vector<PositionIndex>& alig,
  const vector<PositionIndex>& fertility, PositionIndex iNew, PositionIndex j)
{
  PositionIndex iOld = alig[j - 1];
  if (iOld == iNew)
    return 1.0;

  PositionIndex slen = (PositionIndex)nsrc.size() - 1;
  PositionIndex tlen = (PositionIndex)trg.size();
  WordIndex sOld = nsrc[iOld];
  WordIndex sNew = nsrc[iNew];
  WordIndex t = trg[j - 1];
  PositionIndex phi0 = fertility[0];
  PositionIndex phiNew = fertility[iOld];
  PositionIndex phiOld = fertility[iNew];
  Prob p0 = Prob(1.0) - p1;
  Prob score;
  if (iOld == 0)
  {
    score = (p0 * p0 / p1)
      * ((phi0 * (tlen - phi0 + 1.0)) / ((tlen - 2 * phi0 + 1) * (tlen - 2 * phi0 + 2.0)))
      * (phiNew + 1.0)
      * (fertilityProb(sNew, phiNew + 1) / fertilityProb(sNew, phiNew))
      * (pts(sNew, t) / pts(sOld, t))
      * distortionProb(iNew, slen, tlen, j);
  }
  else if (iNew == 0)
  {
    score = (p1 / (p0 * p0))
      * (double((tlen - 2 * phi0) * (tlen - 2 * phi0 - 1)) / ((1 + phi0) * (tlen - phi0)))
      * (1.0 / phiOld)
      * (fertilityProb(sOld, phiOld - 1) / fertilityProb(sOld, phiOld))
      * (pts(sNew, t) / pts(sOld, t))
      * (Prob(1.0) / distortionProb(iOld, slen, tlen, j));
  }
  else
  {
    score = Prob((phiNew + 1.0) / phiOld)
      * (fertilityProb(sOld, phiOld - 1) / fertilityProb(sOld, phiOld))
      * (fertilityProb(sNew, phiNew + 1) / fertilityProb(sNew, phiNew))
      * (pts(sNew, t) / pts(sOld, t))
      * (distortionProb(iNew, slen, tlen, j) / distortionProb(iOld, slen, tlen, j));
  }
  return score;
}

void Ibm3AligModel::clear()
{
  IncrIbm2AligModel::clear();
  distortionTable.clear();
  fertilityTable.clear();
  p1 = 0.5;
  p0Count = 0;
  p1Count = 0;
}

void Ibm3AligModel::clearInfoAboutSentRange()
{
  IncrIbm2AligModel::clearInfoAboutSentRange();
  distortionCounts.clear();
  fertilityCounts.clear();
  p0Count = 0;
  p1Count = 0;
}

void Ibm3AligModel::clearTempVars()
{
  IncrIbm2AligModel::clearTempVars();
  distortionCounts.clear();
  fertilityCounts.clear();
  p0Count = 0;
  p1Count = 0;
}

Ibm3AligModel::~Ibm3AligModel()
{

}
