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

void Ibm3AligModel::initialBatchPass(pair<unsigned int, unsigned int> sentPairRange, int verbose)
{
  p1 = 0.5;
  IncrIbm2AligModel::initialBatchPass(sentPairRange, verbose);
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

    vector<PositionIndex> alig, fertility;
    std::vector<std::pair<std::vector<double>, std::vector<double>>> neighborProbs;
    double totalNeighborProb;
    lexAligM3LpForBestAlig(nsrc, trg, alig, fertility, neighborProbs, totalNeighborProb);

    for (PositionIndex j = 1; j <= trg.size(); j++)
    {
      for (PositionIndex i = 0; i < nsrc.size(); i++)
      {
        PositionIndex iOld = alig[j - 1];
        alig[j - 1] = i;
        fertility[i]++;
        fertility[iOld]--;

        double count = neighborProbs[j - 1].first[i] / totalNeighborProb;
        incrementCounts(nsrc, trg, alig, fertility, count);

        alig[j - 1] = iOld;
        fertility[i]--;
        fertility[iOld]++;
      }

      for (PositionIndex j2 = 1; j2 <= trg.size(); j2++)
      {
        if (j == j2)
          continue;

        PositionIndex i = alig[j - 1];
        PositionIndex i2 = alig[j2 - 1];
        alig[j - 1] = i2;
        alig[j2 - 1] = i;

        double count = neighborProbs[j - 1].second[j2 - 1] / totalNeighborProb;
        incrementCounts(nsrc, trg, alig, fertility, count);

        alig[j - 1] = i;
        alig[j2 - 1] = i2;
      }
    }
  }
}

void Ibm3AligModel::incrementCounts(const Sentence& nsrc, const Sentence& trg, const vector<PositionIndex>& alig,
  const vector<PositionIndex>& fertility, double count)
{
  for (PositionIndex j = 1; j <= trg.size(); j++)
  {
    PositionIndex i = alig[j - 1];
    incrementWordPairCounts(nsrc, trg, i, j, count);
  }

  PositionIndex nullFertility = fertility[0];
  #pragma omp atomic
  p1Count += nullFertility * count;
  #pragma omp atomic
  p0Count += ((double)trg.size() - 2 * nullFertility) * count;

  for (PositionIndex i = 0; i < nsrc.size(); i++)
  {
    WordIndex s = nsrc[i];
    PositionIndex phi = fertility[i];

    #pragma omp atomic
    fertilityCounts[s][phi] += count;
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
  std::vector<std::pair<std::vector<double>, std::vector<double>>> neighborProbs;
  double totalNeighborProb;
  LgProb lgProb = sentLenLgProb((PositionIndex)srcSentIndexVector.size(), (PositionIndex)trgSentIndexVector.size());
  lgProb += lexAligM3LpForBestAlig(addNullWordToWidxVec(srcSentIndexVector), trgSentIndexVector, bestAlig,
    bestFertility, neighborProbs, totalNeighborProb);

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
    return calcIbm3LgProbFromAlig(addNullWordToWidxVec(sSent), tSent, alig, fertility);
  }
}

LgProb Ibm3AligModel::calcIbm3LgProbFromAlig(const vector<WordIndex>& nsSent, const vector<WordIndex>& tSent,
  const vector<PositionIndex>& alig, const vector<PositionIndex>& fertility, int verbose)
{
  PositionIndex slen = (PositionIndex)nsSent.size() - 1;
  PositionIndex tlen = (PositionIndex)tSent.size();

  if (verbose) cerr << "Obtaining IBM Model 3 logprob...\n";

  Prob p0 = 1.0 - (double)p1;

  Prob prob = 1.0;
  PositionIndex nullFertility = fertility[0];
  prob *= pow(p1, (double)nullFertility) * pow(p0, (double)tlen - 2 * nullFertility);

  for (PositionIndex nf = 1; nf <= nullFertility; nf++)
    prob *= (double)(tlen - nullFertility - nf + 1) / nf;

  LgProb lgProb = prob.get_lp();
  for (PositionIndex i = 1; i <= slen; i++)
  {
    PositionIndex phi = fertility[i];
    lgProb += log(MathFuncs::factorial(phi));
    lgProb += logFertilityProb(nsSent[i], phi);
  }

  for (PositionIndex j = 1; j <= alig.size(); ++j)
  {
    PositionIndex i = alig[j - 1];
    WordIndex s = nsSent[i];
    WordIndex t = tSent[j - 1];

    lgProb += logpts(s, t);
    lgProb += logDistortionProb(i, slen, tlen, j);
  }
  return lgProb;
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
  for (PositionIndex phi = 0; phi < min(tlen, MaxFertility); phi++)
  {
    Prob sump = 0;
    Prob nullFertilityProb = 1.0;
    PositionIndex nullFertility = phi;
    nullFertilityProb *= pow(p1, (double)nullFertility) * pow(p0, (double)tlen - 2 * nullFertility);

    for (PositionIndex nf = 1; nf <= nullFertility; nf++)
      nullFertilityProb *= (double)(tlen - nullFertility - nf + 1) / nf;
    sump += nullFertilityProb;

    for (PositionIndex i = 1; i < nsSent.size(); i++)
      sump += MathFuncs::factorial(phi) * (double)fertilityProb(nsSent[i], phi);
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

LgProb Ibm3AligModel::lexAligM3LpForBestAlig(const vector<WordIndex>& nSrcSentIndexVector,
  const vector<WordIndex>& trgSentIndexVector, vector<PositionIndex>& bestAlig, vector<PositionIndex>& bestFertility,
  std::vector<std::pair<std::vector<double>, std::vector<double>>>& neighborProbs, double& totalNeighborProb)
{
  // Initialize variables
  PositionIndex slen = (PositionIndex)nSrcSentIndexVector.size() - 1;
  PositionIndex tlen = (PositionIndex)trgSentIndexVector.size();

  // start with best IBM-2 alignment
  lexAligM2LpForBestAlig(nSrcSentIndexVector, trgSentIndexVector, bestAlig, bestFertility);
  LgProb bestLgProb = calcIbm3LgProbFromAlig(nSrcSentIndexVector, trgSentIndexVector, bestAlig, bestFertility);

  // hillclimbing search
  bool newBest = false;
  do
  {
    vector<PositionIndex> neighborAlig = bestAlig;
    vector<PositionIndex> neighborFertility = bestFertility;
    neighborProbs.clear();
    totalNeighborProb = 0;
    newBest = false;
    for (PositionIndex j = 1; j <= trgSentIndexVector.size(); j++)
    {
      vector<double> moveProbs;
      // move alignment by one position
      for (PositionIndex i = 0; i < nSrcSentIndexVector.size(); i++)
      {
        PositionIndex iOld = neighborAlig[j - 1];
        neighborAlig[j - 1] = i;
        neighborFertility[i]++;
        neighborFertility[iOld]--;

        LgProb neigborLgProb = calcIbm3LgProbFromAlig(nSrcSentIndexVector, trgSentIndexVector, neighborAlig,
          neighborFertility);
        Prob neighborProb = neigborLgProb.get_p();
        moveProbs.push_back(neighborProb);
        totalNeighborProb += (double)neighborProb;
        if (neigborLgProb > bestLgProb)
        {
          bestAlig = neighborAlig;
          bestFertility = neighborFertility;
          bestLgProb = neigborLgProb;
          newBest = true;
        }

        neighborAlig[j - 1] = iOld;
        neighborFertility[i]--;
        neighborFertility[iOld]++;
      }

      vector<double> swapProbs;
      // swap alignments
      for (PositionIndex j2 = 1; j2 <= trgSentIndexVector.size(); j2++)
      {
        if (j == j2)
        {
          swapProbs.push_back(1.0);
          continue;
        }

        PositionIndex i = neighborAlig[j - 1];
        PositionIndex i2 = neighborAlig[j2 - 1];
        neighborAlig[j - 1] = i2;
        neighborAlig[j2 - 1] = i;

        LgProb neigborLgProb = calcIbm3LgProbFromAlig(nSrcSentIndexVector, trgSentIndexVector, neighborAlig,
          neighborFertility);
        Prob neighborProb = neigborLgProb.get_p();
        swapProbs.push_back(neighborProb);
        totalNeighborProb += (double)neighborProb;
        if (neigborLgProb > bestLgProb)
        {
          bestAlig = neighborAlig;
          bestFertility = neighborFertility;
          bestLgProb = neigborLgProb;
          newBest = true;
        }

        neighborAlig[j - 1] = i;
        neighborAlig[j2 - 1] = i2;
      }

      neighborProbs.push_back(make_pair(moveProbs, swapProbs));
    }
  } while (!newBest);

  return bestLgProb;
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
