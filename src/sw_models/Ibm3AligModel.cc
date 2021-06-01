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

Ibm3AligModel::Ibm3AligModel() : Ibm2AligModel(), p0Count(0), p1Count(0), p1(0.5)
{
}

void Ibm3AligModel::initSourceWord(const Sentence& nsrc, const Sentence& trg, PositionIndex i)
{
  Ibm2AligModel::initSourceWord(nsrc, trg, i);

  dSource ds;
  ds.i = i;
  ds.slen = (PositionIndex)nsrc.size() - 1;
  ds.tlen = (PositionIndex)trg.size();
  distortionTable.reserveSpace(ds);

  DistortionCountsElem& distortionEntry = distortionCounts[ds];
  if (distortionEntry.size() < trg.size())
    distortionEntry.resize(trg.size(), 0);
}

void Ibm3AligModel::addTranslationOptions(vector<vector<WordIndex>>& insertBuffer)
{
  WordIndex maxSrcWordIndex = (WordIndex)insertBuffer.size() - 1;

  if (maxSrcWordIndex >= lexCounts.size())
    lexCounts.resize((size_t)maxSrcWordIndex + 1);
  lexTable.reserveSpace(maxSrcWordIndex);

  if (maxSrcWordIndex >= fertilityCounts.size())
    fertilityCounts.resize((size_t)maxSrcWordIndex + 1);
  fertilityTable.reserveSpace(maxSrcWordIndex);

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

    AlignmentInfo alignment(slen, tlen);
    Matrix<double> moveScores, swapScores;
    Prob aligProb = searchForBestAlignment(nsrc, trg, true, alignment, &moveScores, &swapScores);
    Matrix<double> moveCounts(slen + 1, tlen + 1), swapCounts(slen + 1, tlen + 1);
    vector<double> negMove(tlen + 1), negSwap(tlen + 1), plus1Fert(slen + 1), minus1Fert(slen + 1);
    double totalMove = aligProb;
    double totalSwap = 0;

    for (PositionIndex j = 1; j <= tlen; ++j)
    {
      for (PositionIndex i = 0; i <= slen; ++i)
      {
        if (alignment.get(j) != i)
        {
          double prob = aligProb * moveScores(i, j);
          if (prob < SmoothingAnjiNum)
            prob = SmoothingAnjiNum;
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
          if (prob < SmoothingAnjiNum)
            prob = SmoothingAnjiNum;
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
        double count
            = i == alignment.get(j) ? totalCount - (negMove[j] + negSwap[j]) : moveCounts(i, j) + swapCounts(i, j);
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
  Ibm2AligModel::incrementWordPairCounts(nsrc, trg, i, j, count);

  dSource ds;
  ds.i = i;
  ds.slen = (PositionIndex)nsrc.size() - 1;
  ds.tlen = (PositionIndex)trg.size();

#pragma omp atomic
  distortionCounts[ds][j] += count;
}

void Ibm3AligModel::incrementTargetWordCounts(const Sentence& nsrc, const Sentence& trg, const AlignmentInfo& alignment,
                                              PositionIndex j, double count)
{
}

void Ibm3AligModel::batchMaximizeProbs()
{
  Ibm2AligModel::batchMaximizeProbs();

#pragma omp parallel for schedule(dynamic)
  for (int asIndex = 0; asIndex < (int)distortionCounts.size(); ++asIndex)
  {
    double denom = 0;
    const pair<dSource, DistortionCountsElem>& p = distortionCounts.getAt(asIndex);
    const dSource& ds = p.first;
    DistortionCountsElem& elem = const_cast<DistortionCountsElem&>(p.second);
    for (PositionIndex j = 0; j < (PositionIndex)elem.size(); ++j)
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
    FertilityCountsElem& elem = fertilityCounts[s];
    for (PositionIndex phi = 0; phi < (PositionIndex)elem.size(); ++phi)
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
  dSource ds;
  ds.i = i;
  ds.slen = slen;
  ds.tlen = tlen;

  bool found;
  double numer = distortionTable.getDistortionNumer(ds, j, found);
  if (found)
  {
    // numerator for pair ds,j exists
    double denom = distortionTable.getDistortionDenom(ds, found);
    if (found)
      return numer - denom;
  }
  return SMALL_LG_NUM;
}

double Ibm3AligModel::distortionProb(PositionIndex i, PositionIndex slen, PositionIndex tlen, PositionIndex j,
                                     bool training)
{
  if (training)
  {
    double logProb = unsmoothedLogDistortionProb(i, slen, tlen, j);
    if (logProb != SMALL_LG_NUM)
      return exp(logProb);
    return 1.0 / tlen;
  }
  return distortionProb(i, slen, tlen, j);
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
    if (found)
      return numer - denom;
  }
  return SMALL_LG_NUM;
}

double Ibm3AligModel::fertilityProb(WordIndex s, PositionIndex phi, bool training)
{
  if (training)
  {
    double logProb = unsmoothedLogFertilityProb(s, phi);
    if (logProb != SMALL_LG_NUM)
      return exp(logProb);
    if (phi == 0)
      return 0.2;
    if (phi == 1)
      return 0.65;
    if (phi == 2)
      return 0.1;
    if (phi == 3)
      return 0.04;
    if (phi >= 4 && phi < MaxFertility)
      return 0.01 / (MaxFertility - 4);
    return exp(SMALL_LG_NUM);
  }
  return fertilityProb(s, phi);
}

LgProb Ibm3AligModel::obtainBestAlignment(const vector<WordIndex>& src, const vector<WordIndex>& trg,
                                          WordAligMatrix& bestWaMatrix)
{
  PositionIndex slen = (PositionIndex)src.size();
  PositionIndex tlen = (PositionIndex)trg.size();

  AlignmentInfo bestAlignment(slen, tlen);
  LgProb lgProb = sentLenLgProb(slen, tlen);
  lgProb += searchForBestAlignment(addNullWordToWidxVec(src), trg, false, bestAlignment).get_lp();

  bestWaMatrix.init(slen, tlen);
  bestWaMatrix.putAligVec(bestAlignment.getAlignment());

  return lgProb;
}

LgProb Ibm3AligModel::calcLgProbForAlig(const vector<WordIndex>& src, const vector<WordIndex>& trg,
                                        const WordAligMatrix& aligMatrix, int verbose)
{
  PositionIndex slen = (PositionIndex)src.size();
  PositionIndex tlen = (PositionIndex)trg.size();

  vector<PositionIndex> aligVec;
  aligMatrix.getAligVec(aligVec);

  if (verbose)
  {
    for (PositionIndex i = 0; i < slen; ++i)
      cerr << src[i] << " ";
    cerr << "\n";
    for (PositionIndex j = 0; j < tlen; ++j)
      cerr << trg[j] << " ";
    cerr << "\n";
    for (PositionIndex j = 0; j < tlen; ++j)
      cerr << aligVec[j] << " ";
    cerr << "\n";
  }
  if (trg.size() != aligVec.size())
  {
    cerr << "Error: the sentence t and the alignment vector have not the same size." << endl;
    return THOT_ERROR;
  }
  else
  {
    AlignmentInfo alignment(slen, tlen);
    alignment.setAlignment(aligVec);
    return calcProbOfAlignment(addNullWordToWidxVec(src), trg, false, alignment, verbose).get_lp();
  }
}

Prob Ibm3AligModel::calcProbOfAlignment(const vector<WordIndex>& nsrc, const vector<WordIndex>& trg, bool training,
                                        AlignmentInfo& alignment, int verbose)
{
  if (alignment.getProb() >= 0.0)
    return alignment.getProb();

  PositionIndex slen = (PositionIndex)nsrc.size() - 1;
  PositionIndex tlen = (PositionIndex)trg.size();

  if (verbose)
    cerr << "Obtaining IBM Model 3 prob...\n";

  Prob p0 = Prob(1.0) - p1;

  PositionIndex phi0 = alignment.getFertility(0);
  Prob prob = pow(p0, double(tlen - 2 * phi0)) * pow(p1, double(phi0));

  for (PositionIndex phi = 1; phi <= phi0; ++phi)
    prob *= double(tlen - phi0 - phi + 1) / phi;

  for (PositionIndex i = 1; i <= slen; ++i)
  {
    WordIndex s = nsrc[i];
    PositionIndex phi = alignment.getFertility(i);
    prob *= Prob(MathFuncs::factorial(phi)) * fertilityProb(s, phi, training);
  }

  for (PositionIndex j = 1; j <= tlen; ++j)
  {
    PositionIndex i = alignment.get(j);
    WordIndex s = nsrc[i];
    WordIndex t = trg[j - 1];

    prob *= pts(s, t, training) * distortionProb(i, slen, tlen, j, training);
  }
  alignment.setProb(prob);
  return prob;
}

LgProb Ibm3AligModel::calcLgProb(const vector<WordIndex>& src, const vector<WordIndex>& trg, int verbose)
{
  PositionIndex slen = (PositionIndex)src.size();
  PositionIndex tlen = (PositionIndex)trg.size();
  vector<PositionIndex> nsrc = addNullWordToWidxVec(src);

  if (verbose)
    cerr << "Obtaining Sum IBM Model 3 logprob...\n";

  Prob p0 = 1.0 - (double)p1;

  LgProb lgProb = sentLenLgProb(slen, tlen);
  LgProb fertilityContrib = 0;
  for (PositionIndex fertility = 0; fertility < min(tlen, MaxFertility); ++fertility)
  {
    Prob sump = 0;
    Prob prob = 1.0;
    PositionIndex phi0 = fertility;
    prob *= pow(p1, double(phi0)) * pow(p0, double(tlen - 2 * phi0));

    for (PositionIndex phi = 1; phi <= phi0; phi++)
      prob *= double(tlen - phi0 - phi + 1) / phi;
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
      WordIndex t = trg[j - 1];

      sump += pts(s, t) * distortionProb(i, slen, tlen, j);
    }
    lexDistorionContrib += sump.get_lp();
  }

  if (verbose)
    cerr << "- Lexical plus distortion contribution= " << lexDistorionContrib << endl;
  lgProb += lexDistorionContrib;

  return lgProb;
}

bool Ibm3AligModel::load(const char* prefFileName, int verbose)
{
  // Load IBM 2 Model data
  bool retVal = Ibm2AligModel::load(prefFileName, verbose);
  if (retVal == THOT_ERROR)
    return retVal;

  if (verbose)
    cerr << "Loading IBM 3 Model data..." << endl;

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

bool Ibm3AligModel::print(const char* prefFileName, int verbose)
{
  // Print IBM 2 Model data
  bool retVal = Ibm2AligModel::print(prefFileName);
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

Prob Ibm3AligModel::searchForBestAlignment(const vector<WordIndex>& nsrc, const vector<WordIndex>& trg, bool training,
                                           AlignmentInfo& bestAlignment, Matrix<double>* moveScores,
                                           Matrix<double>* swapScores)
{
  PositionIndex slen = (PositionIndex)nsrc.size() - 1;
  PositionIndex tlen = (PositionIndex)trg.size();

  // start with IBM-2 alignment
  getInitialAlignmentForSearch(nsrc, trg, training, bestAlignment);

  if (moveScores != nullptr)
    moveScores->resize(slen + 1, tlen + 1);
  if (swapScores != nullptr)
    swapScores->resize(tlen + 1, tlen + 1);

  // hillclimbing search
  int bestChangeType = -1;
  while (bestChangeType != 0)
  {
    bestChangeType = 0;
    PositionIndex bestChangeArg1;
    PositionIndex bestChangeArg2;
    double bestChangeScore = 1.00001;
    for (PositionIndex j = 1; j <= tlen; j++)
    {
      PositionIndex iAlig = bestAlignment.get(j);

      // swap alignments
      for (PositionIndex j1 = j + 1; j1 <= tlen; j1++)
      {
        if (iAlig != bestAlignment.get(j1))
        {
          double changeScore = swapScore(nsrc, trg, j, j1, training, bestAlignment);
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
          double changeScore = moveScore(nsrc, trg, i, j, training, bestAlignment);
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
  }
  return calcProbOfAlignment(nsrc, trg, training, bestAlignment);
}

void Ibm3AligModel::getInitialAlignmentForSearch(const vector<WordIndex>& nsrc, const vector<WordIndex>& trg,
                                                 bool training, AlignmentInfo& alignment)
{
  PositionIndex slen = (PositionIndex)nsrc.size() - 1;
  PositionIndex tlen = (PositionIndex)trg.size();

  vector<PositionIndex> fertility(slen + 1, 0);

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
        double prob = pts(s, t, training) * aProb(j, slen, tlen, i, training);
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

double Ibm3AligModel::swapScore(const Sentence& nsrc, const Sentence& trg, PositionIndex j1, PositionIndex j2,
                                bool training, AlignmentInfo& alignment)
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
  Prob score = (pts(s2, t1, training) / pts(s1, t1, training)) * (pts(s1, t2, training) / pts(s2, t2, training));
  if (i1 > 0)
    score *= distortionProb(i1, slen, tlen, j2, training) / distortionProb(i1, slen, tlen, j1, training);
  if (i2 > 0)
    score *= distortionProb(i2, slen, tlen, j1, training) / distortionProb(i2, slen, tlen, j2, training);
  return score;
}

double Ibm3AligModel::moveScore(const Sentence& nsrc, const Sentence& trg, PositionIndex iNew, PositionIndex j,
                                bool training, AlignmentInfo& alignment)
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
    score = (p0 * p0 / p1) * ((phi0 * (tlen - phi0 + 1.0)) / ((tlen - 2 * phi0 + 1) * (tlen - 2 * phi0 + 2.0)))
          * (phiNew + 1.0) * (fertilityProb(sNew, phiNew + 1, training) / fertilityProb(sNew, phiNew, training))
          * (pts(sNew, t, training) / pts(sOld, t, training)) * distortionProb(iNew, slen, tlen, j, training);
  }
  else if (iNew == 0)
  {
    score = (p1 / (p0 * p0)) * (double((tlen - 2 * phi0) * (tlen - 2 * phi0 - 1)) / ((1 + phi0) * (tlen - phi0)))
          * (1.0 / phiOld) * (fertilityProb(sOld, phiOld - 1, training) / fertilityProb(sOld, phiOld, training))
          * (pts(sNew, t, training) / pts(sOld, t, training))
          * (Prob(1.0) / distortionProb(iOld, slen, tlen, j, training));
  }
  else
  {
    score = Prob((phiNew + 1.0) / phiOld)
          * (fertilityProb(sOld, phiOld - 1, training) / fertilityProb(sOld, phiOld, training))
          * (fertilityProb(sNew, phiNew + 1, training) / fertilityProb(sNew, phiNew, training))
          * (pts(sNew, t, training) / pts(sOld, t, training))
          * (distortionProb(iNew, slen, tlen, j, training) / distortionProb(iOld, slen, tlen, j, training));
  }
  return score;
}

void Ibm3AligModel::clear()
{
  Ibm2AligModel::clear();
  distortionTable.clear();
  fertilityTable.clear();
  p1 = 0.5;
  p0Count = 0;
  p1Count = 0;
}

void Ibm3AligModel::clearInfoAboutSentRange()
{
  Ibm2AligModel::clearInfoAboutSentRange();
  distortionCounts.clear();
  fertilityCounts.clear();
  p0Count = 0;
  p1Count = 0;
}

void Ibm3AligModel::clearTempVars()
{
  Ibm2AligModel::clearTempVars();
  distortionCounts.clear();
  fertilityCounts.clear();
  p0Count = 0;
  p1Count = 0;
}

Ibm3AligModel::~Ibm3AligModel()
{
}
