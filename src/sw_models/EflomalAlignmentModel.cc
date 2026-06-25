#include "sw_models/EflomalAlignmentModel.h"

#include "nlp_common/ErrorDefs.h"
#include "nlp_common/MathDefs.h"
#include "sw_models/Md.h"
#include "sw_models/SamplingUtils.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>

using namespace std;

EflomalAlignmentModel::EflomalAlignmentModel()
    : lexTable{make_shared<MemoryLexTable>()}, jumpTable{make_shared<EflomalJumpTable>()},
      fertilityTable{make_shared<FertilityTable>()}
{
  // Initialize with a single chain pointing at the model-level tables so that
  // decode and query methods work before startTraining / load is called.
  SamplerChain ch;
  ch.chainSeed = DefaultSeed;
  ch.lexTable = lexTable;
  ch.jumpTable = jumpTable;
  ch.fertilityTable = fertilityTable;
  chains.push_back(std::move(ch));
}

void EflomalAlignmentModel::setSeed(unsigned int s)
{
  seed = s;
}

unsigned int EflomalAlignmentModel::getSeed() const
{
  return seed;
}

void EflomalAlignmentModel::setNumSamplers(int value)
{
  numSamplers = value < 1 ? 1 : value;
}

int EflomalAlignmentModel::getNumSamplers() const
{
  return numSamplers;
}

void EflomalAlignmentModel::setDeterministic(bool value)
{
  deterministic = value;
}

bool EflomalAlignmentModel::getDeterministic() const
{
  return deterministic;
}

void EflomalAlignmentModel::setIterations(int ibm1, int hmm, int fertility)
{
  ibm1Iters = ibm1;
  hmmIters = hmm;
  fertilityIters = fertility;
  // An explicit schedule overrides the corpus-scaled auto schedule (otherwise
  // startTraining would discard these values when autoIterations is on, which is
  // now the default).
  autoIterations = false;
}

int EflomalAlignmentModel::getIbm1Iters() const
{
  return ibm1Iters;
}

int EflomalAlignmentModel::getHmmIters() const
{
  return hmmIters;
}

int EflomalAlignmentModel::getFertilityIters() const
{
  return fertilityIters;
}

void EflomalAlignmentModel::setP0(double value)
{
  p0 = value;
}

double EflomalAlignmentModel::getP0() const
{
  return p0;
}

void EflomalAlignmentModel::setAlphaLex(double value)
{
  alphaLex = value;
}

double EflomalAlignmentModel::getAlphaLex() const
{
  return alphaLex;
}

void EflomalAlignmentModel::setAlphaJump(double value)
{
  alphaJump = value;
}

double EflomalAlignmentModel::getAlphaJump() const
{
  return alphaJump;
}

void EflomalAlignmentModel::setAlphaFertility(double value)
{
  alphaFertility = value;
}

double EflomalAlignmentModel::getAlphaFertility() const
{
  return alphaFertility;
}

void EflomalAlignmentModel::setJumpWindow(int value)
{
  jumpWindow = value;
}

int EflomalAlignmentModel::getJumpWindow() const
{
  return jumpWindow;
}

void EflomalAlignmentModel::setEflomalLexNorm(bool value)
{
  eflomalLexNorm = value;
}

bool EflomalAlignmentModel::getEflomalLexNorm() const
{
  return eflomalLexNorm;
}

void EflomalAlignmentModel::setAutoIterations(bool value)
{
  autoIterations = value;
}

bool EflomalAlignmentModel::getAutoIterations() const
{
  return autoIterations;
}

int EflomalAlignmentModel::getScheduledIterations() const
{
  return ibm1Iters + hmmIters + fertilityIters;
}

void EflomalAlignmentModel::setDecodeParams(int samplers, int iters, int burnIn)
{
  decodeSamplers = samplers;
  decodeIters = iters;
  decodeBurnIn = burnIn;
  decodeParamsExplicit = true;
}

int EflomalAlignmentModel::getDecodeSamplers() const
{
  return decodeSamplers;
}

int EflomalAlignmentModel::getDecodeIters() const
{
  return decodeIters;
}

int EflomalAlignmentModel::getDecodeBurnIn() const
{
  return decodeBurnIn;
}

EflomalAlignmentModel::Stage EflomalAlignmentModel::stageForIter(int it) const
{
  if (it < ibm1Iters)
    return Ibm1Stage;
  if (it < ibm1Iters + hmmIters)
    return HmmStage;
  if (it < ibm1Iters + hmmIters + fertilityIters)
    return FertilityStage;
  // Extra iterations beyond the schedule repeat the last non-empty stage.
  if (fertilityIters > 0)
    return FertilityStage;
  if (hmmIters > 0)
    return HmmStage;
  return Ibm1Stage;
}

int EflomalAlignmentModel::jumpBucket(int offset) const
{
  int clamped = std::max(-jumpWindow, std::min(jumpWindow, offset));
  return clamped + jumpWindow;
}

vector<WordIndex> EflomalAlignmentModel::getSrcSent(unsigned int n)
{
  vector<string> srcStr;
  vector<WordIndex> result;
  sentenceHandler->getSrcSentence(n, srcStr);
  for (const string& w : srcStr)
  {
    WordIndex widx = stringToSrcWordIndex(w);
    if (widx == UNK_WORD)
      widx = addSrcSymbol(w);
    result.push_back(widx);
  }
  return result;
}

vector<WordIndex> EflomalAlignmentModel::getTrgSent(unsigned int n)
{
  vector<string> trgStr;
  vector<WordIndex> result;
  sentenceHandler->getTrgSentence(n, trgStr);
  for (const string& w : trgStr)
  {
    WordIndex widx = stringToTrgWordIndex(w);
    if (widx == UNK_WORD)
      widx = addTrgSymbol(w);
    result.push_back(widx);
  }
  return result;
}

unsigned int EflomalAlignmentModel::startTraining(int /*verbosity*/)
{
  clearTempVars();
  trainingAlignments.clear(); // recomputed in endTraining if emitTrainingAlignments
  buildCorpus();
  if (autoIterations)
  {
    // eflomal's corpus-scaled schedule for the full (model 3) cascade:
    //   iters  = max(2, round(5000 / sqrt(N)))
    //   iters4 = max(1, iters / 4)
    //   IBM1 = max(2, iters4), HMM = iters4, fertility = iters
    // i.e. the earlier stages get ~a quarter of the sweeps and the final
    // fertility stage gets the full count (NOT all three equal).
    size_t n = corpusSrc.size();
    int iters = n == 0 ? 2 : std::max(2, (int)llround(5000.0 / std::sqrt((double)n)));
    int iters4 = std::max(1, iters / 4);
    ibm1Iters = std::max(2, iters4);
    hmmIters = iters4;
    fertilityIters = iters;
  }

  // Tie the decode parameters to the (now-resolved) training schedule unless the
  // caller pinned them with setDecodeParams (or they came from a loaded config).
  // Decode is the same Gibbs sampler over the same posterior, so its sampler /
  // iteration needs track training's; a 27-dataset AER sweep confirmed this beats
  // the old fixed (8,20,4). These tied values then serialize via createConfig.
  if (!decodeParamsExplicit)
  {
    // decodeSamplers=1: the numSamplers training chains already supply the decode
    // ensemble (validated equivalent to tying it to numSamplers, but linear not
    // quadratic in decode cost). decodeIters tracks the schedule; no burn-in.
    decodeSamplers = 1;
    decodeIters = std::max(MinDecodeIters, ibm1Iters + hmmIters + fertilityIters);
    decodeBurnIn = 0;
  }

  size_t srcVocabSize = getSrcVocabSize();
  size_t buckets = size_t{2} * jumpWindow + 1;

  chains.resize((size_t)numSamplers);
  for (int s = 0; s < numSamplers; ++s)
  {
    SamplerChain& chain = chains[(size_t)s];
    chain.chainSeed = seed + (unsigned int)s * 2654435761u;
    chain.rng.seed(chain.chainSeed);
    chain.lexCounts.assign(srcVocabSize, {});
    chain.lexCountSum.assign(srcVocabSize, 0);
    chain.jumpCounts.assign(buckets, 0);
    chain.jumpCountSum = 0;
    chain.fertCounts.assign(srcVocabSize, vector<double>(MaxFertility + 1, 0));
    chain.fertCountSum.assign(srcVocabSize, 0);
    chain.fertRatioSampled.clear();
    chain.lexPriorMass = chain.jumpPriorMass = chain.fertPriorMass = 0;
    chain.accumLexCounts.assign(srcVocabSize, {});
    chain.accumLexCountSum.assign(srcVocabSize, 0);
    chain.accumulated = false;
    chain.lexTable = make_shared<MemoryLexTable>();
    chain.jumpTable = make_shared<EflomalJumpTable>();
    chain.fertilityTable = make_shared<FertilityTable>();
    initializeChain(chain);
  }

  // Wire model-level query tables to chain[0]'s tables.
  lexTable = chains[0].lexTable;
  jumpTable = chains[0].jumpTable;
  fertilityTable = chains[0].fertilityTable;

  return (unsigned int)corpusSrc.size();
}

void EflomalAlignmentModel::buildCorpus()
{
  corpusSrc.clear();
  corpusTrg.clear();
  totLenRatio = 0;

  for (unsigned int n = 0; n < numSentencePairs(); ++n)
  {
    vector<WordIndex> src = getSrcSent(n);
    vector<WordIndex> trg = getTrgSent(n);
    if (sentenceLengthIsOk(src) && sentenceLengthIsOk(trg))
    {
      totLenRatio += static_cast<double>(trg.size()) / static_cast<double>(src.size());
      corpusSrc.push_back(addNullWordToWidxVec(src));
      corpusTrg.push_back(trg);
    }
  }
}

void EflomalAlignmentModel::initializeChain(SamplerChain& chain)
{
  chain.alig.assign(corpusSrc.size(), {});
  const bool randomInit = numSamplers > 1;
  std::uniform_real_distribution<double> u(0.0, 1.0);
  for (size_t n = 0; n < corpusSrc.size(); ++n)
  {
    const vector<WordIndex>& nsrc = corpusSrc[n];
    const vector<WordIndex>& trg = corpusTrg[n];
    PositionIndex slen = (PositionIndex)nsrc.size() - 1;
    PositionIndex tlen = (PositionIndex)trg.size();
    chain.alig[n].assign(tlen, 0);

    for (PositionIndex j = 1; j <= tlen; ++j)
    {
      PositionIndex i;
      if (randomInit)
      {
        i = 0;
        if (slen > 0 && u(chain.rng) >= p0)
          i = (PositionIndex)(1 + (chain.rng() % (uint64_t)slen));
      }
      else
      {
        i = (PositionIndex)llround((double)j * slen / tlen);
        if (i < 1)
          i = 1;
        if (i > slen)
          i = slen;
      }
      chain.alig[n][j - 1] = i;
      WordIndex e = nsrc[i];
      WordIndex f = trg[j - 1];
      chain.lexCounts[e][f] += 1;
      chain.lexCountSum[e] += 1;
    }
  }
}

// Recomputes the jump distribution from the current alignment using eflomal's
// nearest-non-NULL neighbour scheme: every non-NULL token contributes an incoming
// jump (from the nearest non-NULL word on its left) and an outgoing jump (to the
// nearest non-NULL word on its right); a NULL-aligned token contributes the single
// "skip" jump between its two non-NULL neighbours. Virtual BOS = -1, EOS = slen.
void EflomalAlignmentModel::computeJumpCounts(SamplerChain& chain)
{
  fill(chain.jumpCounts.begin(), chain.jumpCounts.end(), 0.0);
  chain.jumpCountSum = 0;
  std::vector<int> aaRight;
  for (size_t n = 0; n < corpusSrc.size(); ++n)
  {
    PositionIndex slen = (PositionIndex)corpusSrc[n].size() - 1;
    PositionIndex tlen = (PositionIndex)corpusTrg[n].size();
    aaRight.assign(tlen, (int)slen);
    int aa = (int)slen;
    for (PositionIndex j = tlen; j >= 1; --j)
    {
      aaRight[j - 1] = aa;
      if (chain.alig[n][j - 1] > 0)
        aa = (int)chain.alig[n][j - 1] - 1;
    }
    int aaLeft = -1;
    for (PositionIndex j = 1; j <= tlen; ++j)
    {
      PositionIndex i = chain.alig[n][j - 1];
      int aR = aaRight[j - 1];
      if (i > 0)
      {
        chain.jumpCounts[jumpBucket(((int)i - 1) - aaLeft)] += 1;
        chain.jumpCounts[jumpBucket(aR - ((int)i - 1))] += 1;
        chain.jumpCountSum += 2;
        aaLeft = (int)i - 1;
      }
      else
      {
        chain.jumpCounts[jumpBucket(aR - aaLeft)] += 1;
        chain.jumpCountSum += 1;
      }
    }
  }
}

void EflomalAlignmentModel::computeFertilityCounts(SamplerChain& chain)
{
  for (auto& row : chain.fertCounts)
    fill(row.begin(), row.end(), 0.0);
  fill(chain.fertCountSum.begin(), chain.fertCountSum.end(), 0.0);

  for (size_t n = 0; n < corpusSrc.size(); ++n)
  {
    const vector<WordIndex>& nsrc = corpusSrc[n];
    PositionIndex slen = (PositionIndex)nsrc.size() - 1;
    vector<int> phi(slen + 1, 0);
    for (PositionIndex j = 1; j <= (PositionIndex)corpusTrg[n].size(); ++j)
      phi[chain.alig[n][j - 1]]++;
    for (PositionIndex i = 1; i <= slen; ++i)
    {
      WordIndex e = nsrc[i];
      chain.fertCounts[e][std::min(phi[i], (int)MaxFertility)] += 1;
      chain.fertCountSum[e] += 1;
    }
  }
}

void EflomalAlignmentModel::sampleFertilityRatios(SamplerChain& chain)
{
  size_t srcVocabSize = getSrcVocabSize();
  chain.fertRatioSampled.assign(srcVocabSize, vector<double>(MaxFertility + 1, 1.0));
  vector<double> draw(MaxFertility + 1);
  for (WordIndex s = 0; s < (WordIndex)srcVocabSize; ++s)
  {
    if (chain.fertCountSum[s] <= 0)
      continue;
    // Draw an unnormalized categorical fertility distribution from the Dirichlet
    // posterior alpha[phi] = FERT_ALPHA + count(s, phi) via gamma variates.
    for (PositionIndex phi = 0; phi <= MaxFertility; ++phi)
    {
      std::gamma_distribution<double> gamma(alphaFertility + chain.fertCounts[s][phi], 1.0);
      draw[phi] = gamma(chain.rng);
    }
    // Store the ratio P(phi)/P(phi-1); reaching the capped maximum fertility is
    // made very unlikely, as in eflomal.
    chain.fertRatioSampled[s][MaxFertility] = 1e-10;
    for (PositionIndex phi = MaxFertility - 1; phi >= 1; --phi)
      chain.fertRatioSampled[s][phi] = draw[phi - 1] > 0 ? draw[phi] / draw[phi - 1] : 0.0;
  }
}

void EflomalAlignmentModel::train(int /*verbosity*/)
{
  Stage stage = stageForIter(iter);
  bool accumulate = stage == FertilityStage;

#pragma omp parallel for schedule(static) if(!deterministic)
  for (int s = 0; s < (int)chains.size(); ++s)
  {
    SamplerChain& chain = chains[(size_t)s];
    if (stage != Ibm1Stage)
      computeJumpCounts(chain);
    if (stage == FertilityStage)
    {
      computeFertilityCounts(chain);
      sampleFertilityRatios(chain);
    }
    sampleSweep(chain, stage, accumulate);
    if (accumulate)
      chain.accumulated = true;
  }
  ++iter;
}

void EflomalAlignmentModel::sampleSweep(SamplerChain& chain, Stage stage, bool accumulate)
{
  // Dirichlet prior masses are constant for the whole sweep; cache them once
  // instead of recomputing inside the per-candidate inner loop.
  chain.lexPriorMass = alphaLex * (double)getTrgVocabSize();
  chain.jumpPriorMass = alphaJump * (double)chain.jumpCounts.size();
  chain.fertPriorMass = alphaFertility * (double)(MaxFertility + 1);

  // Floor for (jumpCount + alphaJump): the per-token collapse can briefly drive a
  // bucket slightly negative (the start-of-sweep right-neighbour table does not see
  // a left token flipping NULL<->non-NULL mid-sweep), which would otherwise yield a
  // non-positive sampling weight.
  constexpr double kMinJumpWeight = 1e-9;

  // Fully collapsed Gibbs: the lexical AND jump counts are updated per token within
  // the sweep, so every token is resampled against the current state of all tokens
  // already resampled this sweep.
  vector<float> lexDenomInv(chain.lexCountSum.size());
  auto recomputeInv = [&](WordIndex e) {
    double d = eflomalLexNorm ? std::max(chain.lexCountSum[e], 1.0) : chain.lexCountSum[e] + chain.lexPriorMass;
    lexDenomInv[e] = (float)(1.0 / d);
  };
  for (size_t e = 0; e < lexDenomInv.size(); ++e)
    recomputeInv((WordIndex)e);

  vector<double> weights;
  vector<int> aaRight;

  for (size_t n = 0; n < corpusSrc.size(); ++n)
  {
    const vector<WordIndex>& nsrc = corpusSrc[n];
    const vector<WordIndex>& trg = corpusTrg[n];
    PositionIndex slen = (PositionIndex)nsrc.size() - 1;
    PositionIndex tlen = (PositionIndex)trg.size();

    vector<int> phi;
    if (stage == FertilityStage)
    {
      phi.assign(slen + 1, 0);
      for (PositionIndex j = 1; j <= tlen; ++j)
        phi[chain.alig[n][j - 1]]++;
    }

    aaRight.assign(tlen, (int)slen);
    if (stage != Ibm1Stage)
    {
      int aa = (int)slen;
      for (PositionIndex j = tlen; j >= 1; --j)
      {
        aaRight[j - 1] = aa;
        if (chain.alig[n][j - 1] > 0)
          aa = (int)chain.alig[n][j - 1] - 1;
      }
    }

    weights.assign(slen + 1, 0.0);
    int aaLeft = -1;

    for (PositionIndex j = 1; j <= tlen; ++j)
    {
      WordIndex f = trg[j - 1];
      PositionIndex iOld = chain.alig[n][j - 1];
      WordIndex eOld = nsrc[iOld];
      int aR = aaRight[j - 1];

      // Remove the current token's lexical contribution (collapsed Gibbs).
      auto itOld = chain.lexCounts[eOld].find(f);
      if (itOld != chain.lexCounts[eOld].end())
      {
        // tsl::robin_map requires value() (not ->second) to mutate the mapped value.
        itOld.value() -= 1;
        if (itOld.value() <= 0)
          chain.lexCounts[eOld].erase(itOld);
      }
      chain.lexCountSum[eOld] -= 1;
      recomputeInv(eOld);
      if (stage == FertilityStage)
        phi[iOld]--;

      // Remove this token's jump contribution from the live counts (collapsed):
      // a non-NULL token contributes two jumps (into and out of its position), a
      // NULL token one skip jump connecting its two non-NULL neighbours.
      if (stage != Ibm1Stage)
      {
        if (iOld == 0)
        {
          chain.jumpCounts[jumpBucket(aR - aaLeft)] -= 1;
          chain.jumpCountSum -= 1;
        }
        else
        {
          int p0 = (int)iOld - 1;
          chain.jumpCounts[jumpBucket(p0 - aaLeft)] -= 1;
          chain.jumpCounts[jumpBucket(aR - p0)] -= 1;
          chain.jumpCountSum -= 2;
        }
      }

      // Jump normalizer Z (live). A non-NULL candidate has two jump factors (each
      // divided by Z) and NULL has one; rather than divide every candidate, the
      // weights are scaled by Z^2 (a positive constant that cancels in the
      // categorical normalization), so non-NULL keeps the raw count products and
      // NULL is multiplied by Z. Avoids a per-candidate division.
      double Z = chain.jumpCountSum + chain.jumpPriorMass;

      double total = 0.0;
      for (PositionIndex i = 0; i <= slen; ++i)
      {
        WordIndex e = nsrc[i];
        double count = 0;
        auto it = chain.lexCounts[e].find(f);
        if (it != chain.lexCounts[e].end())
          count = it->second;
        double w = (count + alphaLex) * lexDenomInv[e];
        if (stage == Ibm1Stage)
        {
          w *= i == 0 ? p0 : (1.0 - p0) / slen;
        }
        else if (i == 0)
        {
          // NULL: the two non-NULL neighbours connect directly (skip jump).
          double js = std::max(chain.jumpCounts[jumpBucket(aR - aaLeft)] + alphaJump, kMinJumpWeight);
          w *= p0 * js * Z;
        }
        else
        {
          int pos0 = (int)i - 1;
          double j1 = std::max(chain.jumpCounts[jumpBucket(pos0 - aaLeft)] + alphaJump, kMinJumpWeight);
          double j2 = std::max(chain.jumpCounts[jumpBucket(aR - pos0)] + alphaJump, kMinJumpWeight);
          w *= j1 * j2;
          if (stage == FertilityStage)
            w *= chain.fertRatioSampled[e][std::min((PositionIndex)(phi[i] + 1), MaxFertility)];
        }
        weights[i] = w;
        total += w;
      }

      PositionIndex iNew = SamplingUtils::sampleCategorical(weights, total, chain.rng);

      chain.alig[n][j - 1] = iNew;
      WordIndex eNew = nsrc[iNew];
      chain.lexCounts[eNew][f] += 1;
      chain.lexCountSum[eNew] += 1;
      recomputeInv(eNew);
      if (stage == FertilityStage)
        phi[iNew]++;

      // Re-add the chosen candidate's jump contribution to the live counts.
      if (stage != Ibm1Stage)
      {
        if (iNew == 0)
        {
          chain.jumpCounts[jumpBucket(aR - aaLeft)] += 1;
          chain.jumpCountSum += 1;
        }
        else
        {
          int p0 = (int)iNew - 1;
          chain.jumpCounts[jumpBucket(p0 - aaLeft)] += 1;
          chain.jumpCounts[jumpBucket(aR - p0)] += 1;
          chain.jumpCountSum += 2;
        }
      }

      if (accumulate)
      {
        chain.accumLexCounts[eNew][f] += 1;
        chain.accumLexCountSum[eNew] += 1;
      }

      if (iNew > 0)
        aaLeft = (int)iNew - 1;
    }
  }
}

void EflomalAlignmentModel::endTraining()
{
#pragma omp parallel for schedule(static) if(!deterministic)
  for (int s = 0; s < (int)chains.size(); ++s)
    normalizeChain(chains[(size_t)s]);

  // Rewire model-level tables to chain[0]'s trained tables.
  lexTable = chains[0].lexTable;
  jumpTable = chains[0].jumpTable;
  fertilityTable = chains[0].fertilityTable;

  // Emit training alignments (if enabled) and clear temporary state via the base
  // tail, run here while the chains' converged alignments and the corpus are
  // still resident (computeTrainingAlignments needs them before they are cleared).
  AlignmentModelBase::endTraining();
}

// Warm final-argmax alignments for the training corpus, computed once here while
// the chains' converged alignments and the corpus are still resident (cheap; one
// pass per pair). Stored per target token (1-based source, 0 = NULL), the same
// form getBestAlignment returns. Overrides the base getBestAlignment-based default
// with this cheaper, slightly more accurate training-time variant.
void EflomalAlignmentModel::computeTrainingAlignments()
{
  trainingAlignments.clear();
  if (!emitTrainingAlignments)
    return;

  // The corpus is the length-filtered subset in added order; map each corpus
  // position back to its sentence-handler pair index so trainingAlignments stays
  // indexed by pair index (matching getSentencePair), with filtered pairs empty.
  vector<unsigned int> sents = trainingPairSentenceIndices();
  trainingAlignments.assign(numSentencePairs(), {});
  size_t np = corpusSrc.size();
#pragma omp parallel for schedule(dynamic, 16) if(!deterministic)
  for (int n = 0; n < (int)np; ++n)
  {
    PositionIndex slen = (PositionIndex)corpusSrc[(size_t)n].size() - 1;
    vector<vector<double>> acc;
    accumulateWarmDecodeMarginal((size_t)n, acc);
    vector<PositionIndex> alig(acc.size(), 0);
    for (PositionIndex j = 0; j < (PositionIndex)acc.size(); ++j)
    {
      PositionIndex bestK = 0;
      double bestP = acc[j][0];
      for (PositionIndex k = 1; k <= slen; ++k)
        if (acc[j][k] > bestP)
        {
          bestP = acc[j][k];
          bestK = k;
        }
      alig[j] = (bestK == slen) ? 0 : (PositionIndex)(bestK + 1); // 0 = NULL, else 1-based source
    }
    // Scatter to the pair's handler index; indices are distinct, so no data race.
    if ((size_t)n < sents.size())
      trainingAlignments[sents[(size_t)n]] = std::move(alig);
  }
}

void EflomalAlignmentModel::normalizeChain(SamplerChain& chain)
{
  // Lexical counts are the variance-reduced marginal accumulated over the final
  // stage; the jump and fertility distributions are read off the final alignment.
  const vector<LexCountMap>& lex = chain.accumulated ? chain.accumLexCounts : chain.lexCounts;
  const vector<double>& lexSum = chain.accumulated ? chain.accumLexCountSum : chain.lexCountSum;

  computeJumpCounts(chain);
  computeFertilityCounts(chain);
  const vector<double>& jumps = chain.jumpCounts;
  double jumpSum = chain.jumpCountSum;
  const vector<vector<double>>& fert = chain.fertCounts;
  const vector<double>& fertSum = chain.fertCountSum;

  double trgVocabSize = (double)getTrgVocabSize();

  // Lexical table p(t|s).
  chain.lexTable->clear();
  for (WordIndex s = 0; s < (WordIndex)lex.size(); ++s)
  {
    if (lex[s].empty())
      continue;
    double denom = lexSum[s] + alphaLex * trgVocabSize;
    chain.lexTable->setDenominator(s, (float)log(denom));
    for (const auto& entry : lex[s])
      chain.lexTable->setNumerator(s, entry.first, (float)log(entry.second + alphaLex));
  }

  // Jump table p(delta).
  chain.jumpTable->setFromCounts(jumps, jumpSum, alphaJump, jumpWindow);

  // Fertility table p(phi|s).
  chain.fertilityTable->clear();
  double fertNorm = alphaFertility * (double)(MaxFertility + 1);
  for (WordIndex s = 0; s < (WordIndex)fert.size(); ++s)
  {
    if (fertSum[s] <= 0)
      continue;
    double denom = fertSum[s] + fertNorm;
    chain.fertilityTable->setDenominator(s, (float)log(denom));
    for (PositionIndex phi = 0; phi <= MaxFertility; ++phi)
      chain.fertilityTable->setNumerator(s, phi, (float)log(fert[s][phi] + alphaFertility));
  }
}

Prob EflomalAlignmentModel::translationProb(WordIndex s, WordIndex t)
{
  return translationLogProb(s, t).get_p();
}

LgProb EflomalAlignmentModel::translationLogProb(WordIndex s, WordIndex t)
{
  bool found;
  double numer = lexTable->getNumerator(s, t, found);
  if (!found)
    return SmoothingLogProb;
  double denom = lexTable->getDenominator(s, found);
  if (!found)
    return SmoothingLogProb;
  return numer - denom;
}

Prob EflomalAlignmentModel::jumpProb(int offset)
{
  return jumpTable->prob(offset);
}

LgProb EflomalAlignmentModel::jumpLogProb(int offset)
{
  return jumpTable->logProb(offset);
}

Prob EflomalAlignmentModel::fertilityProb(WordIndex s, PositionIndex phi)
{
  return fertilityLogProb(s, phi).get_p();
}

LgProb EflomalAlignmentModel::fertilityLogProb(WordIndex s, PositionIndex phi)
{
  bool found;
  double numer = fertilityTable->getNumerator(s, std::min(phi, MaxFertility), found);
  if (!found)
    return SmoothingLogProb;
  double denom = fertilityTable->getDenominator(s, found);
  if (!found)
    return SmoothingLogProb;
  return numer - denom;
}

Prob EflomalAlignmentModel::sentenceLengthProb(unsigned int slen, unsigned int tlen)
{
  return sentenceLengthLogProb(slen, tlen).get_p();
}

LgProb EflomalAlignmentModel::sentenceLengthLogProb(unsigned int slen, unsigned int tlen)
{
  unsigned int sentenceCount = numSentencePairs();
  double meanSrcLenMultiplier = totLenRatio == 0 || sentenceCount == 0 ? 1.0 : totLenRatio / sentenceCount;
  return Md::log_poisson(tlen, 0.05 + slen * meanSrcLenMultiplier);
}

Prob EflomalAlignmentModel::hmmAlignmentProb(PositionIndex prevI, PositionIndex slen, PositionIndex i)
{
  if (i == 0)
    return p0;
  if (prevI == 0)
    return (1.0 - p0) / slen;
  return (1.0 - p0) * jumpTable->prob((int)i - (int)prevI);
}

LgProb EflomalAlignmentModel::hmmAlignmentLogProb(PositionIndex prevI, PositionIndex slen, PositionIndex i)
{
  if (i == 0)
    return log(p0);
  if (prevI == 0)
    return log((1.0 - p0) / slen);
  return log(1.0 - p0) + jumpTable->logProb((int)i - (int)prevI);
}

void EflomalAlignmentModel::decodeMarginalFromTables(const MemoryLexTable& lex, const EflomalJumpTable& jump,
                                                      const FertilityTable& fert, unsigned int chainSeed,
                                                      const vector<WordIndex>& nsrc, const vector<WordIndex>& trg,
                                                      vector<vector<double>>& acc,
                                                      const vector<PositionIndex>* warmStart)
{
  PositionIndex slen = (PositionIndex)nsrc.size() - 1;
  PositionIndex tlen = (PositionIndex)trg.size();
  acc.assign(tlen, vector<double>(slen + 1, 0.0));
  if (tlen == 0 || slen == 0)
    return;

  // Helpers to read table probabilities without going through virtual dispatch.
  auto lexProb = [&](WordIndex s, WordIndex t) -> double {
    bool found;
    double numer = lex.getNumerator(s, t, found);
    if (!found)
      return SmoothingProb;
    double denom = lex.getDenominator(s, found);
    if (!found)
      return SmoothingProb;
    return std::exp(numer - denom);
  };
  auto fertProb = [&](WordIndex s, PositionIndex phi) -> double {
    bool found;
    double numer = fert.getNumerator(s, std::min(phi, MaxFertility), found);
    if (!found)
      return SmoothingProb;
    double denom = fert.getDenominator(s, found);
    if (!found)
      return SmoothingProb;
    return std::exp(numer - denom);
  };

  // Precompute the lexical emissions p(f|e), the linearized jump distribution and
  // per-source fertility ratios so the inner sampling loop is free of exp/log.
  // Source positions are 1..slen (nsrc[p]); 0-based source index is p-1. The
  // nearest-non-NULL neighbours use a virtual BOS at -1 and EOS at slen, exactly
  // as in eflomal's get_jump_index scheme.
  vector<vector<double>> emission(slen + 1, vector<double>(tlen));
  for (PositionIndex p = 0; p <= slen; ++p)
    for (PositionIndex j = 0; j < tlen; ++j)
      emission[p][j] = lexProb(nsrc[p], trg[j]);

  vector<double> jumpLin((size_t)2 * jumpWindow + 1);
  for (int b = 0; b < (int)jumpLin.size(); ++b)
    jumpLin[b] = (double)jump.prob(b - jumpWindow);

  vector<vector<double>> fertRatio(slen + 1, vector<double>(MaxFertility + 1, 1.0));
  for (PositionIndex p = 1; p <= slen; ++p)
  {
    WordIndex e = nsrc[p];
    for (PositionIndex phi = 0; phi <= MaxFertility; ++phi)
    {
      double cur = fertProb(e, phi);
      double nxt = fertProb(e, std::min((PositionIndex)(phi + 1), MaxFertility));
      fertRatio[p][phi] = cur > 0 ? nxt / cur : 1.0;
    }
  }

  // acc[j][k] (the output) accumulates the marginal: k in [0, slen-1] -> source
  // position k+1, k == slen -> NULL.
  vector<PositionIndex> links(tlen, 0);
  vector<int> phi(slen + 1, 0); // per-source fertility counts within this decode chain
  vector<int> aaRight(tlen, 0);
  vector<double> ps(slen + 1, 0.0);

  // Local, deterministic RNG seeded per chain: decode is reproducible per pair
  // and thread-safe (no shared state with other threads or model fields).
  std::mt19937_64 engine(chainSeed);
  // Guard: if burn-in is >= decodeIters, no samples would be accumulated and
  // every target would map to NULL. Clamp so at least the last iteration counts.
  // Warm decode (warmStart != null): the trained chain has already converged, so
  // seed links from its alignment and do a single accumulate pass (one decode
  // chain, no burn-in) rather than cold-starting from the diagonal and re-warming.
  int nDecodeChains = warmStart ? 1 : decodeSamplers;
  int nIters = warmStart ? 1 : decodeIters;
  int effectiveBurnIn = warmStart ? 0 : std::max(0, std::min(decodeBurnIn, decodeIters - 1));

  for (int chain = 0; chain < nDecodeChains; ++chain)
  {
    if (warmStart && warmStart->size() == (size_t)tlen)
    {
      links = *warmStart;
    }
    else
    {
      for (PositionIndex j = 1; j <= tlen; ++j)
      {
        PositionIndex p = (PositionIndex)llround((double)j * slen / tlen);
        if (p < 1)
          p = 1;
        if (p > slen)
          p = slen;
        links[j - 1] = p;
      }
    }
    fill(phi.begin(), phi.end(), 0);
    for (PositionIndex j = 0; j < tlen; ++j)
      phi[links[j]]++;

    for (int it = 0; it < nIters; ++it)
    {
      bool accumulate = it >= effectiveBurnIn;

      int aa = (int)slen;
      for (PositionIndex j = tlen; j >= 1; --j)
      {
        aaRight[j - 1] = aa;
        if (links[j - 1] > 0)
          aa = (int)links[j - 1] - 1;
      }

      int aL = -1;
      for (PositionIndex j = 0; j < tlen; ++j)
      {
        PositionIndex oldi = links[j];
        if (oldi > 0)
          phi[oldi]--;
        int aR = aaRight[j];

        double sum = 0.0;
        for (PositionIndex p = 1; p <= slen; ++p)
        {
          int pos0 = (int)p - 1;
          double w = emission[p][j] * jumpLin[jumpBucket(pos0 - aL)] * jumpLin[jumpBucket(aR - pos0)]
                   * fertRatio[p][std::min(phi[p], (int)MaxFertility)];
          sum += w;
          ps[p - 1] = sum;
        }
        sum += p0 * emission[0][j] * jumpLin[jumpBucket(aR - aL)];
        ps[slen] = sum;

        if (accumulate && sum > 0.0)
        {
          double scale = 1.0 / sum;
          acc[j][0] += ps[0] * scale;
          for (PositionIndex k = 1; k <= slen; ++k)
            acc[j][k] += (ps[k] - ps[k - 1]) * scale;
        }

        PositionIndex k = SamplingUtils::sampleCumulative(ps, sum, engine, slen + 1);
        PositionIndex newi = k == slen ? 0 : k + 1;
        links[j] = newi;
        if (newi > 0)
        {
          phi[newi]++;
          aL = (int)newi - 1;
        }
      }
    }
  }
}

// Per-target argmax of a (possibly summed over chains) decode marginal; ties
// favour the lowest source index, matching eflomal's argmax pass.
static void argmaxMarginal(const vector<vector<double>>& acc, PositionIndex slen, vector<PositionIndex>& alignment)
{
  PositionIndex tlen = (PositionIndex)acc.size();
  alignment.assign(tlen, 0);
  for (PositionIndex j = 0; j < tlen; ++j)
  {
    PositionIndex bestK = 0;
    double bestP = acc[j][0];
    for (PositionIndex k = 1; k <= slen; ++k)
    {
      if (acc[j][k] > bestP)
      {
        bestP = acc[j][k];
        bestK = k;
      }
    }
    alignment[j] = bestK == slen ? 0 : bestK + 1;
  }
}

void EflomalAlignmentModel::sampleDecode(const vector<WordIndex>& nsrc, const vector<WordIndex>& trg,
                                         vector<PositionIndex>& alignment)
{
  PositionIndex slen = (PositionIndex)nsrc.size() - 1;
  PositionIndex tlen = (PositionIndex)trg.size();
  alignment.assign(tlen, 0);
  if (tlen == 0 || slen == 0)
    return;

  vector<vector<double>> acc;
  for (const auto& chain : chains)
  {
    vector<vector<double>> chainAcc;
    decodeMarginalFromTables(*chain.lexTable, *chain.jumpTable, *chain.fertilityTable, chain.chainSeed, nsrc, trg,
                             chainAcc);
    if (acc.empty())
    {
      acc = std::move(chainAcc);
    }
    else
    {
      for (size_t j = 0; j < chainAcc.size() && j < acc.size(); ++j)
        for (size_t k = 0; k < chainAcc[j].size() && k < acc[j].size(); ++k)
          acc[j][k] += chainAcc[j][k];
    }
  }
  argmaxMarginal(acc, slen, alignment);
}

void EflomalAlignmentModel::accumulateDecodeMarginal(const vector<WordIndex>& srcSentence,
                                                     const vector<WordIndex>& trgSentence,
                                                     vector<vector<double>>& accOut)
{
  if (!(sentenceLengthIsOk(srcSentence) && sentenceLengthIsOk(trgSentence)))
    return;
  vector<WordIndex> nsrc = addNullWordToWidxVec(srcSentence);

  for (const auto& chain : chains)
  {
    vector<vector<double>> acc;
    decodeMarginalFromTables(*chain.lexTable, *chain.jumpTable, *chain.fertilityTable, chain.chainSeed, nsrc,
                             trgSentence, acc);
    if (accOut.empty())
    {
      accOut = std::move(acc);
    }
    else
    {
      for (size_t j = 0; j < acc.size() && j < accOut.size(); ++j)
        for (size_t k = 0; k < acc[j].size() && k < accOut[j].size(); ++k)
          accOut[j][k] += acc[j][k];
    }
  }
}

// Warm decode of training-corpus pair n: each chain's converged alignment seeds a
// single accumulate pass against that chain's tables; marginals sum across chains.
// Called from endTraining (before clearTempVars) while corpus + chain.alig are
// still resident, to produce the persisted training alignments.
void EflomalAlignmentModel::accumulateWarmDecodeMarginal(size_t n, vector<vector<double>>& accOut)
{
  if (n >= corpusSrc.size())
    return;
  const vector<WordIndex>& nsrc = corpusSrc[n]; // already null-extended at index 0
  const vector<WordIndex>& trg = corpusTrg[n];
  for (const auto& chain : chains)
  {
    if (n >= chain.alig.size())
      return;
    vector<vector<double>> acc;
    decodeMarginalFromTables(*chain.lexTable, *chain.jumpTable, *chain.fertilityTable, chain.chainSeed, nsrc, trg, acc,
                             &chain.alig[n]);
    if (accOut.empty())
    {
      accOut = std::move(acc);
    }
    else
    {
      for (size_t j = 0; j < acc.size() && j < accOut.size(); ++j)
        for (size_t k = 0; k < acc[j].size() && k < accOut[j].size(); ++k)
          accOut[j][k] += acc[j][k];
    }
  }
}

double EflomalAlignmentModel::scoreAlignment(const vector<WordIndex>& nsrc, const vector<WordIndex>& trg,
                                             const vector<PositionIndex>& alignment)
{
  PositionIndex slen = (PositionIndex)nsrc.size() - 1;
  PositionIndex tlen = (PositionIndex)trg.size();
  double logProb = (double)sentenceLengthLogProb(slen, tlen);

  PositionIndex prev = 0;
  for (PositionIndex j = 1; j <= tlen; ++j)
  {
    PositionIndex i = alignment[j - 1];
    logProb += (double)translationLogProb(nsrc[i], trg[j - 1]) + (double)hmmAlignmentLogProb(prev, slen, i);
    prev = i;
  }

  vector<int> phi(slen + 1, 0);
  for (PositionIndex j = 1; j <= tlen; ++j)
    phi[alignment[j - 1]]++;
  for (PositionIndex i = 1; i <= slen; ++i)
    logProb += (double)fertilityLogProb(nsrc[i], std::min((PositionIndex)phi[i], MaxFertility));

  return logProb;
}

LgProb EflomalAlignmentModel::getBestAlignment(const vector<WordIndex>& srcSentence,
                                               const vector<WordIndex>& trgSentence,
                                               vector<PositionIndex>& bestAlignment)
{
  if (sentenceLengthIsOk(srcSentence) && sentenceLengthIsOk(trgSentence))
  {
    vector<WordIndex> nsrc = addNullWordToWidxVec(srcSentence);
    sampleDecode(nsrc, trgSentence, bestAlignment);
    return scoreAlignment(nsrc, trgSentence, bestAlignment);
  }
  bestAlignment.assign(trgSentence.size(), 0);
  return SMALL_LG_NUM;
}

LgProb EflomalAlignmentModel::computeLogProb(const vector<WordIndex>& srcSentence, const vector<WordIndex>& trgSentence,
                                             const WordAlignmentMatrix& aligMatrix, int /*verbose*/)
{
  vector<WordIndex> nsrc = addNullWordToWidxVec(srcSentence);
  vector<PositionIndex> alig;
  aligMatrix.getAligVec(alig);
  if (alig.size() != trgSentence.size())
    return SmoothingLogProb;
  return scoreAlignment(nsrc, trgSentence, alig);
}

LgProb EflomalAlignmentModel::computeSumLogProb(const vector<WordIndex>& srcSentence,
                                                const vector<WordIndex>& trgSentence, int /*verbose*/)
{
  vector<WordIndex> nsrc = addNullWordToWidxVec(srcSentence);
  vector<PositionIndex> alig;
  sampleDecode(nsrc, trgSentence, alig);
  return scoreAlignment(nsrc, trgSentence, alig);
}

bool EflomalAlignmentModel::getEntriesForSource(WordIndex s, NbestTableNode<WordIndex>& trgtn)
{
  set<WordIndex> transSet;
  if (!lexTable->getTransForSource(s, transSet))
    return false;
  trgtn.clear();
  for (WordIndex t : transSet)
    trgtn.insert(translationProb(s, t), t);
  return true;
}

pair<double, double> EflomalAlignmentModel::loglikelihoodForPairRange(pair<unsigned int, unsigned int> sentPairRange,
                                                                      int verbosity)
{
  double loglikelihood = 0;
  unsigned int numSents = 0;
  for (unsigned int n = sentPairRange.first; n <= sentPairRange.second; ++n)
  {
    vector<WordIndex> src = getSrcSent(n);
    vector<WordIndex> trg = getTrgSent(n);
    loglikelihood += (double)computeSumLogProb(src, trg, verbosity);
    ++numSents;
  }
  return make_pair(loglikelihood, numSents == 0 ? 0 : loglikelihood / (double)numSents);
}

void EflomalAlignmentModel::loadConfig(const YAML::Node& config)
{
  AlignmentModelBase::loadConfig(config);
  seed = config["seed"].as<unsigned int>();
  if (config["numSamplers"])
    numSamplers = config["numSamplers"].as<int>();
  deterministic = config["deterministic"].as<bool>();
  ibm1Iters = config["ibm1Iters"].as<int>();
  hmmIters = config["hmmIters"].as<int>();
  fertilityIters = config["fertilityIters"].as<int>();
  jumpWindow = config["jumpWindow"].as<int>();
  eflomalLexNorm = config["eflomalLexNorm"].as<bool>();
  autoIterations = config["autoIterations"].as<bool>();
  decodeSamplers = config["decodeSamplers"].as<int>();
  decodeIters = config["decodeIters"].as<int>();
  decodeBurnIn = config["decodeBurnIn"].as<int>();
  decodeParamsExplicit = true; // restored from config; don't re-derive on retrain
  alphaLex = config["alphaLex"].as<double>();
  alphaJump = config["alphaJump"].as<double>();
  alphaFertility = config["alphaFertility"].as<double>();
  p0 = config["p0"].as<double>();
}

void EflomalAlignmentModel::createConfig(YAML::Emitter& out)
{
  AlignmentModelBase::createConfig(out);
  out << YAML::Key << "seed" << YAML::Value << seed;
  out << YAML::Key << "numSamplers" << YAML::Value << numSamplers;
  out << YAML::Key << "deterministic" << YAML::Value << deterministic;
  out << YAML::Key << "ibm1Iters" << YAML::Value << ibm1Iters;
  out << YAML::Key << "hmmIters" << YAML::Value << hmmIters;
  out << YAML::Key << "fertilityIters" << YAML::Value << fertilityIters;
  out << YAML::Key << "jumpWindow" << YAML::Value << jumpWindow;
  out << YAML::Key << "eflomalLexNorm" << YAML::Value << eflomalLexNorm;
  out << YAML::Key << "autoIterations" << YAML::Value << autoIterations;
  out << YAML::Key << "decodeSamplers" << YAML::Value << decodeSamplers;
  out << YAML::Key << "decodeIters" << YAML::Value << decodeIters;
  out << YAML::Key << "decodeBurnIn" << YAML::Value << decodeBurnIn;
  out << YAML::Key << "alphaLex" << YAML::Value << alphaLex;
  out << YAML::Key << "alphaJump" << YAML::Value << alphaJump;
  out << YAML::Key << "alphaFertility" << YAML::Value << alphaFertility;
  out << YAML::Key << "p0" << YAML::Value << p0;
}

bool EflomalAlignmentModel::loadParams(const string& filename)
{
  ifstream in(filename);
  if (!in)
    return THOT_ERROR;
  in >> totLenRatio;
  return THOT_OK;
}

bool EflomalAlignmentModel::printParams(const string& filename) const
{
  ofstream out(filename);
  if (!out)
    return THOT_ERROR;
  out << setprecision(numeric_limits<double>::max_digits10) << totLenRatio;
  return THOT_OK;
}

bool EflomalAlignmentModel::load(const char* prefFileName, int verbose)
{
  if (prefFileName[0] == 0)
    return THOT_ERROR;

  // AlignmentModelBase::load reads the YAML config (calling loadConfig, which
  // sets numSamplers) before we try to load the per-chain table files.
  if (AlignmentModelBase::load(prefFileName, verbose) == THOT_ERROR)
    return THOT_ERROR;

  if (verbose)
    cerr << "Loading Eflomal model data..." << endl;

  string pref = string(prefFileName);
  chains.resize((size_t)numSamplers);
  for (int s = 0; s < numSamplers; ++s)
  {
    // File names: single-chain uses old suffix-free names for backward
    // compatibility; multi-chain appends ".0", ".1", ... to each file.
    string suffix = numSamplers == 1 ? "" : "." + to_string(s);
    chains[(size_t)s].chainSeed = seed + (unsigned int)s * 2654435761u;
    chains[(size_t)s].lexTable = make_shared<MemoryLexTable>();
    chains[(size_t)s].jumpTable = make_shared<EflomalJumpTable>();
    chains[(size_t)s].fertilityTable = make_shared<FertilityTable>();
    if (chains[(size_t)s].lexTable->load((pref + ".efl_lexnd" + suffix).c_str(), verbose) == THOT_ERROR)
      return THOT_ERROR;
    if (chains[(size_t)s].jumpTable->load((pref + ".efl_jumpnd" + suffix).c_str(), verbose) == THOT_ERROR)
      return THOT_ERROR;
    if (chains[(size_t)s].fertilityTable->load((pref + ".efl_fertnd" + suffix).c_str(), verbose) == THOT_ERROR)
      return THOT_ERROR;
  }

  // Wire model-level query tables to chain[0].
  lexTable = chains[0].lexTable;
  jumpTable = chains[0].jumpTable;
  fertilityTable = chains[0].fertilityTable;

  if (loadParams(pref + ".params") == THOT_ERROR)
    return THOT_ERROR;

  // The persisted training alignments (".aligns") are restored by
  // AlignmentModelBase::load above.

  return THOT_OK;
}

bool EflomalAlignmentModel::print(const char* prefFileName, int verbose)
{
  if (AlignmentModelBase::print(prefFileName, verbose) == THOT_ERROR)
    return THOT_ERROR;

  string pref = string(prefFileName);
  for (int s = 0; s < (int)chains.size(); ++s)
  {
    // Single-chain uses suffix-free names (backward compat); multi-chain
    // appends ".0", ".1", ... so each chain's tables are stored separately.
    string suffix = chains.size() == 1 ? "" : "." + to_string(s);
    if (chains[(size_t)s].lexTable->print((pref + ".efl_lexnd" + suffix).c_str()) == THOT_ERROR)
      return THOT_ERROR;
    if (chains[(size_t)s].jumpTable->print((pref + ".efl_jumpnd" + suffix).c_str()) == THOT_ERROR)
      return THOT_ERROR;
    if (chains[(size_t)s].fertilityTable->print((pref + ".efl_fertnd" + suffix).c_str()) == THOT_ERROR)
      return THOT_ERROR;
  }

  if (printParams(pref + ".params") == THOT_ERROR)
    return THOT_ERROR;

  // The training alignments (".aligns") are persisted by
  // AlignmentModelBase::print above.

  return THOT_OK;
}

void EflomalAlignmentModel::clearSentenceLengthModel()
{
  totLenRatio = 0;
}

void EflomalAlignmentModel::clearTempVars()
{
  iter = 0;
  corpusSrc.clear();
  corpusTrg.clear();
  // Per-chain training state is freed; the trained tables (and the warm training
  // alignments, if computed) persist. endTraining computes those alignments before
  // calling this, while the corpus and chain.alig are still resident.
  for (auto& chain : chains)
  {
    chain.alig.clear();
    chain.lexCounts.clear();
    chain.lexCountSum.clear();
    chain.jumpCounts.clear();
    chain.jumpCountSum = 0;
    chain.fertCounts.clear();
    chain.fertCountSum.clear();
    chain.fertRatioSampled.clear();
    chain.lexPriorMass = chain.jumpPriorMass = chain.fertPriorMass = 0;
    chain.accumLexCounts.clear();
    chain.accumLexCountSum.clear();
    chain.accumulated = false;
  }
}

void EflomalAlignmentModel::clear()
{
  AlignmentModelBase::clear();
  lexTable = make_shared<MemoryLexTable>();
  jumpTable = make_shared<EflomalJumpTable>();
  fertilityTable = make_shared<FertilityTable>();
  chains.clear();
  // Restore the single default chain so query/decode methods work without training.
  SamplerChain ch;
  ch.chainSeed = DefaultSeed;
  ch.lexTable = lexTable;
  ch.jumpTable = jumpTable;
  ch.fertilityTable = fertilityTable;
  chains.push_back(std::move(ch));
  corpusSrc.clear();
  corpusTrg.clear();
  totLenRatio = 0;
  iter = 0;
  seed = DefaultSeed;
  numSamplers = 3;
  deterministic = false;
  emitTrainingAlignments = false;
  trainingAlignments.clear();
  ibm1Iters = DefaultIbm1Iters;
  hmmIters = DefaultHmmIters;
  fertilityIters = DefaultFertilityIters;
  jumpWindow = DefaultJumpWindow;
  eflomalLexNorm = true;
  autoIterations = true;
  decodeSamplers = DefaultDecodeSamplers;
  decodeIters = DefaultDecodeIters;
  decodeBurnIn = DefaultDecodeBurnIn;
  decodeParamsExplicit = false;
  alphaLex = DefaultAlphaLex;
  alphaJump = DefaultAlphaJump;
  alphaFertility = DefaultAlphaFertility;
  p0 = DefaultP0;
}






