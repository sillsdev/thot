#include "Ibm2AligModel.h"

#include "nlp_common/ErrorDefs.h"
#include "sw_models/Ibm2AligModel.h"
#include "sw_models/SwDefs.h"

using namespace std;

Ibm2AligModel::Ibm2AligModel() : alignmentTable{make_shared<AlignmentTable>()}
{
}

Ibm2AligModel::Ibm2AligModel(Ibm1AligModel& model) : Ibm1AligModel{model}, alignmentTable{make_shared<AlignmentTable>()}
{
}

Ibm2AligModel::Ibm2AligModel(Ibm2AligModel& model) : Ibm1AligModel{model}, alignmentTable{model.alignmentTable}
{
}

void Ibm2AligModel::initTargetWord(const vector<WordIndex>& nsrc, const vector<WordIndex>& trg, PositionIndex j)
{
  Ibm1AligModel::initTargetWord(nsrc, trg, j);

  PositionIndex slen = (PositionIndex)nsrc.size() - 1;
  PositionIndex tlen = (PositionIndex)trg.size();

  alignmentTable->reserveSpace(j, slen, tlen);

  AlignmentKey key{j, slen, tlen};
  AlignmentCountsElem& elem = alignmentCounts[key];
  if (elem.size() < nsrc.size())
    elem.resize(nsrc.size(), 0);
}

double Ibm2AligModel::getCountNumerator(const vector<WordIndex>& nsrcSent, const vector<WordIndex>& trgSent,
                                        unsigned int i, unsigned int j)
{
  double d = Ibm1AligModel::getCountNumerator(nsrcSent, trgSent, i, j);
  d = d * double{aProb(j, (PositionIndex)nsrcSent.size() - 1, (PositionIndex)trgSent.size(), i)};
  return d;
}

void Ibm2AligModel::incrementWordPairCounts(const vector<WordIndex>& nsrc, const vector<WordIndex>& trg,
                                            PositionIndex i, PositionIndex j, double count)
{
  Ibm1AligModel::incrementWordPairCounts(nsrc, trg, i, j, count);

  AlignmentKey key{j, (PositionIndex)nsrc.size() - 1, (PositionIndex)trg.size()};

#pragma omp atomic
  alignmentCounts[key][i] += count;
}

void Ibm2AligModel::batchMaximizeProbs()
{
  Ibm1AligModel::batchMaximizeProbs();

#pragma omp parallel for schedule(dynamic)
  for (int asIndex = 0; asIndex < (int)alignmentCounts.size(); ++asIndex)
  {
    double denom = 0;
    const pair<AlignmentKey, AlignmentCountsElem>& p = alignmentCounts.getAt(asIndex);
    const AlignmentKey& key = p.first;
    AlignmentCountsElem& elem = const_cast<AlignmentCountsElem&>(p.second);
    for (PositionIndex i = 0; i < elem.size(); ++i)
    {
      double numer = elem[i];
      denom += numer;
      float logNumer = (float)log(numer);
      alignmentTable->setNumerator(key.j, key.slen, key.tlen, i, logNumer);
      elem[i] = 0.0;
    }
    if (denom == 0)
      denom = 1;
    float logDenom = (float)log(denom);
    alignmentTable->setDenominator(key.j, key.slen, key.tlen, logDenom);
  }
}

Prob Ibm2AligModel::aProb(PositionIndex j, PositionIndex slen, PositionIndex tlen, PositionIndex i)
{
  double logProb = unsmoothed_aProb(j, slen, tlen, i);
  double prob = logProb == SMALL_LG_NUM ? 1.0 / (slen + 1) : exp(logProb);
  return max(prob, SW_PROB_SMOOTH);
}

LgProb Ibm2AligModel::logaProb(PositionIndex j, PositionIndex slen, PositionIndex tlen, PositionIndex i)
{
  double logProb = unsmoothed_aProb(j, slen, tlen, i);
  if (logProb == SMALL_LG_NUM)
    logProb = log(1.0 / (slen + 1));
  return max(logProb, SW_LOG_PROB_SMOOTH);
}

double Ibm2AligModel::unsmoothed_aProb(PositionIndex j, PositionIndex slen, PositionIndex tlen, PositionIndex i)
{
  return exp(unsmoothed_logaProb(j, slen, tlen, i));
}

double Ibm2AligModel::unsmoothed_logaProb(PositionIndex j, PositionIndex slen, PositionIndex tlen, PositionIndex i)
{
  bool found;
  double numer = alignmentTable->getNumerator(j, slen, tlen, i, found);
  if (found)
  {
    // aligNumer for pair as,i exists
    double denom = alignmentTable->getDenominator(j, slen, tlen, found);
    if (found)
      return numer - denom;
  }
  return SMALL_LG_NUM;
}

LgProb Ibm2AligModel::obtainBestAlignment(const vector<WordIndex>& srcSentIndexVector,
                                          const vector<WordIndex>& trgSentIndexVector, WordAligMatrix& bestWaMatrix)
{
  vector<PositionIndex> bestAlig;
  LgProb lgProb = sentLenLgProb((PositionIndex)srcSentIndexVector.size(), (PositionIndex)trgSentIndexVector.size());
  lgProb += lexAligM2LpForBestAlig(addNullWordToWidxVec(srcSentIndexVector), trgSentIndexVector, bestAlig);

  bestWaMatrix.init((PositionIndex)srcSentIndexVector.size(), (PositionIndex)trgSentIndexVector.size());
  bestWaMatrix.putAligVec(bestAlig);

  return lgProb;
}

LgProb Ibm2AligModel::calcLgProbForAlig(const vector<WordIndex>& sSent, const vector<WordIndex>& tSent,
                                        const WordAligMatrix& aligMatrix, int verbose)
{
  PositionIndex i;

  vector<PositionIndex> alig;
  aligMatrix.getAligVec(alig);

  if (verbose)
  {
    for (i = 0; i < sSent.size(); ++i)
      cerr << sSent[i] << " ";
    cerr << "\n";
    for (i = 0; i < tSent.size(); ++i)
      cerr << tSent[i] << " ";
    cerr << "\n";
    for (i = 0; i < alig.size(); ++i)
      cerr << alig[i] << " ";
    cerr << "\n";
  }
  if (tSent.size() != alig.size())
  {
    cerr << "Error: the sentence t and the alignment vector have not the same size." << endl;
    return THOT_ERROR;
  }
  else
  {
    return calcIbm2LgProbForAlig(addNullWordToWidxVec(sSent), tSent, alig, verbose);
  }
}

LgProb Ibm2AligModel::calcIbm2LgProbForAlig(const vector<WordIndex>& nsSent, const vector<WordIndex>& tSent,
                                            const vector<PositionIndex>& alig, int verbose)
{
  PositionIndex slen = (PositionIndex)nsSent.size() - 1;
  PositionIndex tlen = (PositionIndex)tSent.size();

  if (verbose)
    cerr << "Obtaining IBM Model 2 logprob...\n";

  LgProb lgProb = 0;
  for (PositionIndex j = 1; j <= alig.size(); ++j)
  {
    Prob p = pts(nsSent[alig[j]], tSent[j - 1]);
    if (verbose)
      cerr << "t(" << tSent[j - 1] << "|" << nsSent[alig[j - 1]] << ")= " << p << " ; logp=" << (double)log((double)p)
           << endl;
    lgProb = lgProb + (double)log((double)p);

    p = aProb(j, slen, tlen, alig[j - 1]);
    lgProb = lgProb + (double)log((double)p);
  }
  return lgProb;
}

LgProb Ibm2AligModel::calcLgProb(const vector<WordIndex>& sSent, const vector<WordIndex>& tSent, int verbose)
{
  return calcSumIbm2LgProb(addNullWordToWidxVec(sSent), tSent, verbose);
}

LgProb Ibm2AligModel::calcSumIbm2LgProb(const vector<WordIndex>& nsSent, const vector<WordIndex>& tSent, int verbose)
{
  PositionIndex slen = (PositionIndex)nsSent.size() - 1;
  PositionIndex tlen = (PositionIndex)tSent.size();
  Prob sump;
  LgProb lexAligContrib;

  if (verbose)
    cerr << "Obtaining Sum IBM Model 2 logprob...\n";

  LgProb lgProb = sentLenLgProb(slen, tlen);
  if (verbose)
    cerr << "- lenLgProb(tlen=" << tSent.size() << " | slen=" << slen << ")= " << sentLenLgProb(slen, tlen) << endl;

  lexAligContrib = 0;
  for (PositionIndex j = 1; j <= tSent.size(); ++j)
  {
    sump = 0;
    for (PositionIndex i = 0; i < nsSent.size(); ++i)
    {
      sump += pts(nsSent[i], tSent[j - 1]) * aProb(j, slen, tlen, i);
      if (verbose == 2)
      {
        cerr << "t( " << tSent[j - 1] << " | " << nsSent[i] << " )= " << pts(nsSent[i], tSent[j - 1]) << endl;
        cerr << "a( " << i << "| j=" << j << ", slen=" << slen << ", tlen=" << tlen << ")= " << aProb(j, slen, tlen, i)
             << endl;
      }
    }
    lexAligContrib += (double)log((double)sump);
    if (verbose)
      cerr << "- sump(j=" << j << ")= " << sump << endl;
    if (verbose == 2)
      cerr << endl;
  }

  if (verbose)
    cerr << "- Lexical plus alignment contribution= " << lexAligContrib << endl;
  lgProb += lexAligContrib;

  return lgProb;
}

bool Ibm2AligModel::load(const char* prefFileName, int verbose)
{
  if (prefFileName[0] != 0)
  {
    bool retVal;

    // Load IBM 1 Model data
    retVal = Ibm1AligModel::load(prefFileName, verbose);
    if (retVal == THOT_ERROR)
      return THOT_ERROR;

    if (verbose)
      cerr << "Loading incremental IBM 2 Model data..." << endl;

    // Load file with alignment nd values
    string aligNumDenFile = prefFileName;
    aligNumDenFile = aligNumDenFile + ".ibm2_alignd";
    retVal = alignmentTable->load(aligNumDenFile.c_str(), verbose);
    if (retVal == THOT_ERROR)
      return THOT_ERROR;

    return THOT_OK;
  }
  else
    return THOT_ERROR;
}

bool Ibm2AligModel::print(const char* prefFileName, int verbose)
{
  bool retVal;

  // Print IBM 1 Model data
  retVal = Ibm1AligModel::print(prefFileName);
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  // Print file with alignment nd values
  string aligNumDenFile = prefFileName;
  aligNumDenFile = aligNumDenFile + ".ibm2_alignd";
  retVal = alignmentTable->print(aligNumDenFile.c_str());
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  return THOT_OK;
}

LgProb Ibm2AligModel::lexAligM2LpForBestAlig(const vector<WordIndex>& nSrcSentIndexVector,
                                             const vector<WordIndex>& trgSentIndexVector,
                                             vector<PositionIndex>& bestAlig)
{
  // Initialize variables
  PositionIndex slen = (PositionIndex)nSrcSentIndexVector.size() - 1;
  PositionIndex tlen = (PositionIndex)trgSentIndexVector.size();
  LgProb aligLgProb = 0;
  bestAlig.clear();

  for (PositionIndex j = 1; j <= trgSentIndexVector.size(); ++j)
  {
    unsigned int best_i = 0;
    LgProb max_lp = -FLT_MAX;
    for (unsigned int i = 0; i < nSrcSentIndexVector.size(); ++i)
    {
      // lexical logprobability
      LgProb lp = log((double)pts(nSrcSentIndexVector[i], trgSentIndexVector[j - 1]));
      // alignment logprobability
      lp += log((double)aProb(j, slen, tlen, i));

      if (max_lp <= lp)
      {
        max_lp = lp;
        best_i = i;
      }
    }
    // Add contribution
    aligLgProb = aligLgProb + max_lp;
    // Add word alignment
    bestAlig.push_back(best_i);
  }
  return aligLgProb;
}

void Ibm2AligModel::clear()
{
  Ibm1AligModel::clear();
  alignmentTable->clear();
}

void Ibm2AligModel::clearTempVars()
{
  Ibm1AligModel::clearTempVars();
  alignmentCounts.clear();
}
