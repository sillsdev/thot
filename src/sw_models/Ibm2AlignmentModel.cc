
#include "sw_models/Ibm2AlignmentModel.h"

#include "nlp_common/ErrorDefs.h"
#include "sw_models/SwDefs.h"

using namespace std;

Ibm2AlignmentModel::Ibm2AlignmentModel() : alignmentTable{make_shared<AlignmentTable>()}
{
}

Ibm2AlignmentModel::Ibm2AlignmentModel(Ibm1AlignmentModel& model)
    : Ibm1AlignmentModel{model}, alignmentTable{make_shared<AlignmentTable>()}
{
}

Ibm2AlignmentModel::Ibm2AlignmentModel(Ibm2AlignmentModel& model)
    : Ibm1AlignmentModel{model}, compactAlignmentTable{model.compactAlignmentTable}, alignmentTable{
                                                                                         model.alignmentTable}
{
}

bool Ibm2AlignmentModel::getCompactAlignmentTable() const
{
  return compactAlignmentTable;
}

void Ibm2AlignmentModel::setCompactAlignmentTable(bool value)
{
  compactAlignmentTable = value;
}

void Ibm2AlignmentModel::initTargetWord(const vector<WordIndex>& nsrc, const vector<WordIndex>& trg, PositionIndex j)
{
  Ibm1AlignmentModel::initTargetWord(nsrc, trg, j);

  PositionIndex slen = (PositionIndex)nsrc.size() - 1;
  PositionIndex tlen = (PositionIndex)trg.size();

  alignmentTable->reserveSpace(j, slen, getCompactedSentenceLength(tlen));

  AlignmentKey key{j, slen, getCompactedSentenceLength(tlen)};
  AlignmentCountsElem& elem = alignmentCounts[key];
  if (elem.size() < nsrc.size())
    elem.resize(nsrc.size(), 0);
}

double Ibm2AlignmentModel::getCountNumerator(const vector<WordIndex>& nsrcSent, const vector<WordIndex>& trgSent,
                                             unsigned int i, unsigned int j)
{
  double d = Ibm1AlignmentModel::getCountNumerator(nsrcSent, trgSent, i, j);
  d = d * double{alignmentProb(j, (PositionIndex)nsrcSent.size() - 1, (PositionIndex)trgSent.size(), i)};
  return d;
}

void Ibm2AlignmentModel::incrementWordPairCounts(const vector<WordIndex>& nsrc, const vector<WordIndex>& trg,
                                                 PositionIndex i, PositionIndex j, double count)
{
  Ibm1AlignmentModel::incrementWordPairCounts(nsrc, trg, i, j, count);

  AlignmentKey key{j, (PositionIndex)nsrc.size() - 1, getCompactedSentenceLength(trg.size())};

#pragma omp atomic
  alignmentCounts[key][i] += count;
}

void Ibm2AlignmentModel::batchMaximizeProbs()
{
  Ibm1AlignmentModel::batchMaximizeProbs();

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

PositionIndex Ibm2AlignmentModel::getCompactedSentenceLength(PositionIndex len)
{
  return compactAlignmentTable ? 0 : len;
}

void Ibm2AlignmentModel::loadConfig(const YAML::Node& config)
{
  Ibm1AlignmentModel::loadConfig(config);

  compactAlignmentTable = config["compactAlignmentTable"].as<bool>();
}

bool Ibm2AlignmentModel::loadOldConfig(const char* prefFileName, int verbose)
{
  Ibm1AlignmentModel::loadOldConfig(prefFileName, verbose);

  compactAlignmentTable = false;

  return THOT_OK;
}

void Ibm2AlignmentModel::createConfig(YAML::Emitter& out)
{
  Ibm1AlignmentModel::createConfig(out);

  out << YAML::Key << "compactAlignmentTable" << YAML::Value << compactAlignmentTable;
}

Prob Ibm2AlignmentModel::alignmentProb(PositionIndex j, PositionIndex slen, PositionIndex tlen, PositionIndex i)
{
  double logProb = unsmoothedAlignmentLogProb(j, slen, tlen, i);
  double prob = logProb == SMALL_LG_NUM ? 1.0 / ((double)slen + 1) : exp(logProb);
  return max(prob, SW_PROB_SMOOTH);
}

LgProb Ibm2AlignmentModel::alignmentLogProb(PositionIndex j, PositionIndex slen, PositionIndex tlen, PositionIndex i)
{
  double logProb = unsmoothedAlignmentLogProb(j, slen, tlen, i);
  if (logProb == SMALL_LG_NUM)
    logProb = log(1.0 / ((double)slen + 1));
  return max(logProb, SW_LOG_PROB_SMOOTH);
}

double Ibm2AlignmentModel::unsmoothedAlignmentLogProb(PositionIndex j, PositionIndex slen, PositionIndex tlen,
                                                      PositionIndex i)
{
  bool found;
  double numer = alignmentTable->getNumerator(j, slen, getCompactedSentenceLength(tlen), i, found);
  if (found)
  {
    // aligNumer for pair as,i exists
    double denom = alignmentTable->getDenominator(j, slen, getCompactedSentenceLength(tlen), found);
    if (found)
      return numer - denom;
  }
  return SMALL_LG_NUM;
}

LgProb Ibm2AlignmentModel::getBestAlignment(const vector<WordIndex>& srcSentence, const vector<WordIndex>& trgSentence,
                                            vector<PositionIndex>& bestAlignment)
{
  if (sentenceLengthIsOk(srcSentence) && sentenceLengthIsOk(trgSentence))
  {
    LgProb lgProb = sentenceLengthLogProb((PositionIndex)srcSentence.size(), (PositionIndex)trgSentence.size());
    lgProb += getIbm2BestAlignment(addNullWordToWidxVec(srcSentence), trgSentence, bestAlignment);
    return lgProb;
  }
  else
  {
    bestAlignment.resize(trgSentence.size(), 0);
    return SMALL_LG_NUM;
  }
}

LgProb Ibm2AlignmentModel::computeLogProb(const vector<WordIndex>& srcSentence, const vector<WordIndex>& trgSentence,
                                          const WordAlignmentMatrix& aligMatrix, int verbose)
{
  PositionIndex i;

  vector<PositionIndex> alig;
  aligMatrix.getAligVec(alig);

  if (verbose)
  {
    for (i = 0; i < srcSentence.size(); ++i)
      cerr << srcSentence[i] << " ";
    cerr << "\n";
    for (i = 0; i < trgSentence.size(); ++i)
      cerr << trgSentence[i] << " ";
    cerr << "\n";
    for (i = 0; i < alig.size(); ++i)
      cerr << alig[i] << " ";
    cerr << "\n";
  }
  if (trgSentence.size() != alig.size())
  {
    cerr << "Error: the sentence t and the alignment vector have not the same size." << endl;
    return THOT_ERROR;
  }
  else
  {
    return computeIbm2LogProb(addNullWordToWidxVec(srcSentence), trgSentence, alig, verbose);
  }
}

LgProb Ibm2AlignmentModel::computeIbm2LogProb(const vector<WordIndex>& nsSent, const vector<WordIndex>& tSent,
                                              const vector<PositionIndex>& alig, int verbose)
{
  PositionIndex slen = (PositionIndex)nsSent.size() - 1;
  PositionIndex tlen = (PositionIndex)tSent.size();

  if (verbose)
    cerr << "Obtaining IBM Model 2 logprob...\n";

  LgProb lgProb = 0;
  for (PositionIndex j = 1; j <= alig.size(); ++j)
  {
    Prob p = translationProb(nsSent[alig[j]], tSent[j - 1]);
    if (verbose)
      cerr << "t(" << tSent[j - 1] << "|" << nsSent[alig[j - 1]] << ")= " << p << " ; logp=" << (double)log((double)p)
           << endl;
    lgProb = lgProb + (double)log((double)p);

    p = alignmentProb(j, slen, tlen, alig[j - 1]);
    lgProb = lgProb + (double)log((double)p);
  }
  return lgProb;
}

LgProb Ibm2AlignmentModel::computeSumLogProb(const vector<WordIndex>& srcSentence, const vector<WordIndex>& trgSentence,
                                             int verbose)
{
  return getIbm2SumLogProb(addNullWordToWidxVec(srcSentence), trgSentence, verbose);
}

LgProb Ibm2AlignmentModel::getIbm2SumLogProb(const vector<WordIndex>& nsSent, const vector<WordIndex>& tSent,
                                             int verbose)
{
  PositionIndex slen = (PositionIndex)nsSent.size() - 1;
  PositionIndex tlen = (PositionIndex)tSent.size();
  Prob sump;
  LgProb lexAligContrib;

  if (verbose)
    cerr << "Obtaining Sum IBM Model 2 logprob...\n";

  LgProb lgProb = sentenceLengthLogProb(slen, tlen);
  if (verbose)
    cerr << "- lenLgProb(tlen=" << tSent.size() << " | slen=" << slen << ")= " << sentenceLengthLogProb(slen, tlen)
         << endl;

  lexAligContrib = 0;
  for (PositionIndex j = 1; j <= tSent.size(); ++j)
  {
    sump = 0;
    for (PositionIndex i = 0; i < nsSent.size(); ++i)
    {
      sump += translationProb(nsSent[i], tSent[j - 1]) * alignmentProb(j, slen, tlen, i);
      if (verbose == 2)
      {
        cerr << "t( " << tSent[j - 1] << " | " << nsSent[i] << " )= " << translationProb(nsSent[i], tSent[j - 1])
             << endl;
        cerr << "a( " << i << "| j=" << j << ", slen=" << slen << ", tlen=" << tlen
             << ")= " << alignmentProb(j, slen, tlen, i) << endl;
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

bool Ibm2AlignmentModel::load(const char* prefFileName, int verbose)
{
  if (prefFileName[0] != 0)
  {
    bool retVal;

    // Load IBM 1 Model data
    retVal = Ibm1AlignmentModel::load(prefFileName, verbose);
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

bool Ibm2AlignmentModel::print(const char* prefFileName, int verbose)
{
  bool retVal;

  // Print IBM 1 Model data
  retVal = Ibm1AlignmentModel::print(prefFileName);
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

LgProb Ibm2AlignmentModel::getIbm2BestAlignment(const vector<WordIndex>& nSrcSentIndexVector,
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
      LgProb lp = translationLogProb(nSrcSentIndexVector[i], trgSentIndexVector[j - 1]);
      // alignment logprobability
      lp += alignmentLogProb(j, slen, tlen, i);

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

void Ibm2AlignmentModel::clear()
{
  Ibm1AlignmentModel::clear();
  alignmentTable->clear();
}

void Ibm2AlignmentModel::clearTempVars()
{
  Ibm1AlignmentModel::clearTempVars();
  alignmentCounts.clear();
}
