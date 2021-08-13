#include "Ibm1AligModel.h"

#include "nlp_common/ErrorDefs.h"
#include "sw_models/Ibm1AligModel.h"
#include "sw_models/Md.h"
#include "sw_models/MemoryLexTable.h"
#include "sw_models/SwDefs.h"

#include <algorithm>

#ifdef _WIN32
#define NOMINMAX
#include <Windows.h>
#endif

using namespace std;

Ibm1AligModel::Ibm1AligModel()
    : sentLengthModel{make_shared<WeightedIncrNormSlm>()}, lexTable{make_shared<MemoryLexTable>()}
{
  // Link pointers with sentence length model
  sentLengthModel->linkVocabPtr(swVocab.get());
  sentLengthModel->linkSentPairInfo(sentenceHandler.get());
}

Ibm1AligModel::Ibm1AligModel(Ibm1AligModel& model)
    : _swAligModel{model}, sentLengthModel{model.sentLengthModel}, lexTable{model.lexTable}
{
}

void Ibm1AligModel::startTraining(int verbosity)
{
  clearTempVars();
  vector<vector<WordIndex>> insertBuffer;
  size_t insertBufferItems = 0;
  for (unsigned int n = 0; n < numSentPairs(); ++n)
  {
    vector<WordIndex> src = getSrcSent(n);
    vector<WordIndex> trg = getTrgSent(n);

    if (sentenceLengthIsOk(src) && sentenceLengthIsOk(trg))
    {
      initSentencePair(src, trg);

      vector<WordIndex> nsrc = extendWithNullWord(src);
      PositionIndex slen = (PositionIndex)src.size();
      PositionIndex tlen = (PositionIndex)trg.size();

      for (PositionIndex i = 0; i <= slen; ++i)
      {
        initSourceWord(nsrc, trg, i);
        WordIndex s = nsrc[i];
        if (s >= insertBuffer.size())
          insertBuffer.resize((size_t)s + 1);
        for (PositionIndex j = 1; j <= tlen; ++j)
        {
          if (i == 0)
            initTargetWord(nsrc, trg, j);
          initWordPair(nsrc, trg, i, j);
          insertBuffer[s].push_back(trg[j - 1]);
        }
        insertBufferItems += trg.size();
      }
      if (insertBufferItems > ThreadBufferSize * 100)
      {
        insertBufferItems = 0;
        addTranslationOptions(insertBuffer);
      }
    }
  }
  if (insertBufferItems > 0)
    addTranslationOptions(insertBuffer);

  // Train sentence length model
  sentLengthModel->trainSentPairRange(make_pair(0, numSentPairs() - 1), verbosity);
}

void Ibm1AligModel::train(int verbosity)
{
  vector<pair<vector<WordIndex>, vector<WordIndex>>> buffer;
  for (unsigned int n = 0; n < numSentPairs(); ++n)
  {
    vector<WordIndex> src = getSrcSent(n);
    vector<WordIndex> trg = getTrgSent(n);
    if (sentenceLengthIsOk(src) && sentenceLengthIsOk(trg))
      buffer.push_back(make_pair(src, trg));

    if (buffer.size() >= ThreadBufferSize)
    {
      batchUpdateCounts(buffer);
      buffer.clear();
    }
  }
  if (buffer.size() > 0)
  {
    batchUpdateCounts(buffer);
    buffer.clear();
  }

  batchMaximizeProbs();
}

void Ibm1AligModel::endTraining()
{
  clearTempVars();
}

void Ibm1AligModel::initSentencePair(const vector<WordIndex>& src, const vector<WordIndex>& trg)
{
}

void Ibm1AligModel::initSourceWord(const vector<WordIndex>& nsrc, const vector<WordIndex>& trg, PositionIndex i)
{
}

void Ibm1AligModel::initTargetWord(const vector<WordIndex>& nsrc, const vector<WordIndex>& trg, PositionIndex j)
{
}

void Ibm1AligModel::initWordPair(const vector<WordIndex>& nsrc, const vector<WordIndex>& trg, PositionIndex i,
                                 PositionIndex j)
{
}

void Ibm1AligModel::addTranslationOptions(vector<vector<WordIndex>>& insertBuffer)
{
  WordIndex maxSrcWordIndex = (WordIndex)insertBuffer.size() - 1;
  if (maxSrcWordIndex >= lexCounts.size())
    lexCounts.resize((size_t)maxSrcWordIndex + 1);
  lexTable->reserveSpace(maxSrcWordIndex);

#pragma omp parallel for schedule(dynamic)
  for (int s = 0; s < (int)insertBuffer.size(); ++s)
  {
    for (WordIndex t : insertBuffer[s])
      lexCounts[s][t] = 0;
    insertBuffer[s].clear();
  }
}

void Ibm1AligModel::batchUpdateCounts(const vector<pair<vector<WordIndex>, vector<WordIndex>>>& pairs)
{
#pragma omp parallel for schedule(dynamic)
  for (int line_idx = 0; line_idx < (int)pairs.size(); ++line_idx)
  {
    vector<WordIndex> src = pairs[line_idx].first;
    vector<WordIndex> nsrc = extendWithNullWord(src);
    vector<WordIndex> trg = pairs[line_idx].second;
    vector<double> probs(nsrc.size());
    for (PositionIndex j = 1; j <= trg.size(); ++j)
    {
      double sum = 0;
      for (PositionIndex i = 0; i < nsrc.size(); ++i)
      {
        probs[i] = getCountNumerator(nsrc, trg, i, j);
        sum += probs[i];
      }
      for (PositionIndex i = 0; i < nsrc.size(); ++i)
      {
        double count = probs[i] / sum;
        incrementWordPairCounts(nsrc, trg, i, j, count);
      }
    }
  }
}

double Ibm1AligModel::getCountNumerator(const vector<WordIndex>& nsrcSent, const vector<WordIndex>& trgSent,
                                        PositionIndex i, PositionIndex j)
{
  WordIndex s = nsrcSent[i];
  WordIndex t = trgSent[j - 1];
  return pts(s, t);
}

void Ibm1AligModel::incrementWordPairCounts(const vector<WordIndex>& nsrc, const vector<WordIndex>& trg,
                                            PositionIndex i, PositionIndex j, double count)
{
  WordIndex s = nsrc[i];
  WordIndex t = trg[j - 1];

#pragma omp atomic
  lexCounts[s].find(t)->second += count;
}

void Ibm1AligModel::batchMaximizeProbs()
{
#pragma omp parallel for schedule(dynamic)
  for (int s = 0; s < (int)lexCounts.size(); ++s)
  {
    double denom = 0;
    LexCountsElem& elem = lexCounts[s];
    for (auto& pair : elem)
    {
      double numer = pair.second;
      if (variationalBayes)
        numer += alpha;
      denom += numer;
      lexTable->setNumerator(s, pair.first, (float)log(numer));
      pair.second = 0.0;
    }
    if (denom == 0)
      denom = 1;
    lexTable->setDenominator(s, (float)log(denom));
  }
}

pair<double, double> Ibm1AligModel::loglikelihoodForPairRange(pair<unsigned int, unsigned int> sentPairRange,
                                                              int verbosity)
{
  double loglikelihood = 0;
  unsigned int numSents = 0;

  for (unsigned int n = sentPairRange.first; n <= sentPairRange.second; ++n)
  {
    if (verbosity)
      cerr << "* Calculating log-likelihood for sentence " << n << std::endl;
    // Add log-likelihood
    vector<WordIndex> nthSrcSent = getSrcSent(n);
    vector<WordIndex> nthTrgSent = getTrgSent(n);
    if (sentenceLengthIsOk(nthSrcSent) && sentenceLengthIsOk(nthTrgSent))
    {
      loglikelihood += (double)calcLgProb(nthSrcSent, nthTrgSent, verbosity);
      ++numSents;
    }
  }
  return make_pair(loglikelihood, loglikelihood / (double)numSents);
}

vector<WordIndex> Ibm1AligModel::getSrcSent(unsigned int n)
{
  vector<string> srcsStr;
  vector<WordIndex> result;

  sentenceHandler->getSrcSent(n, srcsStr);
  for (unsigned int i = 0; i < srcsStr.size(); ++i)
  {
    WordIndex widx = stringToSrcWordIndex(srcsStr[i]);
    if (widx == UNK_WORD)
      widx = addSrcSymbol(srcsStr[i]);
    result.push_back(widx);
  }
  return result;
}

vector<WordIndex> Ibm1AligModel::extendWithNullWord(const vector<WordIndex>& srcWordIndexVec)
{
  return addNullWordToWidxVec(srcWordIndexVec);
}

vector<WordIndex> Ibm1AligModel::getTrgSent(unsigned int n)
{
  vector<string> trgsStr;
  vector<WordIndex> trgs;

  sentenceHandler->getTrgSent(n, trgsStr);
  for (unsigned int i = 0; i < trgsStr.size(); ++i)
  {
    WordIndex widx = stringToTrgWordIndex(trgsStr[i]);
    if (widx == UNK_WORD)
      widx = addTrgSymbol(trgsStr[i]);
    trgs.push_back(widx);
  }
  return trgs;
}

bool Ibm1AligModel::sentenceLengthIsOk(const vector<WordIndex> sentence)
{
  return !sentence.empty() && sentence.size() <= IBM_SWM_MAX_SENT_LENGTH;
}

Prob Ibm1AligModel::pts(WordIndex s, WordIndex t)
{
  double logProb = unsmoothed_logpts(s, t);
  double prob = logProb == SMALL_LG_NUM ? 1.0 / getTrgVocabSize() : exp(logProb);
  return std::max(prob, SW_PROB_SMOOTH);
}

double Ibm1AligModel::unsmoothed_pts(WordIndex s, WordIndex t)
{
  return exp(unsmoothed_logpts(s, t));
}

LgProb Ibm1AligModel::logpts(WordIndex s, WordIndex t)
{
  double logProb = unsmoothed_logpts(s, t);
  if (logProb == SMALL_LG_NUM)
    logProb = log(1.0 / getTrgVocabSize());
  return std::max(logProb, SW_LOG_PROB_SMOOTH);
}

double Ibm1AligModel::unsmoothed_logpts(WordIndex s, WordIndex t)
{
  bool found;
  double numer = lexTable->getNumerator(s, t, found);
  if (found)
  {
    // lexNumer for pair s,t exists
    double denom = lexTable->getDenominator(s, found);
    if (found)
    {
      if (variationalBayes)
      {
        numer = Md::digamma(exp(numer));
        denom = Md::digamma(exp(denom));
      }
      return numer - denom;
    }
  }
  return SMALL_LG_NUM;
}

Prob Ibm1AligModel::aProbIbm1(PositionIndex slen, PositionIndex tlen)
{
  return logaProbIbm1(slen, tlen).get_p();
}

LgProb Ibm1AligModel::logaProbIbm1(PositionIndex slen, PositionIndex tlen)
{
  LgProb aligLgProb = 0;

  for (unsigned int j = 0; j < tlen; ++j)
  {
    aligLgProb = (double)aligLgProb - (double)log((double)slen + 1);
  }
  return aligLgProb;
}

Prob Ibm1AligModel::sentLenProb(PositionIndex slen, PositionIndex tlen)
{
  return sentLengthModel->sentLenProb(slen, tlen);
}

LgProb Ibm1AligModel::sentLenLgProb(PositionIndex slen, PositionIndex tlen)
{
  return sentLengthModel->sentLenLgProb(slen, tlen);
}

LgProb Ibm1AligModel::lexM1LpForBestAlig(const vector<WordIndex>& nSrcSentIndexVector,
                                         const vector<WordIndex>& trgSentIndexVector, vector<PositionIndex>& bestAlig)
{
  LgProb aligLgProb;
  LgProb lp;
  LgProb max_lp;
  unsigned int best_i = 0;

  bestAlig.clear();
  aligLgProb = 0;
  for (PositionIndex j = 1; j <= trgSentIndexVector.size(); ++j)
  {
    max_lp = -FLT_MAX;
    for (PositionIndex i = 0; i < nSrcSentIndexVector.size(); ++i)
    {
      lp = log((double)pts(nSrcSentIndexVector[i], trgSentIndexVector[j - 1]));
      if (max_lp <= lp)
      {
        max_lp = lp;
        best_i = i;
      }
    }
    aligLgProb = aligLgProb + max_lp;
    bestAlig.push_back(best_i);
  }

  return aligLgProb;
}

bool Ibm1AligModel::getEntriesForSource(WordIndex s, NbestTableNode<WordIndex>& trgtn)
{
  set<WordIndex> transSet;
  bool ret = lexTable->getTransForSource(s, transSet);
  if (ret == false)
    return false;

  trgtn.clear();
  set<WordIndex>::const_iterator setIter;
  for (setIter = transSet.begin(); setIter != transSet.end(); ++setIter)
  {
    WordIndex t = *setIter;
    trgtn.insert(pts(s, t), t);
  }
  return true;
}

LgProb Ibm1AligModel::obtainBestAlignment(const vector<WordIndex>& src, const vector<WordIndex>& trg,
                                          WordAligMatrix& bestWaMatrix)
{
  if (sentenceLengthIsOk(src) && sentenceLengthIsOk(trg))
  {
    vector<PositionIndex> bestAlig;
    LgProb lgProb = logaProbIbm1((PositionIndex)src.size(), (PositionIndex)trg.size());
    lgProb += sentLenLgProb((PositionIndex)src.size(), (PositionIndex)trg.size());
    lgProb += lexM1LpForBestAlig(addNullWordToWidxVec(src), trg, bestAlig);

    bestWaMatrix.init((PositionIndex)src.size(), (PositionIndex)trg.size());
    bestWaMatrix.putAligVec(bestAlig);

    return lgProb;
  }
  else
  {
    bestWaMatrix.init((PositionIndex)src.size(), (PositionIndex)trg.size());
    return SMALL_LG_NUM;
  }
}

LgProb Ibm1AligModel::calcLgProbForAlig(const vector<WordIndex>& src, const vector<WordIndex>& trg,
                                        const WordAligMatrix& aligMatrix, int verbose)
{
  PositionIndex i;

  vector<PositionIndex> alig;
  aligMatrix.getAligVec(alig);

  if (verbose)
  {
    for (i = 0; i < src.size(); ++i)
      cerr << src[i] << " ";
    cerr << "\n";
    for (i = 0; i < trg.size(); ++i)
      cerr << trg[i] << " ";
    cerr << "\n";
    for (i = 0; i < alig.size(); ++i)
      cerr << alig[i] << " ";
    cerr << "\n";
  }
  if (trg.size() != alig.size())
  {
    cerr << "Error: the sentence t and the alignment vector have not the same size." << endl;
    return THOT_ERROR;
  }
  else
  {
    return calcIbm1LgProbForAlig(addNullWordToWidxVec(src), trg, alig, verbose);
  }
}

LgProb Ibm1AligModel::calcIbm1LgProbForAlig(const vector<WordIndex>& nsSent, const vector<WordIndex>& tSent,
                                            const vector<PositionIndex>& alig, int verbose)
{
  Prob p;
  LgProb lgProb;
  PositionIndex j;
  if (verbose)
    cerr << "Obtaining IBM Model 1 logprob...\n";

  lgProb = logaProbIbm1((PositionIndex)nsSent.size() - 1, (PositionIndex)tSent.size());
  if (verbose)
    cerr << "- aligLgProb(tlen=" << tSent.size() << " | slen=" << nsSent.size() - 1
         << ")= " << logaProbIbm1((PositionIndex)nsSent.size() - 1, (PositionIndex)tSent.size()) << endl;

  lgProb += sentLenLgProb((PositionIndex)nsSent.size() - 1, (PositionIndex)tSent.size());
  if (verbose)
    cerr << "- lenLgProb(tlen=" << tSent.size() << " | slen=" << nsSent.size() - 1
         << ")= " << sentLenLgProb((PositionIndex)nsSent.size() - 1, (PositionIndex)tSent.size()) << endl;

  for (j = 1; j <= alig.size(); ++j)
  {
    p = pts(nsSent[alig[j - 1]], tSent[j - 1]);
    if (verbose)
      cerr << "t(" << tSent[j - 1] << "|" << nsSent[alig[j - 1]] << ")= " << p << " ; logp=" << (double)log((double)p)
           << endl;
    lgProb = lgProb + (double)log((double)p);
  }

  return lgProb;
}

LgProb Ibm1AligModel::calcLgProb(const vector<WordIndex>& src, const vector<WordIndex>& trg, int verbose)
{
  if (sentenceLengthIsOk(src) && sentenceLengthIsOk(trg))
  {
    return calcSumIbm1LgProb(addNullWordToWidxVec(src), trg, verbose);
  }
  else
  {
    return SMALL_LG_NUM;
  }
}

LgProb Ibm1AligModel::calcSumIbm1LgProb(const vector<WordIndex>& nsSent, const vector<WordIndex>& tSent, int verbose)
{
  Prob sump;
  LgProb lexContrib;
  LgProb lgProb;
  PositionIndex i, j;

  if (verbose)
    cerr << "Obtaining Sum IBM Model 1 logprob...\n";

  lgProb = logaProbIbm1((PositionIndex)nsSent.size() - 1, (PositionIndex)tSent.size());

  if (verbose)
    cerr << "- aligLgProb(tlen=" << tSent.size() << " | slen=" << nsSent.size() - 1
         << ")= " << logaProbIbm1((PositionIndex)nsSent.size() - 1, (PositionIndex)tSent.size()) << endl;

  lgProb += sentLenLgProb((PositionIndex)nsSent.size() - 1, (PositionIndex)tSent.size());
  if (verbose)
    cerr << "- lenLgProb(tlen=" << tSent.size() << " | slen=" << nsSent.size() - 1
         << ")= " << sentLenLgProb((PositionIndex)nsSent.size() - 1, (PositionIndex)tSent.size()) << endl;

  lexContrib = 0;
  for (j = 1; j <= tSent.size(); ++j)
  {
    sump = 0;
    for (i = 0; i < nsSent.size(); ++i)
    {
      sump += pts(nsSent[i], tSent[j - 1]);
      if (verbose == 2)
        cerr << "t( " << tSent[j - 1] << " | " << nsSent[i] << " )= " << pts(nsSent[i], tSent[j - 1]) << endl;
    }
    lexContrib += (double)log((double)sump);
    if (verbose)
      cerr << "- sumt(j=" << j << ")= " << sump << endl;
    if (verbose == 2)
      cerr << endl;
  }

  if (verbose)
    cerr << "- Lexical model contribution= " << lexContrib << endl;
  lgProb += lexContrib;

  return lgProb;
}

bool Ibm1AligModel::load(const char* prefFileName, int verbose)
{
  if (prefFileName[0] != 0)
  {
    bool retVal;

    if (verbose)
      cerr << "Loading incremental IBM 1 Model data..." << endl;

    // Load vocabularies if they exist
    string srcVocFileName = prefFileName;
    srcVocFileName = srcVocFileName + ".svcb";
    loadGIZASrcVocab(srcVocFileName.c_str(), verbose);

    string trgVocFileName = prefFileName;
    trgVocFileName = trgVocFileName + ".tvcb";
    loadGIZATrgVocab(trgVocFileName.c_str(), verbose);

    // Load files with source and target sentences
    // Warning: this must be made before reading file with anji
    // values
    string srcsFile = prefFileName;
    srcsFile = srcsFile + ".src";
    string trgsFile = prefFileName;
    trgsFile = trgsFile + ".trg";
    string srctrgcFile = prefFileName;
    srctrgcFile = srctrgcFile + ".srctrgc";
    pair<unsigned int, unsigned int> pui;
    retVal = readSentencePairs(srcsFile.c_str(), trgsFile.c_str(), srctrgcFile.c_str(), pui, verbose);
    if (retVal == THOT_ERROR)
      return THOT_ERROR;

    // Load file with lexical nd values
    string lexNumDenFile = prefFileName;
    lexNumDenFile = lexNumDenFile + lexNumDenFileExtension;
    retVal = lexTable->load(lexNumDenFile.c_str(), verbose);
    if (retVal == THOT_ERROR)
      return THOT_ERROR;

    // Load average sentence lengths
    string slmodelFile = prefFileName;
    slmodelFile = slmodelFile + ".slmodel";
    retVal = sentLengthModel->load(slmodelFile.c_str(), verbose);
    if (retVal == THOT_ERROR)
      return THOT_ERROR;

    string variationalBayesFile = prefFileName;
    variationalBayesFile = variationalBayesFile + ".var_bayes";
    loadVariationalBayes(variationalBayesFile);

    return THOT_OK;
  }
  else
    return THOT_ERROR;
}

bool Ibm1AligModel::print(const char* prefFileName, int verbose)
{
  bool retVal;

  // Print vocabularies
  string srcVocFileName = prefFileName;
  srcVocFileName = srcVocFileName + ".svcb";
  retVal = printGIZASrcVocab(srcVocFileName.c_str());
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  string trgVocFileName = prefFileName;
  trgVocFileName = trgVocFileName + ".tvcb";
  retVal = printGIZATrgVocab(trgVocFileName.c_str());
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  // Print files with source and target sentences to temp files
  string srcsFileTemp = prefFileName;
  srcsFileTemp = srcsFileTemp + ".src.tmp";
  string trgsFileTemp = prefFileName;
  trgsFileTemp = trgsFileTemp + ".trg.tmp";
  string srctrgcFileTemp = prefFileName;
  srctrgcFileTemp = srctrgcFileTemp + ".srctrgc.tmp";
  retVal = printSentPairs(srcsFileTemp.c_str(), trgsFileTemp.c_str(), srctrgcFileTemp.c_str());
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  // close sentence files
  sentenceHandler->clear();

  string srcsFile = prefFileName;
  srcsFile = srcsFile + ".src";
  string trgsFile = prefFileName;
  trgsFile = trgsFile + ".trg";
  string srctrgcFile = prefFileName;
  srctrgcFile = srctrgcFile + ".srctrgc";

  // move temp files to real destination
#ifdef _WIN32
  if (!MoveFileExA(srcsFileTemp.c_str(), srcsFile.c_str(), MOVEFILE_REPLACE_EXISTING))
    return THOT_ERROR;
  if (!MoveFileExA(trgsFileTemp.c_str(), trgsFile.c_str(), MOVEFILE_REPLACE_EXISTING))
    return THOT_ERROR;
  if (!MoveFileExA(srctrgcFileTemp.c_str(), srctrgcFile.c_str(), MOVEFILE_REPLACE_EXISTING))
    return THOT_ERROR;
#else
  if (rename(srcsFileTemp.c_str(), srcsFile.c_str()) != 0)
    return THOT_ERROR;
  if (rename(trgsFileTemp.c_str(), trgsFile.c_str()) != 0)
    return THOT_ERROR;
  if (rename(srctrgcFileTemp.c_str(), srctrgcFile.c_str()) != 0)
    return THOT_ERROR;
#endif

  // reload sentence files
  pair<unsigned int, unsigned int> pui;
  retVal = readSentencePairs(srcsFile.c_str(), trgsFile.c_str(), srctrgcFile.c_str(), pui, verbose);
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  // Print file with lexical nd values
  string lexNumDenFile = prefFileName;
  lexNumDenFile = lexNumDenFile + ".ibm_lexnd";
  retVal = lexTable->print(lexNumDenFile.c_str());
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  // Print file with sentence length model
  string slmodelFile = prefFileName;
  slmodelFile = slmodelFile + ".slmodel";
  retVal = sentLengthModel->print(slmodelFile.c_str());
  if (retVal == THOT_ERROR)
    return THOT_ERROR;

  string variationalBayesFile = prefFileName;
  variationalBayesFile = variationalBayesFile + ".var_bayes";
  return printVariationalBayes(variationalBayesFile);
}

void Ibm1AligModel::clear()
{
  _swAligModel::clear();
  clearSentLengthModel();
  clearTempVars();
  lexTable->clear();
}

void Ibm1AligModel::clearInfoAboutSentRange()
{
  // Clear info about sentence range
  sentenceHandler->clear();
}

void Ibm1AligModel::clearTempVars()
{
  lexCounts.clear();
}

void Ibm1AligModel::clearSentLengthModel()
{
  sentLengthModel->clear();
}
