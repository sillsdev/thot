#include "FastAlignModel.h"

#include <algorithm>
#include <sstream>
#include <iomanip>
#ifdef _WIN32
#include <Windows.h>
#endif
#include "da.h"

using namespace std;

struct PairHash {
  size_t operator()(const pair<short, short>& x) const {
    return (unsigned short)x.first << 16 | (unsigned)x.second;
  }
};

void FastAlignModel::trainSentPairRange(pair<unsigned int, unsigned int> sentPairRange, int verbosity)
{
  if (iter == 0)
  {
    initialPass(sentPairRange);
    s2t.freeze();
  }

  SentPairCont buffer;
  for (unsigned int n = sentPairRange.first; n <= sentPairRange.second; ++n)
  {
    Sentence src = getSrcSent(n);
    Sentence trg = getTrgSent(n);
    buffer.push_back(pair<Sentence, Sentence>(src, trg));

    if (buffer.size() >= ThreadBufferSize)
    {
      updateFromPairs(buffer);
      buffer.clear();
    }
  }
  if (buffer.size() > 0)
  {
    updateFromPairs(buffer);
    buffer.clear();
  }

  empFeat /= nTrgTokens;
  if (iter > 0)
  {
    for (int ii = 0; ii < 8; ++ii)
    {
      double modFeat = 0;
#pragma omp parallel for reduction(+:modFeat)
      for (int i = 0; i < sizeCounts.size(); ++i)
      {
        const pair<short, short>& p = sizeCounts[i].first;
        for (short j = 1; j <= p.first; ++j)
        {
          double dLogZ = DiagonalAlignment::ComputeDLogZ(j, p.first, p.second, diagonalTension);
          modFeat += sizeCounts[i].second * dLogZ;
        }
      }
      modFeat /= nTrgTokens;
      diagonalTension += (empFeat - modFeat) * 20.0;
      if (diagonalTension <= 0.1)
        diagonalTension = 0.1;
      if (diagonalTension > 14)
        diagonalTension = 14;
    }
  }
  if (variationalBayes)
    s2t.normalizeVB(alpha);
  else
    s2t.normalize();
  iter++;
}

void FastAlignModel::trainAllSents(int verbosity)
{
  if (numSentPairs() > 0)
    trainSentPairRange(std::make_pair(0, numSentPairs() - 1), verbosity);
}

LgProb FastAlignModel::obtainBestAlignment(vector<WordIndex> srcSentIndexVector, vector<WordIndex> trgSentIndexVector,
  WordAligMatrix& bestWaMatrix)
{
  bestWaMatrix.clear();
  unsigned int slen = (unsigned int)srcSentIndexVector.size();
  unsigned int tlen = (unsigned int)trgSentIndexVector.size();

  bestWaMatrix.init(slen, tlen);

  double logProb = sentLenLgProb(slen, tlen);

  // compute likelihood
  for (PositionIndex j = 0; j < trgSentIndexVector.size(); ++j)
  {
    WordIndex fj = trgSentIndexVector[j];
    double sum = 0;
    int aj = 0;
    double maxPat = pts(NULL_WORD, fj) * aProb(j + 1, slen, tlen, 0);
    sum += maxPat;
    double az = computeAZ(j + 1, slen, tlen);
    for (PositionIndex i = 1; i <= srcSentIndexVector.size(); ++i)
    {
      double pat = pts(srcSentIndexVector[i - 1], fj) * aProb(az, j + 1, slen, tlen, i);
      if (pat > maxPat)
      {
        maxPat = pat;
        aj = i;
      }
      sum += pat;
    }
    logProb += log(sum);
    if (aj > 0)
      bestWaMatrix.set(aj - 1, j);
  }
  return logProb;
}

Prob FastAlignModel::pts(WordIndex s, WordIndex t)
{
  return s2t.safeProb(s, t);
}

LgProb FastAlignModel::logpts(WordIndex s, WordIndex t)
{
  return pts(s, t).get_lp();
}

double FastAlignModel::computeAZ(PositionIndex j, PositionIndex slen, PositionIndex tlen)
{
  double z = DiagonalAlignment::ComputeZ(j, tlen, slen, diagonalTension);
  return z / (1.0 - probAlignNull);
}

Prob FastAlignModel::aProb(double az, PositionIndex j, PositionIndex slen, PositionIndex tlen, PositionIndex i)
{
  double unnormalizedProb = DiagonalAlignment::UnnormalizedProb(j, i, tlen, slen, diagonalTension);
  return unnormalizedProb / az;
}

Prob FastAlignModel::aProb(PositionIndex j, PositionIndex slen, PositionIndex tlen, PositionIndex i)
{
  if (i == 0)
    return probAlignNull;

  double z = DiagonalAlignment::ComputeZ(j, tlen, slen, diagonalTension);
  double az = computeAZ(j, slen, tlen);
  return aProb(az, j, slen, tlen, i);
}

LgProb FastAlignModel::logaProb(PositionIndex j, PositionIndex slen, PositionIndex tlen, PositionIndex i)
{
  return aProb(j, slen, tlen, i).get_lp();
}

Prob FastAlignModel::sentLenProb(unsigned int slen, unsigned int tlen)
{
  return sentLenLgProb(slen, tlen).get_p();
}

//-------------------------
LgProb FastAlignModel::sentLenLgProb(unsigned int slen, unsigned int tlen)
{
  return Md::log_poisson(tlen, 0.05 + slen * meanSrcLenMultipler);
}

LgProb FastAlignModel::calcLgProb(const vector<WordIndex>& sSent, const vector<WordIndex>& tSent, int verbose)
{
  WordAligMatrix waMatrix;
  return obtainBestAlignment(sSent, tSent, waMatrix);
}

LgProb FastAlignModel::calcLgProbForAlig(const vector<WordIndex>& sSent, const vector<WordIndex>& tSent,
  WordAligMatrix aligMatrix, int verbose)
{
  Sentence nsSent = addNullWordToWidxVec(sSent);
  vector<PositionIndex> alig;
  aligMatrix.getAligVec(alig);

  unsigned int slen = (unsigned int)sSent.size();
  unsigned int tlen = (unsigned int)tSent.size();

  double logProb = sentLenLgProb(slen, tlen);

  // compute likelihood
  for (PositionIndex j = 0; j < alig.size(); ++j)
  {
    PositionIndex i = alig[j];
    double pat = pts(nsSent[i], tSent[j]) * aProb(j + 1, slen, tlen, i);
    logProb += log(pat);
  }
  return logProb;
}

void FastAlignModel::initPpInfo(unsigned int slen, const vector<WordIndex>& tSent, PpInfo& ppInfo)
{
  // Make room in ppInfo
  ppInfo.clear();
  for (unsigned int j = 0; j < tSent.size(); ++j)
    ppInfo.push_back(0);
  // Add NULL word
  unsigned int tlen = tSent.size();
  for (unsigned int j = 0; j < tSent.size(); ++j)
    ppInfo[j] += pts(NULL_WORD, tSent[j]) * aProb(j + 1, slen, tlen, 0);
}

void FastAlignModel::partialProbWithoutLen(unsigned int srcPartialLen, unsigned int slen, const vector<WordIndex>& s_,
  const vector<WordIndex>& tSent, PpInfo& ppInfo)
{
  unsigned int tlen = tSent.size();

  for (unsigned int i = 0; i < s_.size(); ++i)
  {
    for (unsigned int j = 0; j < tSent.size(); ++j)
    {
      ppInfo[j] += pts(s_[i], tSent[j]) * aProb(j + 1, slen, tlen, srcPartialLen + i + 1);
      // srcPartialLen+i is added 1 because the first source word has index 1
    }
  }
}

LgProb FastAlignModel::lpFromPpInfo(const PpInfo& ppInfo)
{
  LgProb lgProb = 0;

  for (unsigned int j = 0; j < ppInfo.size(); ++j)
    lgProb += log((double)ppInfo[j]);
  return lgProb;
}

void FastAlignModel::addHeurForNotAddedWords(int numSrcWordsToBeAdded, const vector<WordIndex>& tSent, PpInfo& ppInfo)
{
  for (unsigned int j = 0; j < tSent.size(); ++j)
    ppInfo[j] += numSrcWordsToBeAdded * exp((double)lgProbOfBestTransForTrgWord(tSent[j]));
}

void FastAlignModel::sustHeurForNotAddedWords(int numSrcWordsToBeAdded, const vector<WordIndex>& tSent, PpInfo& ppInfo)
{
  for (unsigned int j = 0; j < tSent.size(); ++j)
    ppInfo[j] -= numSrcWordsToBeAdded * exp((double)lgProbOfBestTransForTrgWord(tSent[j]));
}

LgProb FastAlignModel::lgProbOfBestTransForTrgWord(WordIndex t)
{
  BestLgProbForTrgWord::iterator tnIter;

  tnIter = bestLgProbForTrgWord.find(std::make_pair(0, t));
  if (tnIter == bestLgProbForTrgWord.end())
  {

    map<WordIndex, Prob> tNode;
    s2t.getEntriesForF(t, tNode);
    if (tNode.size() > 0)
    {
      map<WordIndex, Prob>::const_iterator ctnIter;
      Prob bestProb = 0;
      for (ctnIter = tNode.begin(); ctnIter != tNode.end(); ++ctnIter)
      {
        if (bestProb < ctnIter->second)
        {
          bestProb = ctnIter->second;
        }
      }
      bestLgProbForTrgWord[std::make_pair(0, t)] = log((double)bestProb);
      return log((double)bestProb);
    }
    else
    {
      bestLgProbForTrgWord[std::make_pair(0, t)] = -FLT_MAX;
      return -FLT_MAX;
    }
  }
  else return tnIter->second;
}

bool FastAlignModel::load(const char* prefFileName, int verbose)
{
  if (prefFileName[0] != 0)
  {
    bool retVal;

    if (verbose)
      std::cerr << "Loading FastAlign Model data..." << std::endl;

    // Load vocabularies if they exist
    std::string srcVocFileName = prefFileName;
    srcVocFileName = srcVocFileName + ".svcb";
    loadGIZASrcVocab(srcVocFileName.c_str(), verbose);

    std::string trgVocFileName = prefFileName;
    trgVocFileName = trgVocFileName + ".tvcb";
    loadGIZATrgVocab(trgVocFileName.c_str(), verbose);

    // Load files with source and target sentences
    // Warning: this must be made before reading file with anji
    // values
    std::string srcsFile = prefFileName;
    srcsFile = srcsFile + ".src";
    std::string trgsFile = prefFileName;
    trgsFile = trgsFile + ".trg";
    std::string srctrgcFile = prefFileName;
    srctrgcFile = srctrgcFile + ".srctrgc";
    std::pair<unsigned int, unsigned int> pui;
    retVal = readSentencePairs(srcsFile.c_str(), trgsFile.c_str(), srctrgcFile.c_str(), pui, verbose);
    if (retVal == THOT_ERROR) return THOT_ERROR;

    std::string lexProbFile = prefFileName;
    lexProbFile = lexProbFile + ".lexprob";
    s2t.deserializeLogProbsFromText(lexProbFile, swVocab);

    string paramsFile = prefFileName;
    paramsFile = paramsFile + ".params";
    loadParams(paramsFile);

    return THOT_OK;
  }
  else return THOT_ERROR;
}

void FastAlignModel::loadParams(const std::string& filename)
{
  ifstream in(filename);
  in >> meanSrcLenMultipler >> diagonalTension;
}

bool FastAlignModel::print(const char* prefFileName, int verbose)
{
  bool retVal;

  // Print vocabularies 
  string srcVocFileName = prefFileName;
  srcVocFileName = srcVocFileName + ".svcb";
  retVal = printGIZASrcVocab(srcVocFileName.c_str());
  if (retVal == THOT_ERROR) return THOT_ERROR;

  string trgVocFileName = prefFileName;
  trgVocFileName = trgVocFileName + ".tvcb";
  retVal = printGIZATrgVocab(trgVocFileName.c_str());
  if (retVal == THOT_ERROR) return THOT_ERROR;

  // Print files with source and target sentences to temp files
  string srcsFileTemp = prefFileName;
  srcsFileTemp = srcsFileTemp + ".src.tmp";
  string trgsFileTemp = prefFileName;
  trgsFileTemp = trgsFileTemp + ".trg.tmp";
  string srctrgcFileTemp = prefFileName;
  srctrgcFileTemp = srctrgcFileTemp + ".srctrgc.tmp";
  retVal = printSentPairs(srcsFileTemp.c_str(), trgsFileTemp.c_str(), srctrgcFileTemp.c_str());
  if (retVal == THOT_ERROR) return THOT_ERROR;

  // close sentence files
  sentenceHandler.clear();

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
  if (retVal == THOT_ERROR) return THOT_ERROR;

  string lexProbFile = prefFileName;
  lexProbFile = lexProbFile + ".lexprob";
  s2t.exportToFile(lexProbFile, swVocab, -4.0);

  string paramsFile = prefFileName;
  paramsFile = paramsFile + ".params";
  printParams(paramsFile);
}

void FastAlignModel::printParams(const std::string& filename)
{
  ofstream out(filename);
  out << meanSrcLenMultipler << " " << diagonalTension << endl;
}

Sentence FastAlignModel::getSrcSent(unsigned int n)
{
  std::vector<std::string> srcsStr;
  std::vector<WordIndex> result;

  sentenceHandler.getSrcSent(n, srcsStr);
  for (unsigned int i = 0; i < srcsStr.size(); ++i)
  {
    WordIndex widx = stringToSrcWordIndex(srcsStr[i]);
    if (widx == UNK_WORD)
      widx = addSrcSymbol(srcsStr[i]);
    result.push_back(widx);
  }
  return result;
}

Sentence FastAlignModel::getTrgSent(unsigned int n)
{
  std::vector<std::string> trgsStr;
  std::vector<WordIndex> trgs;

  sentenceHandler.getTrgSent(n, trgsStr);
  for (unsigned int i = 0; i < trgsStr.size(); ++i)
  {
    WordIndex widx = stringToTrgWordIndex(trgsStr[i]);
    if (widx == UNK_WORD)
      widx = addTrgSymbol(trgsStr[i]);
    trgs.push_back(widx);
  }
  return trgs;
}

void FastAlignModel::initialPass(std::pair<unsigned int, unsigned int> sentPairRange)
{
  unordered_map<pair<short, short>, unsigned, PairHash> tempSizeCounts;
  vector<vector<unsigned>> insertBuffer;
  size_t insertBufferItems = 0;
  double totLenRatio = 0;
  int lc = 0;
  for (unsigned int n = sentPairRange.first; n <= sentPairRange.second; ++n)
  {
    Sentence src = getSrcSent(n);
    Sentence trg = getTrgSent(n);
    lc++;
    unsigned int slen = (unsigned int)src.size();
    unsigned int tlen = (unsigned int)trg.size();
    totLenRatio += static_cast<double>(tlen) / static_cast<double>(slen);
    nTrgTokens += tlen;
    for (const WordIndex f : trg)
      s2t.insert(NULL_WORD, f);
    for (const WordIndex e : src)
    {
      if (e >= insertBuffer.size())
        insertBuffer.resize((size_t)e + 1);
      for (const WordIndex f : trg)
        insertBuffer[e].push_back(f);
      insertBufferItems += tlen;
    }
    if (insertBufferItems > ThreadBufferSize * 100)
    {
      insertBufferItems = 0;
      addTranslationOptions(insertBuffer);
    }
    tempSizeCounts[make_pair<short, short>((short)tlen, (short)slen)]++;
  }
  for (const auto& p : tempSizeCounts)
    sizeCounts.push_back(p);
  addTranslationOptions(insertBuffer);

  meanSrcLenMultipler = totLenRatio / lc;
}

void FastAlignModel::addTranslationOptions(vector<vector<WordIndex>>& insertBuffer) {
  s2t.setMaxE((WordIndex)insertBuffer.size() - 1);
#pragma omp parallel for schedule(dynamic)
  for (int e = 0; e < insertBuffer.size(); ++e)
  {
    for (WordIndex f : insertBuffer[e])
      s2t.insert(e, f);
    insertBuffer[e].clear();
  }
}

void FastAlignModel::updateFromPairs(const SentPairCont& pairs)
{
  double tempEmpFeat = 0.0;
#pragma omp parallel for schedule(dynamic) reduction(+:tempEmpFeat)
  for (int line_idx = 0; line_idx < pairs.size(); ++line_idx)
  {
    Sentence src = pairs[line_idx].first;
    Sentence trg = pairs[line_idx].second;
    unsigned int slen = (unsigned int)src.size();
    unsigned int tlen = (unsigned int)trg.size();
    vector<double> probs(src.size() + 1);
    for (PositionIndex j = 0; j < trg.size(); ++j)
    {
      const WordIndex& fj = trg[j];
      double sum = 0;
      probs[0] = s2t.prob(NULL_WORD, fj) * (double)aProb(j + 1, slen, tlen, 0);
      sum += probs[0];
      double az = computeAZ(j + 1, slen, tlen);
      for (PositionIndex i = 1; i <= src.size(); ++i)
      {
        probs[i] = s2t.prob(src[i - 1], fj) * (double)aProb(az, j + 1, slen, tlen, i);
        sum += probs[i];
      }
      double count = probs[0] / sum;
      s2t.increment(NULL_WORD, fj, count);
      for (PositionIndex i = 1; i <= src.size(); ++i)
      {
        const double p = probs[i] / sum;
        s2t.increment(src[i - 1], fj, p);
        tempEmpFeat += DiagonalAlignment::Feature(j, i, tlen, slen) * p;
      }
    }
  }
  empFeat += tempEmpFeat;
}

void FastAlignModel::clearSentLengthModel(void)
{
  meanSrcLenMultipler = 1.0;
}

void FastAlignModel::clearTempVars(void)
{
  bestLgProbForTrgWord.clear();
}

void FastAlignModel::clearInfoAboutSentRange(void)
{
  // Clear info about sentence range
  sentenceHandler.clear();
  iter = 0;
  empFeat = 0;
  nTrgTokens = 0;
  sizeCounts.clear();
}

void FastAlignModel::clear(void)
{
  _swAligModel<vector<Prob>>::clear();
  clearSentLengthModel();
  clearInfoAboutSentRange();
  clearTempVars();
  diagonalTension = 4.0;
  s2t.clear();
}
