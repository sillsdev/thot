#ifndef _FastAlignModel_h
#define _FastAlignModel_h

#if HAVE_CONFIG_H
#include <thot_config.h>
#endif /* HAVE_CONFIG_H */
#include <unordered_map>

#include "_incrSwAligModel.h"
#include "IncrLexTable.h"
#include "anjiMatrix.h"
#include "LexAuxVar.h"
#include "BestLgProbForTrgWord.h"

struct PairLess {
  bool operator()(const std::pair<short, short>& x, const std::pair<short, short>& y) const
  {
    if (x.first < y.first)
      return true;
    if (x.first == y.first && x.second < y.second)
      return true;
    return false;
  }
};

class FastAlignModel : public _incrSwAligModel<std::vector<Prob>>
{
public:
  typedef _incrSwAligModel<std::vector<Prob>>::PpInfo PpInfo;
  typedef std::map<WordIndex, Prob> SrcTableNode;
  typedef OrderedVector<std::pair<short, short>, unsigned int, PairLess> SizeCounts;

  void set_expval_maxnsize(unsigned int _anji_maxnsize);

  void trainSentPairRange(std::pair<unsigned int, unsigned int> sentPairRange, int verbosity = 0);
  void trainAllSents(int verbosity = 0);
  void efficientBatchTrainingForRange(std::pair<unsigned int, unsigned int> sentPairRange, int verbosity = 0);
  void efficientBatchTrainingForAllSents(int verbosity = 0);
  void clearInfoAboutSentRange(void);

  bool getEntriesForTarget(WordIndex t, SrcTableNode& srctn);
  LgProb obtainBestAlignment(std::vector<WordIndex> srcSentIndexVector, std::vector<WordIndex> trgSentIndexVector,
    WordAligMatrix& bestWaMatrix);

  Prob pts(WordIndex s, WordIndex t);
  LgProb logpts(WordIndex s, WordIndex t);

  Prob aProb(PositionIndex j, PositionIndex slen, PositionIndex tlen, PositionIndex i);
  LgProb logaProb(PositionIndex j, PositionIndex slen, PositionIndex tlen, PositionIndex i);

  Prob sentLenProb(unsigned int slen, unsigned int tlen);
  LgProb sentLenLgProb(unsigned int slen, unsigned int tlen);

  LgProb calcLgProb(const std::vector<WordIndex>& sSent, const std::vector<WordIndex>& tSent, int verbose = 0);
  LgProb calcLgProbForAlig(const std::vector<WordIndex>& sSent, const std::vector<WordIndex>& tSent,
    WordAligMatrix aligMatrix, int verbose = 0);

  void initPpInfo(unsigned int slen, const std::vector<WordIndex>& tSent, PpInfo& ppInfo);
  void partialProbWithoutLen(unsigned int srcPartialLen, unsigned int slen, const std::vector<WordIndex>& s_,
    const std::vector<WordIndex>& tSent, PpInfo& ppInfo);
  LgProb lpFromPpInfo(const PpInfo& ppInfo);
  void addHeurForNotAddedWords(int numSrcWordsToBeAdded, const std::vector<WordIndex>& tSent, PpInfo& ppInfo);
  void sustHeurForNotAddedWords(int numSrcWordsToBeAdded, const std::vector<WordIndex>& tSent, PpInfo& ppInfo);

  bool load(const char* prefFileName, int verbose = 0);
  bool print(const char* prefFileName, int verbose = 0);

  void clearSentLengthModel(void);
  void clearTempVars(void);
  void clear(void);

  bool variationalBayes = true;
  double probAlignNull = 0.08;
  double alpha = 0.01;

private:
  const std::size_t ThreadBufferSize = 10000;
  const double SmallProb = 1e-9;
  const double SmallLogProb = log(SmallProb);
  const double SmoothingAnjiNum = 1e-9;
  const double SmoothingWeightedAnji = 1e-9;
  const double ArbitraryPts = 0.05;

  void initialBatchPass(std::pair<unsigned int, unsigned int> sentPairRange, int verbose);
  void addTranslationOptions(std::vector<std::vector<WordIndex>>& insertBuffer);
  void updateFromPairs(const SentPairCont& pairs);
  Sentence getSrcSent(unsigned int n);
  Sentence getTrgSent(unsigned int n);
  double computeAZ(PositionIndex j, PositionIndex slen, PositionIndex tlen);
  Prob aProb(double az, PositionIndex j, PositionIndex slen, PositionIndex tlen, PositionIndex i);
  LgProb lgProbOfBestTransForTrgWord(WordIndex t);
  bool printParams(const std::string& filename);
  bool loadParams(const std::string& filename);
  bool printSizeCounts(const std::string& filename);
  bool loadSizeCounts(const std::string& filename);
  void normalizeCounts(void);
  void optimizeDiagonalTension(unsigned int nIters, int verbose);
  void incrementSizeCount(unsigned int tlen, unsigned int slen);

  void initialIncrPass(std::pair<unsigned int, unsigned int> sentPairRange, int verbose);
  void calcNewLocalSuffStats(std::pair<unsigned int, unsigned int> sentPairRange, int verbosity = 0);
  void calc_anji(unsigned int n, const std::vector<WordIndex>& nsrcSent, const std::vector<WordIndex>& trgSent,
    const Count& weight);
  double calc_anji_num(const std::vector<WordIndex>& nsrcSent, const std::vector<WordIndex>& trgSent, unsigned int i,
    unsigned int j);
  void fillEmAuxVars(unsigned int mapped_n, unsigned int mapped_n_aux, PositionIndex i, PositionIndex j,
    const std::vector<WordIndex>& nsrcSent, const std::vector<WordIndex>& trgSent, const Count& weight);
  void updatePars(void);
  float obtainLogNewSuffStat(float lcurrSuffStat, float lLocalSuffStatCurr, float lLocalSuffStatNew);

  inline void setCountMaxSrcWordIndex(const WordIndex s)
  {
    // NOT thread safe
    if (s >= counts.size())
      counts.resize((size_t)s + 1);
  }

  inline void initCountSlot(const WordIndex s, const WordIndex t)
  {
    // NOT thread safe
    if (s >= counts.size())
      counts.resize((size_t)s + 1);
    counts[s][t] = 0;
  }

  inline void incrementCount(const WordIndex s, const WordIndex t, const double x)
  {
#pragma omp atomic
    counts[s].find(t)->second += x;
  }

  IncrLexTable incrLexTable;
  double diagonalTension = 4.0;
  double totLenRatio = 0;
  double empFeatSum = 0;
  double trgTokenCount = 0;
  SizeCounts sizeCounts;
  anjiMatrix anji;

  anjiMatrix anji_aux;
  LexAuxVar lexAuxVar;
  int iter = 0;
  std::vector<std::unordered_map<WordIndex, double>> counts;

  BestLgProbForTrgWord bestLgProbForTrgWord;

};

#endif