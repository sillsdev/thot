#pragma once

#include "nlp_common/MathFuncs.h"
#include "sw_models/CachedHmmAligLgProb.h"
#include "sw_models/DoubleMatrix.h"
#include "sw_models/HmmAligInfo.h"
#include "sw_models/IncrHmmAligTable.h"
#include "sw_models/LexCounts.h"
#include "sw_models/LexTable.h"
#include "sw_models/WeightedIncrNormSlm.h"
#include "sw_models/_incrSwAligModel.h"
#include "sw_models/aSourceHmm.h"
#include "sw_models/anjiMatrix.h"
#include "sw_models/anjm1ip_anjiMatrix.h"
#include "sw_models/ashPidxPairHashF.h"

#include <unordered_map>

#define DEFAULT_ALIG_SMOOTH_INTERP_FACTOR 0.3
#define DEFAULT_LEX_SMOOTH_INTERP_FACTOR 0.1

class _incrHmmAligModel : public _swAligModel, public _incrSwAligModel
{
public:
  // Constructor
  _incrHmmAligModel();

  // Function to set a maximum size for the matrices of expected
  // values (by default the size is not restricted)
  void set_expval_maxnsize(unsigned int _expval_maxnsize);

  unsigned int numSentPairs(void);

  void startTraining(int verbosity = 0);
  void train(int verbosity = 0);
  void endTraining();

  void startIncrTraining(std::pair<unsigned int, unsigned int> sentPairRange, int verbosity = 0);
  void incrTrain(std::pair<unsigned int, unsigned int> sentPairRange, int verbosity = 0);
  void endIncrTraining();

  std::pair<double, double> loglikelihoodForPairRange(std::pair<unsigned int, unsigned int> sentPairRange,
                                                      int verbosity = 0);
  // Returns log-likelihood. The first double contains the
  // loglikelihood for all sentences, and the second one, the same
  // loglikelihood normalized by the number of sentences
  std::pair<double, double> vitLoglikelihoodForPairRange(std::pair<unsigned int, unsigned int> sentPairRange,
                                                         int verbosity = 0);
  // The same as the previous one, but Viterbi alignments are
  // computed
  void clearInfoAboutSentRange();
  // clear info about the whole sentence range without clearing
  // information about current model parameters

  // Functions to set model factors

  void setLexSmIntFactor(double _lexSmoothInterpFactor, int verbose = 0);
  // Sets lexical smoothing interpolation factor
  void setAlSmIntFactor(double _aligSmoothInterpFactor, int verbose = 0);
  // Sets alignment smoothing interpolation factor

  // Functions to access model parameters

  Prob pts(WordIndex s, WordIndex t);
  // returns p(t|s)
  virtual LgProb logpts(WordIndex s, WordIndex t);
  // returns log(p(t|s))

  // alignment model functions
  Prob aProb(PositionIndex prev_i, PositionIndex slen, PositionIndex i);
  // Returns p(i|prev_i,slen)
  virtual LgProb logaProb(PositionIndex prev_i, PositionIndex slen, PositionIndex i);
  // Returns log(p(i|prev_i,slen))

  // Sentence length model functions
  Prob sentLenProb(unsigned int slen, unsigned int tlen);
  // returns p(tlen|slen)
  LgProb sentLenLgProb(unsigned int slen, unsigned int tlen);

  // Functions to get translations for word
  bool getEntriesForSource(WordIndex s, NbestTableNode<WordIndex>& trgtn);

  // Functions to generate alignments
  virtual LgProb obtainBestAlignmentVecStrCached(const std::vector<std::string>& srcSentenceVector,
                                                 const std::vector<std::string>& trgSentenceVector,
                                                 CachedHmmAligLgProb& cached_logap, WordAligMatrix& bestWaMatrix);
  LgProb obtainBestAlignment(const std::vector<WordIndex>& srcSentIndexVector,
                             const std::vector<WordIndex>& trgSentIndexVector, WordAligMatrix& bestWaMatrix);
  virtual LgProb obtainBestAlignmentCached(const std::vector<WordIndex>& srcSentIndexVector,
                                           const std::vector<WordIndex>& trgSentIndexVector,
                                           CachedHmmAligLgProb& cached_logap, WordAligMatrix& bestWaMatrix);

  // Functions to calculate probabilities for alignments
  LgProb calcLgProbForAlig(const std::vector<WordIndex>& sSent, const std::vector<WordIndex>& tSent,
                           const WordAligMatrix& aligMatrix, int verbose = 0);

  // Scoring functions without giving an alignment
  LgProb calcLgProb(const std::vector<WordIndex>& sSent, const std::vector<WordIndex>& tSent, int verbose = 0);
  LgProb calcLgProbPhr(const std::vector<WordIndex>& sPhr, const std::vector<WordIndex>& tPhr, int verbose = 0);
  // Scoring function for phrase pairs

  // load function
  bool load(const char* prefFileName, int verbose = 0);

  // print function
  bool print(const char* prefFileName, int verbose = 0);

  // clear() function
  void clear();

  // clearTempVars() function
  void clearTempVars();

  void clearSentLengthModel();

  // Destructor
  ~_incrHmmAligModel();

protected:
  const std::size_t ThreadBufferSize = 10000;
  const double ExpValLogMax = -0.01;
  const double ExpValLogMin = -9;
  const double Log1 = log(1.0);

  anjiMatrix lanji;
  anjiMatrix lanji_aux;
  anjm1ip_anjiMatrix lanjm1ip_anji;
  anjm1ip_anjiMatrix lanjm1ip_anji_aux;
  // Data structures for manipulating expected values

  std::string lexNumDenFileExtension;
  // Extensions for input files for loading

  LexCounts lexCounts;
  IncrLexCounts incrLexCounts;
  // EM algorithm auxiliary variables

  typedef std::unordered_map<std::pair<aSourceHmm, PositionIndex>, std::pair<float, float>, ashPidxPairHashF>
      IncrAligCounts;
  typedef std::vector<double> AligCountsEntry;
  typedef OrderedVector<aSourceHmm, AligCountsEntry> AligCounts;
  AligCounts aligCounts;
  IncrAligCounts incrAligCounts;
  CachedHmmAligLgProb cachedAligLogProbs;
  // EM algorithm auxiliary variables

  LexTable* lexTable = NULL;
  // Pointer to table with lexical parameters

  IncrHmmAligTable aligTable;
  // Table with alignment parameters

  WeightedIncrNormSlm sentLengthModel;

  double aligSmoothInterpFactor;
  double lexSmoothInterpFactor;

  // Functions to get sentence pairs
  std::vector<WordIndex> getSrcSent(unsigned int n);
  // get n-th source sentence
  virtual std::vector<WordIndex> extendWithNullWord(const std::vector<WordIndex>& srcWordIndexVec);
  // given a vector with source words, returns a extended vector
  // including extra NULL words
  virtual std::vector<WordIndex> extendWithNullWordAlig(const std::vector<WordIndex>& srcWordIndexVec);
  // the same as the previous one, but it is specific when calculating suff.
  // statistics for the alignment parameters
  PositionIndex getSrcLen(const std::vector<WordIndex>& nsrcWordIndexVec);

  std::vector<WordIndex> getTrgSent(unsigned int n);
  // get n-th target sentence

  bool sentenceLengthIsOk(const std::vector<WordIndex> sentence);

  // Auxiliary functions to load and print models
  bool loadLexSmIntFactor(const char* lexSmIntFactorFile, int verbose);
  bool printLexSmIntFactor(const char* lexSmIntFactorFile, int verbose);
  bool loadAlSmIntFactor(const char* alSmIntFactorFile, int verbose);
  bool printAlSmIntFactor(const char* alSmIntFactorFile, int verbose);

  // Auxiliary scoring functions
  double unsmoothed_logpts(WordIndex s, WordIndex t);
  // Returns log(p(t|s)) without smoothing
  virtual double unsmoothed_logaProb(PositionIndex prev_i, PositionIndex slen, PositionIndex i);
  double cached_logaProb(PositionIndex prev_i, PositionIndex slen, PositionIndex i);
  void nullAligSpecialPar(unsigned int ip, unsigned int slen, aSourceHmm& asHmm, unsigned int& i);
  // Given ip and slen values, returns (asHmm,i) pair expressing a
  // valid alignment with the null word

  void viterbiAlgorithm(const std::vector<WordIndex>& nSrcSentIndexVector,
                        const std::vector<WordIndex>& trgSentIndexVector, std::vector<std::vector<double>>& vitMatrix,
                        std::vector<std::vector<PositionIndex>>& predMatrix);
  // Execute the Viterbi algorithm to obtain the best HMM word
  // alignment
  void viterbiAlgorithmCached(const std::vector<WordIndex>& nSrcSentIndexVector,
                              const std::vector<WordIndex>& trgSentIndexVector, CachedHmmAligLgProb& cached_logap,
                              std::vector<std::vector<double>>& vitMatrix,
                              std::vector<std::vector<PositionIndex>>& predMatrix);
  // Cached version of viterbiAlgorithm()

  double bestAligGivenVitMatricesRaw(const std::vector<std::vector<double>>& vitMatrix,
                                     const std::vector<std::vector<PositionIndex>>& predMatrix,
                                     std::vector<PositionIndex>& bestAlig);
  // Obtain best alignment vector from Viterbi algorithm matrices,
  // index of null word depends on how the source index vector is
  // transformed
  double bestAligGivenVitMatrices(PositionIndex slen, const std::vector<std::vector<double>>& vitMatrix,
                                  const std::vector<std::vector<PositionIndex>>& predMatrix,
                                  std::vector<PositionIndex>& bestAlig);
  // Obtain best alignment vector from Viterbi algorithm matrices,
  // index of null word is zero
  double forwardAlgorithm(const std::vector<WordIndex>& nSrcSentIndexVector,
                          const std::vector<WordIndex>& trgSentIndexVector, int verbose = 0);
  // Execute Forward algorithm to obtain the log-probability of a
  // sentence pair
  double lgProbGivenForwardMatrix(const std::vector<std::vector<double>>& forwardMatrix);
  LgProb calcVitIbm1LgProb(const std::vector<WordIndex>& srcSentIndexVector,
                           const std::vector<WordIndex>& trgSentIndexVector);
  virtual LgProb calcSumIBM1LgProb(const std::vector<WordIndex>& sSent, const std::vector<WordIndex>& tSent,
                                   int verbose);
  LgProb logaProbIbm1(PositionIndex slen, PositionIndex tlen);
  LgProb noisyOrLgProb(const std::vector<WordIndex>& sSent, const std::vector<WordIndex>& tSent, int verbose);

  // EM-related functions
  void calcNewLocalSuffStats(std::pair<unsigned int, unsigned int> sentPairRange, int verbosity = 0);
  void calcNewLocalSuffStatsVit(std::pair<unsigned int, unsigned int> sentPairRange, int verbosity = 0);
  void calcAlphaBetaMatrices(const std::vector<WordIndex>& nsrcSent, const std::vector<WordIndex>& trgSent,
                             PositionIndex slen, std::vector<std::vector<double>>& cachedLexLogProbs,
                             std::vector<std::vector<double>>& alphaMatrix,
                             std::vector<std::vector<double>>& betaMatrix);
  void calc_lanji(unsigned int n, const std::vector<WordIndex>& nsrcSent, const std::vector<WordIndex>& trgSent,
                  PositionIndex slen, const Count& weight, const std::vector<std::vector<double>>& alphaMatrix,
                  const std::vector<std::vector<double>>& betaMatrix);
  void calc_lanji_vit(unsigned int n, const std::vector<WordIndex>& nsrcSent, const std::vector<WordIndex>& trgSent,
                      const std::vector<PositionIndex>& bestAlig, const Count& weight);
  void incrUpdateCountsLex(unsigned int mapped_n, unsigned int mapped_n_aux, PositionIndex i, PositionIndex j,
                           const std::vector<WordIndex>& nsrcSent, const std::vector<WordIndex>& trgSent,
                           const Count& weight);
  void calc_lanjm1ip_anji(unsigned int n, const std::vector<WordIndex>& nsrcSent, const std::vector<WordIndex>& trgSent,
                          PositionIndex slen, const Count& weight,
                          const std::vector<std::vector<double>>& cachedLexLogProbs,
                          const std::vector<std::vector<double>>& alphaMatrix,
                          const std::vector<std::vector<double>>& betaMatrix);
  void calc_lanjm1ip_anji_vit(unsigned int n, const std::vector<WordIndex>& nsrcSent,
                              const std::vector<WordIndex>& trgSent, PositionIndex slen,
                              const std::vector<PositionIndex>& bestAlig, const Count& weight);
  bool isFirstNullAligPar(PositionIndex ip, unsigned int slen, PositionIndex i);
  double calc_lanji_num(PositionIndex i, PositionIndex j, const std::vector<std::vector<double>>& alphaMatrix,
                        const std::vector<std::vector<double>>& betaMatrix);
  double calc_lanjm1ip_anji_num_je1(PositionIndex slen, PositionIndex i,
                                    const std::vector<std::vector<double>>& cachedLexLogProbs,
                                    const std::vector<std::vector<double>>& betaMatrix);
  double calc_lanjm1ip_anji_num_jg1(PositionIndex ip, PositionIndex slen, PositionIndex i, PositionIndex j,
                                    const std::vector<std::vector<double>>& cachedLexLogProbs,
                                    const std::vector<std::vector<double>>& alphaMatrix,
                                    const std::vector<std::vector<double>>& betaMatrix);
  void gatherLexSuffStats(unsigned int mapped_n, unsigned int mapped_n_aux, const std::vector<WordIndex>& nsrcSent,
                          const std::vector<WordIndex>& trgSent, const Count& weight);
  void gatherAligSuffStats(unsigned int mapped_n, unsigned int mapped_n_aux, const std::vector<WordIndex>& nsrcSent,
                           const std::vector<WordIndex>& trgSent, PositionIndex slen, const Count& weight);
  void incrUpdateCountsAlig(unsigned int mapped_n, unsigned int mapped_n_aux, PositionIndex slen, PositionIndex ip,
                            PositionIndex i, PositionIndex j, const Count& weight);
  void getHmmAligInfo(PositionIndex ip, unsigned int slen, PositionIndex i, HmmAligInfo& hmmAligInfo);
  bool isValidAlig(PositionIndex ip, unsigned int slen, PositionIndex i);
  bool isNullAlig(PositionIndex ip, unsigned int slen, PositionIndex i);
  PositionIndex getModifiedIp(PositionIndex ip, unsigned int slen, PositionIndex i);
  void incrMaximizeProbsLex();
  void incrMaximizeProbsAlig();
  virtual float obtainLogNewSuffStat(float lcurrSuffStat, float lLocalSuffStatCurr, float lLocalSuffStatNew);

  void addTranslationOptions(std::vector<std::vector<WordIndex>>& insertBuffer);
  void batchUpdateCounts(const std::vector<std::pair<std::vector<WordIndex>, std::vector<WordIndex>>>& pairs);
  void batchMaximizeProbs();
};
