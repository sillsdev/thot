#pragma once

#include "sw_models/LexCounts.h"
#include "sw_models/LexTable.h"
#include "sw_models/WeightedIncrNormSlm.h"
#include "sw_models/_incrSwAligModel.h"
#include "sw_models/anjiMatrix.h"

#include <memory>
#include <unordered_map>

class Ibm1AligModel : public _swAligModel
{
  friend class IncrIbm1AligTrainer;

public:
  Ibm1AligModel();

  void startTraining(int verbosity = 0);
  void train(int verbosity = 0);
  void endTraining();

  // Returns log-likelihood. The first double contains the
  // loglikelihood for all sentences, and the second one, the same
  // loglikelihood normalized by the number of sentences
  std::pair<double, double> loglikelihoodForPairRange(std::pair<unsigned int, unsigned int> sentPairRange,
                                                      int verbosity = 0);

  // returns p(t|s)
  Prob pts(WordIndex s, WordIndex t);
  // returns log(p(t|s))
  LgProb logpts(WordIndex s, WordIndex t);

  // alignment model functions
  Prob aProbIbm1(PositionIndex slen, PositionIndex tlen);
  LgProb logaProbIbm1(PositionIndex slen, PositionIndex tlen);

  // Sentence length model functions
  Prob sentLenProb(PositionIndex slen, PositionIndex tlen);
  LgProb sentLenLgProb(PositionIndex slen, PositionIndex tlen);

  bool getEntriesForSource(WordIndex s, NbestTableNode<WordIndex>& trgtn);

  LgProb obtainBestAlignment(const std::vector<WordIndex>& src, const std::vector<WordIndex>& trg,
                             WordAligMatrix& bestWaMatrix);
  LgProb calcLgProbForAlig(const std::vector<WordIndex>& src, const std::vector<WordIndex>& trg,
                           const WordAligMatrix& aligMatrix, int verbose = 0);
  LgProb calcLgProb(const std::vector<WordIndex>& src, const std::vector<WordIndex>& trg, int verbose = 0);

  bool load(const char* prefFileName, int verbose = 0);
  bool print(const char* prefFileName, int verbose = 0);

  void clear();
  void clearTempVars();
  void clearSentLengthModel();
  // clear info about the whole sentence range without clearing
  // information about current model parameters
  void clearInfoAboutSentRange();

  virtual ~Ibm1AligModel()
  {
  }

protected:
  const std::size_t ThreadBufferSize = 10000;

  Ibm1AligModel(Ibm1AligModel& model);

  std::vector<WordIndex> getSrcSent(unsigned int n);

  // given a vector with source words, returns a extended vector including extra NULL words
  std::vector<WordIndex> extendWithNullWord(const std::vector<WordIndex>& srcWordIndexVec);

  std::vector<WordIndex> getTrgSent(unsigned int n);

  virtual bool sentenceLengthIsOk(const std::vector<WordIndex> sentence);

  double unsmoothed_pts(WordIndex s, WordIndex t);
  double unsmoothed_logpts(WordIndex s, WordIndex t);

  LgProb lexM1LpForBestAlig(const std::vector<WordIndex>& nSrcSentIndexVector,
                            const std::vector<WordIndex>& trgSentIndexVector, std::vector<PositionIndex>& bestAlig);
  LgProb calcIbm1LgProbForAlig(const std::vector<WordIndex>& nsSent, const std::vector<WordIndex>& tSent,
                               const std::vector<PositionIndex>& alig, int verbose = 0);
  LgProb calcSumIbm1LgProb(const std::vector<WordIndex>& nsSent, const std::vector<WordIndex>& tSent, int verbose = 0);

  // Batch EM functions
  virtual void initSourceWord(const std::vector<WordIndex>& nsrc, const std::vector<WordIndex>& trg, PositionIndex i);
  virtual void initTargetWord(const std::vector<WordIndex>& nsrc, const std::vector<WordIndex>& trg, PositionIndex j);
  virtual void initWordPair(const std::vector<WordIndex>& nsrc, const std::vector<WordIndex>& trg, PositionIndex i,
                            PositionIndex j);
  virtual void addTranslationOptions(std::vector<std::vector<WordIndex>>& insertBuffer);
  virtual void batchUpdateCounts(const std::vector<std::pair<std::vector<WordIndex>, std::vector<WordIndex>>>& pairs);
  virtual double getCountNumerator(const std::vector<WordIndex>& nsrc, const std::vector<WordIndex>& trg,
                                   PositionIndex i, PositionIndex j);
  virtual void incrementWordPairCounts(const std::vector<WordIndex>& nsrc, const std::vector<WordIndex>& trg,
                                       PositionIndex i, PositionIndex j, double count);
  virtual void batchMaximizeProbs();

  // model parameters
  std::shared_ptr<WeightedIncrNormSlm> sentLengthModel;
  std::shared_ptr<LexTable> lexTable;

  // EM counts
  LexCounts lexCounts;
};
