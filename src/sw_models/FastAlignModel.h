#pragma once

#include "sw_models/AlignmentModelBase.h"
#include "sw_models/IncrAlignmentModel.h"
#include "sw_models/LexCounts.h"
#include "sw_models/MemoryLexTable.h"
#include "sw_models/anjiMatrix.h"

struct PairLess
{
  bool operator()(const std::pair<short, short>& x, const std::pair<short, short>& y) const
  {
    if (x.first < y.first)
      return true;
    if (x.first == y.first && x.second < y.second)
      return true;
    return false;
  }
};

class FastAlignModel : public AlignmentModelBase, public virtual IncrAlignmentModel
{
public:
  FastAlignModel();

  AlignmentModelType getModelType() const override
  {
    return FastAlign;
  }

  void set_expval_maxnsize(unsigned int _anji_maxnsize) override;
  double getFastAlignP0() const;
  void setFastAlignP0(double value);

  unsigned int startTraining(int verbosity = 0) override;
  void train(int verbosity = 0) override;
  void endTraining() override;

  void startIncrTraining(std::pair<unsigned int, unsigned int> sentPairRange, int verbosity = 0) override;
  void incrTrain(std::pair<unsigned int, unsigned int> sentPairRange, int verbosity = 0) override;
  void endIncrTraining() override;

  std::pair<double, double> loglikelihoodForPairRange(std::pair<unsigned int, unsigned int> sentPairRange,
                                                      int verbosity = 0) override;

  Prob translationProb(WordIndex s, WordIndex t) override;
  LgProb translationLogProb(WordIndex s, WordIndex t) override;

  Prob alignmentProb(PositionIndex j, PositionIndex slen, PositionIndex tlen, PositionIndex i);
  LgProb alignmentLogProb(PositionIndex j, PositionIndex slen, PositionIndex tlen, PositionIndex i);

  Prob sentenceLengthProb(unsigned int slen, unsigned int tlen) override;
  LgProb sentenceLengthLogProb(unsigned int slen, unsigned int tlen) override;

  // Functions to get translations for word
  bool getEntriesForSource(WordIndex s, NbestTableNode<WordIndex>& trgtn) override;

  using AlignmentModel::getBestAlignment;
  LgProb getBestAlignment(const std::vector<WordIndex>& srcSentence, const std::vector<WordIndex>& trgSentence,
                          std::vector<PositionIndex>& bestAlignment) override;
  using AlignmentModel::computeLogProb;
  LgProb computeLogProb(const std::vector<WordIndex>& srcSentence, const std::vector<WordIndex>& trgSentence,
                        const WordAlignmentMatrix& aligMatrix, int verbose = 0) override;
  using AlignmentModel::computeSumLogProb;
  LgProb computeSumLogProb(const std::vector<WordIndex>& srcSentence, const std::vector<WordIndex>& trgSentence,
                           int verbose = 0) override;

  bool load(const char* prefFileName, int verbose = 0) override;
  bool print(const char* prefFileName, int verbose = 0) override;

  void clearSentenceLengthModel() override;
  void clearTempVars() override;
  void clear() override;

  virtual ~FastAlignModel()
  {
  }

private:
  typedef OrderedVector<std::pair<short, short>, unsigned int, PairLess> SizeCounts;

  const std::size_t ThreadBufferSize = 10000;
  const float SmoothingAnjiNum = 1e-9f;
  const float SmoothingWeightedAnji = 1e-9f;
  const float SmoothingProb = 1e-9f;
  const float SmoothingLogProb = log(SmoothingProb);
  const double ArbitraryPts = 0.05;
  const double DefaultFastAlignP0 = 0.08;

  std::string getModelTypeStr() const override
  {
    return "fastAlign";
  }

  void addTranslationOptions(std::vector<std::vector<WordIndex>>& insertBuffer);
  void batchUpdateCounts(const std::vector<std::pair<std::vector<WordIndex>, std::vector<WordIndex>>>& pairs);
  std::vector<WordIndex> getSrcSent(unsigned int n);
  std::vector<WordIndex> getTrgSent(unsigned int n);
  double computeAZ(PositionIndex j, PositionIndex slen, PositionIndex tlen);
  Prob alignmentProb(double az, PositionIndex j, PositionIndex slen, PositionIndex tlen, PositionIndex i);
  bool printParams(const std::string& filename);
  bool loadParams(const std::string& filename);
  bool printSizeCounts(const std::string& filename);
  bool loadSizeCounts(const std::string& filename);
  void batchMaximizeProbs();
  void optimizeDiagonalTension(unsigned int nIters, int verbose);
  void incrementSizeCount(unsigned int tlen, unsigned int slen);
  void initCountSlot(WordIndex s, WordIndex t);
  void incrementCount(WordIndex s, WordIndex t, double x);

  void calcNewLocalSuffStats(std::pair<unsigned int, unsigned int> sentPairRange, int verbosity = 0);
  void calc_anji(unsigned int n, const std::vector<WordIndex>& nsrcSent, const std::vector<WordIndex>& trgSent,
                 const Count& weight);
  double calc_anji_num(double az, const std::vector<WordIndex>& nsrcSent, const std::vector<WordIndex>& trgSent,
                       unsigned int i, unsigned int j);
  void incrUpdateCounts(unsigned int mapped_n, unsigned int mapped_n_aux, PositionIndex i, PositionIndex j,
                        const std::vector<WordIndex>& nsrcSent, const std::vector<WordIndex>& trgSent,
                        const Count& weight);
  void incrMaximizeProbs(void);
  float obtainLogNewSuffStat(float lcurrSuffStat, float lLocalSuffStatCurr, float lLocalSuffStatNew);

  void loadConfig(const YAML::Node& config) override;
  void createConfig(YAML::Emitter& out) override;

  double fastAlignP0 = DefaultFastAlignP0;

  MemoryLexTable lexTable;
  double diagonalTension = 4.0;
  double totLenRatio = 0;
  double empFeatSum = 0;
  double trgTokenCount = 0;
  SizeCounts sizeCounts;
  anjiMatrix anji;

  anjiMatrix anji_aux;
  LexCounts lexCounts;
  IncrLexCounts incrLexCounts;
  int iter = 0;
};
