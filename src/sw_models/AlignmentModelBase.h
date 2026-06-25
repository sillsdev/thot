#pragma once

#include "nlp_common/SingleWordVocab.h"
#include "nlp_common/WordClasses.h"
#include "sw_models/AlignmentModel.h"
#include "sw_models/LightSentenceHandler.h"

#include <memory>
#include <set>
#include <yaml-cpp/yaml.h>

class AlignmentModelBase : public virtual AlignmentModel
{
public:
  // Thread/Process safety related functions
  bool modelReadsAreProcessSafe() override;

  void setVariationalBayes(bool variationalBayes) override;
  bool getVariationalBayes() override;

  // Functions to read and add sentence pairs
  bool readSentencePairs(const char* srcFileName, const char* trgFileName, const char* sentCountsFile,
                         std::pair<unsigned int, unsigned int>& sentRange, int verbose = 0) override;
  std::pair<unsigned int, unsigned int> addSentencePair(std::vector<std::string> srcSentStr,
                                                        std::vector<std::string> trgSentStr, Count c) override;
  unsigned int numSentencePairs() override;
  // NOTE: the whole valid range in a given moment is
  // [ 0 , numSentPairs() )
  int getSentencePair(unsigned int n, std::vector<std::string>& srcSentStr, std::vector<std::string>& trgSentStr,
                      Count& c) override;

  PositionIndex getMaxSentenceLength() override;

  // Functions to print sentence pairs
  bool printSentencePairs(const char* srcSentFile, const char* trgSentFile, const char* sentCountsFile) override;

  // Returns log-likelihood. The first double contains the
  // loglikelihood for all sentences, and the second one, the same
  // loglikelihood normalized by the number of sentences
  std::pair<double, double> loglikelihoodForAllSentences(int verbosity = 0) override;

  // Scoring functions for a given alignment
  using AlignmentModel::computeLogProb;
  LgProb computeLogProb(const char* srcSentence, const char* trgSentence, const WordAlignmentMatrix& aligMatrix,
                        int verbose = 0) override;
  LgProb computeLogProb(const std::vector<std::string>& srcSentence, const std::vector<std::string>& trgSentence,
                        const WordAlignmentMatrix& aligMatrix, int verbose = 0) override;

  // Scoring functions without giving an alignment
  using AlignmentModel::computeSumLogProb;
  LgProb computeSumLogProb(const char* srcSentence, const char* trgSentence, int verbose = 0) override;
  LgProb computeSumLogProb(const std::vector<std::string>& srcSentence, const std::vector<std::string>& trgSentence,
                           int verbose = 0) override;
  LgProb computePhraseSumLogProb(const std::vector<WordIndex>& srcPhrase, const std::vector<WordIndex>& trgPhrase,
                                 int verbose = 0) override;

  // Best-alignment functions
  bool getBestAlignments(const char* sourceTestFileName, const char* targetTestFilename,
                         const char* outFileName) override;
  using AlignmentModel::getBestAlignment;
  // Obtains the best alignments for the sentence pairs given in
  // the files 'sourceTestFileName' and 'targetTestFilename'. The
  // results are stored in the file 'outFileName'
  LgProb getBestAlignment(const char* srcSentence, const char* trgSentence, WordAlignmentMatrix& bestWaMatrix) override;
  // Obtains the best alignment for the given sentence pair
  LgProb getBestAlignment(const std::vector<std::string>& srcSentence, const std::vector<std::string>& trgSentence,
                          WordAlignmentMatrix& bestWaMatrix) override;
  // Obtains the best alignment for the given sentence pair (input
  // parameters are now string vectors)
  LgProb getBestAlignment(const std::vector<WordIndex>& srcSentence, const std::vector<WordIndex>& trgSentence,
                          WordAlignmentMatrix& bestWaMatrix) override;
  LgProb getBestAlignment(const char* srcSentence, const char* trgSentence,
                          std::vector<PositionIndex>& bestAlignment) override;
  LgProb getBestAlignment(const std::vector<std::string>& srcSentence, const std::vector<std::string>& trgSentence,
                          std::vector<PositionIndex>& bestAlignment) override;
  // Obtains the best alignment for the given sentence pair (input
  // parameters are now string vectors)

  std::ostream& printAligInGizaFormat(const char* sourceSentence, const char* targetSentence, Prob p,
                                      std::vector<PositionIndex> alig, std::ostream& outS) override;
  // Prints the given alignment to 'outS' stream in GIZA format

  // Functions for loading vocabularies
  bool loadGIZASrcVocab(const char* srcInputVocabFileName, int verbose = 0) override;
  // Reads source vocabulary from a file in GIZA format
  bool loadGIZATrgVocab(const char* trgInputVocabFileName, int verbose = 0) override;
  // Reads target vocabulary from a file in GIZA format

  // Functions for printing vocabularies
  bool printGIZASrcVocab(const char* srcOutputVocabFileName) override;
  // Reads source vocabulary from a file in GIZA format
  bool printGIZATrgVocab(const char* trgOutputVocabFileName) override;
  // Reads target vocabulary from a file in GIZA format

  // Source and target vocabulary functions
  size_t getSrcVocabSize() const override; // Returns the source vocabulary size
  WordIndex stringToSrcWordIndex(std::string s) const override;
  std::string wordIndexToSrcString(WordIndex w) const override;
  bool existSrcSymbol(std::string s) const override;
  std::vector<WordIndex> strVectorToSrcIndexVector(std::vector<std::string> s) override;
  WordIndex addSrcSymbol(std::string s) override;

  size_t getTrgVocabSize() const override; // Returns the target vocabulary size
  WordIndex stringToTrgWordIndex(std::string t) const override;
  std::string wordIndexToTrgString(WordIndex w) const override;
  bool existTrgSymbol(std::string t) const override;
  std::vector<WordIndex> strVectorToTrgIndexVector(std::vector<std::string> t) override;
  WordIndex addTrgSymbol(std::string t) override;

  // Utilities
  std::vector<WordIndex> addNullWordToWidxVec(const std::vector<WordIndex>& vw) override;
  std::vector<std::string> addNullWordToStrVec(const std::vector<std::string>& vw) override;

  WordClassIndex addSrcWordClass(const std::string& c) override;
  WordClassIndex addTrgWordClass(const std::string& c) override;
  void mapSrcWordToWordClass(WordIndex s, const std::string& c) override;
  void mapSrcWordToWordClass(WordIndex s, WordClassIndex c) override;
  void mapTrgWordToWordClass(WordIndex t, const std::string& c) override;
  void mapTrgWordToWordClass(WordIndex t, WordClassIndex c) override;

  // Default end-of-training: emit training alignments (if enabled) then clear
  // temporary state. Models that need to finalize parameters first (e.g. Eflomal
  // normalizing its chains) override and chain to this for the shared tail.
  void endTraining() override;

  // Training-alignment functions (see AlignmentModel for semantics). The flag
  // and storage live here so every model gains the capability; the default
  // computeTrainingAlignments() produces them via getBestAlignment, and models
  // with a cheaper/better source (e.g. Eflomal's warm argmax) override it.
  void setEmitTrainingAlignments(bool value) override;
  bool getEmitTrainingAlignments() override;
  LgProb getTrainingAlignment(size_t n, std::vector<PositionIndex>& alignment) override;
  LgProb getTrainingAlignment(size_t n, WordAlignmentMatrix& bestWaMatrix) override;

  bool load(const char* prefFileName, int verbose = 0) override;
  bool print(const char* prefFileName, int verbose = 0) override;

  void clear() override;
  // clear info about the whole sentence range without clearing
  // information about current model parameters
  void clearInfoAboutSentenceRange() override;

  virtual ~AlignmentModelBase()
  {
  }

protected:
  AlignmentModelBase();
  AlignmentModelBase(AlignmentModelBase& model);

  bool loadVariationalBayes(const std::string& filename);
  bool sentenceLengthIsOk(const std::vector<WordIndex> sentence);

  // Fills trainingAlignments (per training pair that passes the length filter,
  // in added order) when emitTrainingAlignments is set; a no-op otherwise. The
  // default uses getBestAlignment on each pair; called by each model's
  // endTraining before its temporary state is cleared. Models override to use a
  // training-time alignment instead of recomputing it; an override must index
  // trainingAlignments by sentence-handler pair index (empty for filtered pairs).
  virtual void computeTrainingAlignments();
  // The sentence-handler indices of the pairs that pass the length filter, in
  // order. Lets a model that works over the filtered corpus map a corpus position
  // back to its sentence-handler pair index.
  std::vector<unsigned int> trainingPairSentenceIndices();
  // Writes/restores trainingAlignments as 0-based Pharaoh links in
  // "<prefix>.aligns". Called from print()/load(); the read reconstructs the
  // exact per-target form (including trailing NULL-aligned targets that Pharaoh
  // omits) by replaying the length filter to recover each pair's target length.
  bool printTrainingAlignments(const char* prefFileName);
  bool loadTrainingAlignments(const char* prefFileName);

  virtual std::string getModelTypeStr() const = 0;

  virtual void loadConfig(const YAML::Node& config);
  virtual bool loadOldConfig(const char* prefFileName, int verbose = 0);
  virtual void createConfig(YAML::Emitter& out);

  bool emitTrainingAlignments = false;
  // Indexed by sentence-handler pair index (matching getSentencePair); a pair that
  // was filtered out of training holds an empty alignment.
  std::vector<std::vector<PositionIndex>> trainingAlignments;

  PositionIndex maxSentenceLength = 1024;
  double alpha;
  bool variationalBayes;
  std::shared_ptr<SingleWordVocab> swVocab;
  std::shared_ptr<LightSentenceHandler> sentenceHandler;
  std::shared_ptr<WordClasses> wordClasses;
};
