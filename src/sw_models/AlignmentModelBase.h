#pragma once

#include "nlp_common/SingleWordVocab.h"
#include "sw_models/AlignmentModel.h"
#include "sw_models/LightSentenceHandler.h"

#include <memory>
#include <set>

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

  // Functions to print sentence pairs
  bool printSentencePairs(const char* srcSentFile, const char* trgSentFile, const char* sentCountsFile) override;

  // Returns log-likelihood. The first double contains the
  // loglikelihood for all sentences, and the second one, the same
  // loglikelihood normalized by the number of sentences
  std::pair<double, double> loglikelihoodForAllSentences(int verbosity = 0) override;

  // Scoring functions for a given alignment
  using AlignmentModel::getAlignmentLgProb;
  LgProb getAlignmentLgProb(const char* srcSentence, const char* trgSentence, const WordAlignmentMatrix& aligMatrix,
                            int verbose = 0) override;
  LgProb getAlignmentLgProb(const std::vector<std::string>& srcSentence, const std::vector<std::string>& trgSentence,
                            const WordAlignmentMatrix& aligMatrix, int verbose = 0) override;

  // Scoring functions without giving an alignment
  using AlignmentModel::getSumLgProb;
  LgProb getSumLgProb(const char* srcSentence, const char* trgSentence, int verbose = 0) override;
  LgProb getSumLgProb(const std::vector<std::string>& srcSentence, const std::vector<std::string>& trgSentence,
                      int verbose = 0) override;
  LgProb getPhraseSumLgProb(const std::vector<WordIndex>& srcPhrase, const std::vector<WordIndex>& trgPhrase,
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

  void clear() override;

  virtual ~AlignmentModelBase()
  {
  }

protected:
  AlignmentModelBase();
  AlignmentModelBase(AlignmentModelBase& model);

  bool printVariationalBayes(const std::string& filename);
  bool loadVariationalBayes(const std::string& filename);

  double alpha;
  bool variationalBayes;
  std::shared_ptr<SingleWordVocab> swVocab;
  std::shared_ptr<LightSentenceHandler> sentenceHandler;
};
