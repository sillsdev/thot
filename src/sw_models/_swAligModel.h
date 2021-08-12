#pragma once

#include "nlp_common/SingleWordVocab.h"
#include "sw_models/BaseSwAligModel.h"
#include "sw_models/LightSentenceHandler.h"

#include <memory>
#include <set>

class _swAligModel : public virtual BaseSwAligModel
{
public:
  // Thread/Process safety related functions
  bool modelReadsAreProcessSafe() override;

  void setVariationalBayes(bool variationalBayes) override;
  bool getVariationalBayes() override;

  // Functions to read and add sentence pairs
  bool readSentencePairs(const char* srcFileName, const char* trgFileName, const char* sentCountsFile,
                         std::pair<unsigned int, unsigned int>& sentRange, int verbose = 0) override;
  void addSentPair(std::vector<std::string> srcSentStr, std::vector<std::string> trgSentStr, Count c,
                   std::pair<unsigned int, unsigned int>& sentRange) override;
  unsigned int numSentPairs() override;
  // NOTE: the whole valid range in a given moment is
  // [ 0 , numSentPairs() )
  int nthSentPair(unsigned int n, std::vector<std::string>& srcSentStr, std::vector<std::string>& trgSentStr,
                  Count& c) override;

  // Functions to print sentence pairs
  bool printSentPairs(const char* srcSentFile, const char* trgSentFile, const char* sentCountsFile) override;

  // Returns log-likelihood. The first double contains the
  // loglikelihood for all sentences, and the second one, the same
  // loglikelihood normalized by the number of sentences
  std::pair<double, double> loglikelihoodForAllSents(int verbosity = 0) override;

  // Scoring functions for a given alignment
  LgProb calcLgProbForAligChar(const char* sSent, const char* tSent, const WordAligMatrix& aligMatrix,
                               int verbose = 0) override;
  LgProb calcLgProbForAligVecStr(const std::vector<std::string>& sSent, const std::vector<std::string>& tSent,
                                 const WordAligMatrix& aligMatrix, int verbose = 0) override;

  // Scoring functions without giving an alignment
  LgProb calcLgProbChar(const char* sSent, const char* tSent, int verbose = 0) override;
  LgProb calcLgProbVecStr(const std::vector<std::string>& sSent, const std::vector<std::string>& tSent,
                          int verbose = 0) override;
  LgProb calcLgProbPhr(const std::vector<WordIndex>& sPhr, const std::vector<WordIndex>& tPhr,
                       int verbose = 0) override;

  // Best-alignment functions
  bool obtainBestAlignments(const char* sourceTestFileName, const char* targetTestFilename,
                            const char* outFileName) override;
  // Obtains the best alignments for the sentence pairs given in
  // the files 'sourceTestFileName' and 'targetTestFilename'. The
  // results are stored in the file 'outFileName'
  LgProb obtainBestAlignmentChar(const char* sourceSentence, const char* targetSentence,
                                 WordAligMatrix& bestWaMatrix) override;
  // Obtains the best alignment for the given sentence pair
  LgProb obtainBestAlignmentVecStr(const std::vector<std::string>& srcSentenceVector,
                                   const std::vector<std::string>& trgSentenceVector,
                                   WordAligMatrix& bestWaMatrix) override;
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

  virtual ~_swAligModel()
  {
  }

protected:
  _swAligModel();
  _swAligModel(_swAligModel& model);

  bool printVariationalBayes(const std::string& filename);
  bool loadVariationalBayes(const std::string& filename);

  double alpha;
  bool variationalBayes;
  std::shared_ptr<SingleWordVocab> swVocab;
  std::shared_ptr<LightSentenceHandler> sentenceHandler;
};
