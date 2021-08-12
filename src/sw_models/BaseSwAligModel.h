#pragma once

#include "nlp_common/Count.h"
#include "nlp_common/NbestTableNode.h"
#include "nlp_common/Prob.h"
#include "nlp_common/WordAligMatrix.h"
#include "nlp_common/WordIndex.h"

class BaseSwAligModel
{
public:
  // Thread/Process safety related functions
  virtual bool modelReadsAreProcessSafe() = 0;

  virtual void setVariationalBayes(bool variationalBayes) = 0;
  virtual bool getVariationalBayes() = 0;

  // Functions to read and add sentence pairs
  virtual bool readSentencePairs(const char* srcFileName, const char* trgFileName, const char* sentCountsFile,
                                 std::pair<unsigned int, unsigned int>& sentRange, int verbose = 0) = 0;
  virtual void addSentPair(std::vector<std::string> srcSentStr, std::vector<std::string> trgSentStr, Count c,
                           std::pair<unsigned int, unsigned int>& sentRange) = 0;
  virtual unsigned int numSentPairs(void) = 0;
  // NOTE: the whole valid range in a given moment is
  // [ 0 , numSentPairs() )
  virtual int nthSentPair(unsigned int n, std::vector<std::string>& srcSentStr, std::vector<std::string>& trgSentStr,
                          Count& c) = 0;

  // Functions to print sentence pairs
  virtual bool printSentPairs(const char* srcSentFile, const char* trgSentFile, const char* sentCountsFile) = 0;

  // Functions to train model
  virtual void startTraining(int verbosity = 0) = 0;
  virtual void train(int verbosity = 0) = 0;
  virtual void endTraining() = 0;
  virtual std::pair<double, double> loglikelihoodForPairRange(std::pair<unsigned int, unsigned int> sentPairRange,
                                                              int verbosity = 0) = 0;
  // Returns log-likelihood. The first double contains the
  // loglikelihood for all sentences, and the second one, the same
  // loglikelihood normalized by the number of sentences
  virtual std::pair<double, double> loglikelihoodForAllSents(int verbosity = 0) = 0;

  // Sentence length model functions
  virtual Prob sentLenProb(unsigned int slen, unsigned int tlen) = 0;
  // returns p(tlen|slen)
  virtual LgProb sentLenLgProb(unsigned int slen, unsigned int tlen) = 0;

  // Scoring functions for a given alignment
  virtual LgProb calcLgProbForAligChar(const char* sSent, const char* tSent, const WordAligMatrix& aligMatrix,
                                       int verbose = 0) = 0;
  virtual LgProb calcLgProbForAligVecStr(const std::vector<std::string>& sSent, const std::vector<std::string>& tSent,
                                         const WordAligMatrix& aligMatrix, int verbose = 0) = 0;
  virtual LgProb calcLgProbForAlig(const std::vector<WordIndex>& sSent, const std::vector<WordIndex>& tSent,
                                   const WordAligMatrix& aligMatrix, int verbose = 0) = 0;

  // Scoring functions without giving an alignment
  virtual LgProb calcLgProbChar(const char* sSent, const char* tSent, int verbose = 0) = 0;
  virtual LgProb calcLgProbVecStr(const std::vector<std::string>& sSent, const std::vector<std::string>& tSent,
                                  int verbose = 0) = 0;
  virtual LgProb calcLgProb(const std::vector<WordIndex>& sSent, const std::vector<WordIndex>& tSent,
                            int verbose = 0) = 0;
  virtual LgProb calcLgProbPhr(const std::vector<WordIndex>& sPhr, const std::vector<WordIndex>& tPhr,
                               int verbose = 0) = 0;
  // Scoring function for phrase pairs

  // Best-alignment functions
  virtual bool obtainBestAlignments(const char* sourceTestFileName, const char* targetTestFilename,
                                    const char* outFileName) = 0;
  // Obtains the best alignments for the sentence pairs given in
  // the files 'sourceTestFileName' and 'targetTestFilename'. The
  // results are stored in the file 'outFileName'
  virtual LgProb obtainBestAlignmentChar(const char* sourceSentence, const char* targetSentence,
                                         WordAligMatrix& bestWaMatrix) = 0;
  // Obtains the best alignment for the given sentence pair
  virtual LgProb obtainBestAlignmentVecStr(const std::vector<std::string>& srcSentenceVector,
                                           const std::vector<std::string>& trgSentenceVector,
                                           WordAligMatrix& bestWaMatrix) = 0;
  // Obtains the best alignment for the given sentence pair (input
  // parameters are now string vectors)
  virtual LgProb obtainBestAlignment(const std::vector<WordIndex>& srcSentIndexVector,
                                     const std::vector<WordIndex>& trgSentIndexVector,
                                     WordAligMatrix& bestWaMatrix) = 0;
  // Obtains the best alignment for the given sentence pair
  // (input parameters are now index vectors) depending on the
  // value of the modelNumber data member.

  virtual std::ostream& printAligInGizaFormat(const char* sourceSentence, const char* targetSentence, Prob p,
                                              std::vector<PositionIndex> alig, std::ostream& outS) = 0;
  // Prints the given alignment to 'outS' stream in GIZA format

  // load() function
  virtual bool load(const char* prefFileName, int verbose = 0) = 0;

  // print() function
  virtual bool print(const char* prefFileName, int verbose = 0) = 0;

  // Functions for loading vocabularies
  virtual bool loadGIZASrcVocab(const char* srcInputVocabFileName, int verbose = 0) = 0;
  // Reads source vocabulary from a file in GIZA format
  virtual bool loadGIZATrgVocab(const char* trgInputVocabFileName, int verbose = 0) = 0;
  // Reads target vocabulary from a file in GIZA format

  // Functions for printing vocabularies
  virtual bool printGIZASrcVocab(const char* srcOutputVocabFileName) = 0;
  // Reads source vocabulary from a file in GIZA format
  virtual bool printGIZATrgVocab(const char* trgOutputVocabFileName) = 0;
  // Reads target vocabulary from a file in GIZA format

  // Source and target vocabulary functions
  virtual size_t getSrcVocabSize() const = 0;
  // Returns the source vocabulary size
  virtual WordIndex stringToSrcWordIndex(std::string s) const = 0;
  virtual std::string wordIndexToSrcString(WordIndex w) const = 0;
  virtual bool existSrcSymbol(std::string s) const = 0;
  virtual std::vector<WordIndex> strVectorToSrcIndexVector(std::vector<std::string> s) = 0;
  virtual WordIndex addSrcSymbol(std::string s) = 0;

  virtual size_t getTrgVocabSize() const = 0;
  // Returns the target vocabulary size
  virtual WordIndex stringToTrgWordIndex(std::string t) const = 0;
  virtual std::string wordIndexToTrgString(WordIndex w) const = 0;
  virtual bool existTrgSymbol(std::string t) const = 0;
  virtual std::vector<WordIndex> strVectorToTrgIndexVector(std::vector<std::string> t) = 0;
  virtual WordIndex addTrgSymbol(std::string t) = 0;

  // Functions to get translations for word
  virtual bool getEntriesForSource(WordIndex s, NbestTableNode<WordIndex>& trgtn) = 0;

  // Utilities
  virtual std::vector<WordIndex> addNullWordToWidxVec(const std::vector<WordIndex>& vw) = 0;
  virtual std::vector<std::string> addNullWordToStrVec(const std::vector<std::string>& vw) = 0;

  virtual void clear() = 0;

  virtual void clearTempVars(){};

  virtual void clearSentLengthModel() = 0;

  // clear info about the whole sentence range without clearing
  // information about current model parameters
  virtual void clearInfoAboutSentRange() = 0;

  virtual Prob pts(WordIndex s, WordIndex t) = 0;
  virtual LgProb logpts(WordIndex s, WordIndex t) = 0;

  // Destructor
  virtual ~BaseSwAligModel()
  {
  }
};
