/*
thot package for statistical machine translation
Copyright (C) 2013 Daniel Ortiz-Mart\'inez

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public License
as published by the Free Software Foundation; either version 3
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program; If not, see <http://www.gnu.org/licenses/>.
*/

/**
 * @file BaseSwAligModel.h
 *
 * @brief Defines the BaseSwAligModel class. BaseSwAligModel is a base
 * class for derivating single-word statistical alignment models.
 */

#ifndef _BaseSwAligModel_h
#define _BaseSwAligModel_h

#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string>
#include "AwkInputStream.h"
#include "SwDefs.h"
#include <ErrorDefs.h>
#include <StrProcUtils.h>
#include <WordAligMatrix.h>
#include <NbestTableNode.h>

class BaseSwAligModel
{
public:
  // Declarations related to dynamic class loading
  typedef BaseSwAligModel* create_t(const char*);
  typedef const char* type_id_t();

  // Constructor
  BaseSwAligModel();

  // Thread/Process safety related functions
  virtual bool modelReadsAreProcessSafe();

  void setVariationalBayes(bool variationalBayes);
  bool getVariationalBayes();

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
  virtual void trainSentPairRange(std::pair<unsigned int, unsigned int> sentPairRange, int verbosity = 0) = 0;
    // train model for range [uint,uint]
  virtual void trainAllSents(int verbosity = 0) = 0;
  virtual std::pair<double, double> loglikelihoodForPairRange(std::pair<unsigned int, unsigned int> sentPairRange,
    int verbosity = 0) = 0;
    // Returns log-likelihood. The first double contains the
    // loglikelihood for all sentences, and the second one, the same
    // loglikelihood normalized by the number of sentences
  virtual std::pair<double, double> loglikelihoodForAllSents(int verbosity = 0);
    // Returns log-likelihood. The first double contains the
    // loglikelihood for all sentences, and the second one, the same
    // loglikelihood normalized by the number of sentences
  virtual void clearInfoAboutSentRange() = 0;
    // clear info about the whole sentence range without clearing
    // information about current model parameters

  // Sentence length model functions
  virtual Prob sentLenProb(unsigned int slen, unsigned int tlen) = 0;
    // returns p(tlen|slen)
  virtual LgProb sentLenLgProb(unsigned int slen, unsigned int tlen) = 0;

  // Scoring functions for a given alignment
  LgProb calcLgProbForAligChar(const char* sSent, const char* tSent, const WordAligMatrix& aligMatrix, int verbose = 0);
  LgProb calcLgProbForAligVecStr(const std::vector<std::string>& sSent, const std::vector<std::string>& tSent,
    const WordAligMatrix& aligMatrix, int verbose = 0);
  virtual LgProb calcLgProbForAlig(const std::vector<WordIndex>& sSent, const std::vector<WordIndex>& tSent,
    const WordAligMatrix& aligMatrix, int verbose = 0) = 0;

  // Scoring functions without giving an alignment
  LgProb calcLgProbChar(const char* sSent, const char* tSent, int verbose = 0);
  LgProb calcLgProbVecStr(const std::vector<std::string>& sSent, const std::vector<std::string>& tSent,
    int verbose = 0);
  virtual LgProb calcLgProb(const std::vector<WordIndex>& sSent, const std::vector<WordIndex>& tSent,
    int verbose = 0) = 0;
  virtual LgProb calcLgProbPhr(const std::vector<WordIndex>& sPhr, const std::vector<WordIndex>& tPhr,
    int verbose = 0);
    // Scoring function for phrase pairs

  // Best-alignment functions
  bool obtainBestAlignments(const char* sourceTestFileName, const char* targetTestFilename, const char* outFileName);
    // Obtains the best alignments for the sentence pairs given in
    // the files 'sourceTestFileName' and 'targetTestFilename'. The
    // results are stored in the file 'outFileName'
  LgProb obtainBestAlignmentChar(const char* sourceSentence, const char* targetSentence, WordAligMatrix& bestWaMatrix);
    // Obtains the best alignment for the given sentence pair
  LgProb obtainBestAlignmentVecStr(std::vector<std::string> srcSentenceVector,
    std::vector<std::string> trgSentenceVector, WordAligMatrix& bestWaMatrix);
    // Obtains the best alignment for the given sentence pair (input
    // parameters are now string vectors)
  virtual LgProb obtainBestAlignment(std::vector<WordIndex> srcSentIndexVector,
    std::vector<WordIndex> trgSentIndexVector, WordAligMatrix& bestWaMatrix) = 0;
    // Obtains the best alignment for the given sentence pair
    // (input parameters are now index vectors) depending on the
    // value of the modelNumber data member.

  std::ostream& printAligInGizaFormat(const char* sourceSentence, const char* targetSentence, Prob p,
    std::vector<PositionIndex> alig, std::ostream& outS);
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
  virtual bool existSrcSymbol(std::string s)const = 0;
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

  // clear() function
  virtual void clear() = 0;

  // clearTempVars() function
  virtual void clearTempVars() {};

  virtual void clearSentLengthModel() = 0;

  // Utilities
  std::vector<WordIndex> addNullWordToWidxVec(const std::vector<WordIndex>& vw);
  std::vector<std::string> addNullWordToStrVec(const std::vector<std::string>& vw);

  virtual Prob pts(WordIndex s, WordIndex t) = 0;
  virtual LgProb logpts(WordIndex s, WordIndex t) = 0;

  // Destructor
  virtual ~BaseSwAligModel();

protected:
  double alpha = 0.01;
  bool variationalBayes = false;
};

#endif
