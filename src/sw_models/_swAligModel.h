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
 * @file _swAligModel.h
 *
 * @brief Defines the _swAligModel class. _swAligModel is a predecessor
 * class for derivating single-word statistical alignment models.
 *
 */

#pragma once

#include "nlp_common/SingleWordVocab.h"
#include "sw_models/BaseSwAligModel.h"
#include "sw_models/LightSentenceHandler.h"

#include <set>

class _swAligModel : public BaseSwAligModel
{
public:
  // Constructor
  _swAligModel(void);

  // Functions to read and add sentence pairs
  bool readSentencePairs(const char* srcFileName, const char* trgFileName, const char* sentCountsFile,
                         std::pair<unsigned int, unsigned int>& sentRange, int verbose = 0);
  void addSentPair(std::vector<std::string> srcSentStr, std::vector<std::string> trgSentStr, Count c,
                   std::pair<unsigned int, unsigned int>& sentRange);
  unsigned int numSentPairs(void);
  // NOTE: the whole valid range in a given moment is
  // [ 0 , numSentPairs() )
  int nthSentPair(unsigned int n, std::vector<std::string>& srcSentStr, std::vector<std::string>& trgSentStr, Count& c);

  // Functions to print sentence pairs
  bool printSentPairs(const char* srcSentFile, const char* trgSentFile, const char* sentCountsFile);

  // Functions for loading vocabularies
  bool loadGIZASrcVocab(const char* srcInputVocabFileName, int verbose = 0);
  // Reads source vocabulary from a file in GIZA format
  bool loadGIZATrgVocab(const char* trgInputVocabFileName, int verbose = 0);
  // Reads target vocabulary from a file in GIZA format

  // Functions for printing vocabularies
  bool printGIZASrcVocab(const char* srcOutputVocabFileName);
  // Reads source vocabulary from a file in GIZA format
  bool printGIZATrgVocab(const char* trgOutputVocabFileName);
  // Reads target vocabulary from a file in GIZA format

  // Source and target vocabulary functions
  size_t getSrcVocabSize() const; // Returns the source vocabulary size
  WordIndex stringToSrcWordIndex(std::string s) const;
  std::string wordIndexToSrcString(WordIndex w) const;
  bool existSrcSymbol(std::string s) const;
  std::vector<WordIndex> strVectorToSrcIndexVector(std::vector<std::string> s);
  WordIndex addSrcSymbol(std::string s);

  size_t getTrgVocabSize() const; // Returns the target vocabulary size
  WordIndex stringToTrgWordIndex(std::string t) const;
  std::string wordIndexToTrgString(WordIndex w) const;
  bool existTrgSymbol(std::string t) const;
  std::vector<WordIndex> strVectorToTrgIndexVector(std::vector<std::string> t);
  WordIndex addTrgSymbol(std::string t);

  // clear() function
  void clear();

  // Destructor
  virtual ~_swAligModel();

protected:
  SingleWordVocab swVocab;
  LightSentenceHandler sentenceHandler;

  bool printVariationalBayes(const std::string& filename);
  bool loadVariationalBayes(const std::string& filename);
};

