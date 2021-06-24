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
 * @file SingleWordVocab.h
 *
 * @brief Manages a single-word vocabulary.
 */

#pragma once

//--------------- Include files ---------------------------------------

#include "nlp_common/WordIndex.h"

#include <string>
#include <vector>

#ifdef THOT_DISABLE_SPACE_EFFICIENT_VOCAB_STRUCTURES
#include <map>
#else
#include <unordered_map>
#endif

//--------------- Constants -------------------------------------------

//--------------- typedefs --------------------------------------------

//--------------- function declarations -------------------------------

//--------------- Classes ---------------------------------------------

class StringHashF
{
public:
  enum
  {
    bucket_size = 1
  };

  std::size_t operator()(const std::string& str) const
  {
    unsigned int hash = 1315423911;

    for (std::size_t i = 0; i < str.length(); i++)
    {
      hash ^= ((hash << 5) + str[i] + (hash >> 2));
    }

    return (hash & 0x7FFFFFFF);
  }

  bool operator()(const std::string& left, const std::string& right) const
  {
    return left.compare(right) < 0;
  }
};

//--------------- SingleWordVocab class

class SingleWordVocab
{
public:
#ifdef THOT_DISABLE_SPACE_EFFICIENT_VOCAB_STRUCTURES
  typedef std::map<std::string, WordIndex> StrToIdxVocab;
  typedef std::map<WordIndex, std::string> IdxToStrVocab;
#else
  typedef std::unordered_map<std::string, WordIndex, StringHashF> StrToIdxVocab;
  typedef std::unordered_map<WordIndex, std::string> IdxToStrVocab;
#endif

  // Constructor
  SingleWordVocab(void);

  // Functions related to the source vocabulary
  StrToIdxVocab getSrcVocab(void) const;
  size_t getSrcVocabSize(void) const; // Returns the source vocabulary size
  WordIndex stringToSrcWordIndex(std::string s) const;
  std::string wordIndexToSrcString(WordIndex w) const;
  bool existSrcSymbol(std::string s) const;
  std::vector<WordIndex> strVectorToSrcIndexVector(std::vector<std::string> s);
  // converts a string vector into a source word index vector, this
  // function automatically handles the source vocabulary,
  // increasing and modifying it if necessary
  WordIndex addSrcSymbol(std::string s);
  bool loadSrcVocab(const char* srcInputVocabFileName, int verbose = 0);
  bool printSrcVocab(const char* outputFileName);
  bool loadGIZASrcVocab(const char* srcInputVocabFileName, int verbose = 0);
  bool printGIZASrcVocab(const char* outputFileName);
  // Reads source vocabulary from a file in GIZA format

  // Functions related to the target vocabulary
  StrToIdxVocab getTrgVocab(void) const;
  size_t getTrgVocabSize(void) const; // Returns the target vocabulary size
  WordIndex stringToTrgWordIndex(std::string t) const;
  std::string wordIndexToTrgString(WordIndex w) const;
  bool existTrgSymbol(std::string t) const;
  std::vector<WordIndex> strVectorToTrgIndexVector(std::vector<std::string> t);
  // converts a string vector into a target word index vector, this
  // function automatically handles the target vocabulary,
  // increasing and modifying it if necessary
  WordIndex addTrgSymbol(std::string t);
  bool loadTrgVocab(const char* trgInputVocabFileName, int verbose = 0);
  bool printTrgVocab(const char* outputFileName);
  bool loadGIZATrgVocab(const char* trgInputVocabFileName, int verbose = 0);
  bool printGIZATrgVocab(const char* trgInputVocabFileName);
  // Reads target vocabulary from a file in GIZA format

  // clear() function
  void clear(void);

  // Destructor
  ~SingleWordVocab();

protected:
  StrToIdxVocab stringToSrcWordIndexMap;
  IdxToStrVocab srcWordIndexMapToString;
  StrToIdxVocab stringToTrgWordIndexMap;
  IdxToStrVocab trgWordIndexMapToString;

  void clearSrcVocab(void);
  void clearTrgVocab(void);
  void add_null_word_to_srcvoc(void);
  void add_unk_word_to_srcvoc(void);
  void add_unused_word_to_srcvoc(void);
  void add_null_word_to_trgvoc(void);
  void add_unk_word_to_trgvoc(void);
  void add_unused_word_to_trgvoc(void);
};
