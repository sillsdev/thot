#pragma once

#include "nlp_common/SingleWordVocab.h"
#include "nlp_common/WordIndex.h"

#include <unordered_map>
#include <vector>

typedef unsigned int WordClassIndex;

const std::string NULL_WORD_CLASS_STR = "NULL";
constexpr WordClassIndex NULL_WORD_CLASS = 0;

class WordClasses
{
public:
  WordClasses();

  WordClassIndex addSrcWordClass(const std::string& c);
  WordClassIndex addTrgWordClass(const std::string& c);

  WordClassIndex mapSrcWordToWordClass(WordIndex s, const std::string& c);
  WordClassIndex mapTrgWordToWordClass(WordIndex t, const std::string& c);
  void mapSrcWordToWordClass(WordIndex s, WordClassIndex c);
  void mapTrgWordToWordClass(WordIndex t, WordClassIndex c);

  WordClassIndex getSrcWordClass(WordIndex s) const;
  WordClassIndex getTrgWordClass(WordIndex t) const;

  WordClassIndex getSrcWordClassCount() const;
  WordClassIndex getTrgWordClassCount() const;

  bool load(const char* prefFileName, int verbose = 0);
  bool print(const char* prefFileName, int verbose = 0) const;

  void clear();

private:
  bool loadSrcWordClasses(const char* srcWordClassesFile, int verbose);
  bool loadTrgWordClasses(const char* trgWordClassesFile, int verbose);
  bool loadSrcWordClassNames(const char* srcWordClassNamesFile, int verbose);
  bool loadTrgWordClassNames(const char* trgWordClassNamesFile, int verbose);

  bool printSrcWordClasses(const char* srcWordClassesFile, int verbose) const;
  bool printTrgWordClasses(const char* trgWordClassesFile, int verbose) const;
  bool printSrcWordClassNames(const char* srcWordClassNamesFile, int verbose) const;
  bool printTrgWordClassNames(const char* trgWordClassNamesFile, int verbose) const;

  bool loadWordClasses(const char* wordClassesFile, std::vector<WordClassIndex>& wordClasses, int verbose);
  bool loadWordClassNames(const char* wordClassNamesFile,
                          std::unordered_map<std::string, WordClassIndex>& wordClassNames, int verbose);
  bool printWordClasses(const char* wordClassesFile, const std::vector<WordClassIndex>& wordClasses, int verbose) const;
  bool printWordClassNames(const char* wordClassNamesFile,
                           const std::unordered_map<std::string, WordClassIndex>& wordClassNames, int verbose) const;

  std::unordered_map<std::string, WordClassIndex> srcWordClassNames;
  std::unordered_map<std::string, WordClassIndex> trgWordClassNames;

  std::vector<WordClassIndex> srcWordClasses;
  std::vector<WordClassIndex> trgWordClasses;
};
