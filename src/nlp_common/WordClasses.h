#pragma once

#include "nlp_common/SingleWordVocab.h"
#include "nlp_common/WordIndex.h"

#include <vector>

typedef unsigned int WordClassIndex;

constexpr WordClassIndex NULL_WORD_CLASS = 0;

class WordClasses
{
public:
  WordClasses();

  void addSrcWordClass(WordIndex s, WordClassIndex c);
  void addTrgWordClass(WordIndex t, WordClassIndex c);

  WordClassIndex getSrcWordClass(WordIndex s) const;
  WordClassIndex getTrgWordClass(WordIndex t) const;

  WordClassIndex getSrcWordClassCount() const;
  WordClassIndex getTrgWordClassCount() const;

  bool loadSrcWordClasses(const char* srcWordClassesFile, int verbose = 0);
  bool loadTrgWordClasses(const char* trgWordClassesFile, int verbose = 0);

  bool printSrcWordClasses(const char* srcWordClassesFile, int verbose = 0) const;
  bool printTrgWordClasses(const char* trgWordClassesFile, int verbose = 0) const;

  void clear();

private:
  bool loadBin(const char* wordClassesFile, std::vector<WordClassIndex>& wordClasses, WordClassIndex& wordClassCount,
               int verbose);
  bool printBin(const char* wordClassesFile, const std::vector<WordClassIndex>& wordClasses, int verbose) const;

  WordClassIndex srcWordClassCount;
  WordClassIndex trgWordClassCount;

  std::vector<WordClassIndex> srcWordClasses;
  std::vector<WordClassIndex> trgWordClasses;
};
