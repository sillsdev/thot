#pragma once

#include "sw_models/LexTable.h"

#include <set>
#include <vector>

#ifdef THOT_DISABLE_SPACE_EFFICIENT_LEXDATA_STRUCTURES
#include <unordered_map>
#else
#include "nlp_common/OrderedVector.h"
#endif

class MemoryLexTable : public LexTable
{
public:
  void setNumerator(WordIndex s, WordIndex t, float f);
  float getNumerator(WordIndex s, WordIndex t, bool& found) const;

  void setDenominator(WordIndex s, float f);
  float getDenominator(WordIndex s, bool& found) const;

  void set(WordIndex s, WordIndex t, float num, float den);

  bool getTransForSource(WordIndex t, std::set<WordIndex>& transSet) const;

  bool load(const char* lexNumDenFile, int verbose = 0);

  bool print(const char* lexNumDenFile, int verbose = 0) const;

  void reserveSpace(WordIndex s);

  void clear();

  virtual ~MemoryLexTable()
  {
  }

protected:
#ifdef THOT_DISABLE_SPACE_EFFICIENT_LEXDATA_STRUCTURES
  typedef std::unordered_map<WordIndex, float> NumeratorsElem;
#else
  typedef OrderedVector<WordIndex, float> NumeratorsElem;
#endif
  typedef std::vector<NumeratorsElem> Numerators;
  typedef std::vector<std::pair<bool, float>> Denominators;

  Numerators numerators;
  Denominators denominators;

  // load and print auxiliary functions
  bool loadBin(const char* lexNumDenFile, int verbose);
  bool loadPlainText(const char* lexNumDenFile, int verbose);
  bool printBin(const char* lexNumDenFile, int verbose) const;
  bool printPlainText(const char* lexNumDenFile, int verbose) const;
};
