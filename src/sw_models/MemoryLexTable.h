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
  void setNumerator(WordIndex s, WordIndex t, float f) override;
  float getNumerator(WordIndex s, WordIndex t, bool& found) const override;

  void setDenominator(WordIndex s, float f) override;
  float getDenominator(WordIndex s, bool& found) const override;

  void set(WordIndex s, WordIndex t, float num, float den) override;

  bool getTransForSource(WordIndex t, std::set<WordIndex>& transSet) const override;

  bool load(const char* lexNumDenFile, int verbose = 0) override;

  bool print(const char* lexNumDenFile, int verbose = 0) const override;

  void reserveSpace(WordIndex s) override;

  void clear() override;

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
