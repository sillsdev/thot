#pragma once

#include "nlp_common/WordIndex.h"

#include <set>
#include <vector>

class LexTable
{
public:
  virtual void setNumerator(WordIndex s, WordIndex t, float f) = 0;
  virtual float getNumerator(WordIndex s, WordIndex t, bool& found) const = 0;

  virtual void setDenominator(WordIndex s, float f) = 0;
  virtual float getDenominator(WordIndex s, bool& found) const = 0;

  virtual void set(WordIndex s, WordIndex t, float num, float den) = 0;

  virtual bool getTransForSource(WordIndex s, std::set<WordIndex>& transSet) const = 0;

  virtual bool load(const char* lexNumDenFile, int verbose = 0) = 0;

  virtual bool print(const char* lexNumDenFile, int verbose = 0) const = 0;

  virtual void reserveSpace(WordIndex s) = 0;

  virtual void clear() = 0;

  virtual ~LexTable(){};
};
