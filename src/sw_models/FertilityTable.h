#pragma once

#include "nlp_common/AwkInputStream.h"
#include "nlp_common/ErrorDefs.h"
#include "nlp_common/PositionIndex.h"
#include "nlp_common/WordIndex.h"

#include <fstream>
#include <vector>

class FertilityTable
{
public:
  void setNumerator(WordIndex s, PositionIndex phi, float f);
  float getNumerator(WordIndex s, PositionIndex phi, bool& found) const;

  void setDenominator(WordIndex s, float f);
  float getDenominator(WordIndex s, bool& found) const;

  void set(WordIndex s, PositionIndex phi, float num, float den);

  bool load(const char* fertilityNumDenFile, int verbose = 0);

  bool print(const char* fertilityNumDenFile) const;

  void reserveSpace(WordIndex s);

  void clear();

private:
  typedef std::vector<float> NumeratorsElem;
  typedef std::vector<NumeratorsElem> Numerators;
  typedef std::vector<float> Denominators;

  Numerators numerators;
  Denominators denominators;

  bool loadBin(const char* fertilityNumDenFile, int verbose);
  bool loadPlainText(const char* fertilityNumDenFile, int verbose);
  bool printBin(const char* fertilityNumDenFile) const;
  bool printPlainText(const char* fertilityNumDenFile) const;
};
