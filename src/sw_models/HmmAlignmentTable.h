#pragma once

#include "nlp_common/PositionIndex.h"

#include <vector>

class HmmAlignmentTable
{
public:
  void setNumerator(PositionIndex prev_i, PositionIndex slen, PositionIndex i, float f);
  float getNumerator(PositionIndex prev_i, PositionIndex slen, PositionIndex i, bool& found);

  void setDenominator(PositionIndex prev_i, PositionIndex slen, float f);
  float getDenominator(PositionIndex prev_i, PositionIndex slen, bool& found);

  void set(PositionIndex prev_i, PositionIndex slen, PositionIndex i, float num, float den);

  void reserveSpace(PositionIndex prev_i, PositionIndex slen);

  bool load(const char* lexNumDenFile, int verbose = 0);

  bool print(const char* lexNumDenFile);

  void clear();

protected:
  typedef std::vector<std::vector<std::pair<bool, float>>> NumeratorsElem;
  typedef std::vector<NumeratorsElem> Numerators;
  typedef std::vector<std::vector<std::pair<bool, float>>> Denominators;

  Numerators numerators;
  Denominators denominators;

  // load and print auxiliary functions
  bool loadBin(const char* lexNumDenFile, int verbose);
  bool loadPlainText(const char* lexNumDenFile, int verbose);
  bool printBin(const char* lexNumDenFile);
  bool printPlainText(const char* lexNumDenFile);
};
