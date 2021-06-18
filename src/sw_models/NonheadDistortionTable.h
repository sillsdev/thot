#ifndef _NonheadDistortionTable_h
#define _NonheadDistortionTable_h

#include "nlp_common/OrderedVector.h"
#include "nlp_common/PositionIndex.h"
#include "nlp_common/WordClassIndex.h"

#include <unordered_map>
#include <vector>

class NonheadDistortionTable
{
public:
  void setNumerator(WordClassIndex trgWordClass, int dj, float f);
  float getNumerator(WordClassIndex trgWordClass, int dj, bool& found) const;

  void setDenominator(WordClassIndex trgWordClass, float f);
  float getDenominator(WordClassIndex trgWordClass, bool& found) const;

  void setNumeratorDenominator(WordClassIndex trgWordClass, int dj, float num, float den);

  void reserveSpace(WordClassIndex trgWordClass);

  bool load(const char* tableFile, int verbose = 0);
  bool print(const char* tableFile) const;

  void clear();

private:
  typedef OrderedVector<int, float> NumeratorsElem;
  typedef std::vector<NumeratorsElem> Numerators;
  typedef std::vector<std::pair<bool, float>> Denominators;

  Numerators numerators;
  Denominators denominators;

  bool loadBin(const char* tableFile, int verbose);
  bool loadPlainText(const char* tableFile, int verbose);
  bool printBin(const char* tableFile) const;
  bool printPlainText(const char* tableFile) const;
};

#endif