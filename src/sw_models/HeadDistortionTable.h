#pragma once

#include "nlp_common/OrderedVector.h"
#include "nlp_common/PositionIndex.h"
#include "nlp_common/WordClasses.h"

#include <unordered_map>
#include <vector>

struct HeadDistortionKey
{
public:
  WordClassIndex srcWordClass;
  WordClassIndex trgWordClass;

  bool operator<(const HeadDistortionKey& right) const
  {
    if (right.srcWordClass < srcWordClass)
      return false;
    if (srcWordClass < right.srcWordClass)
      return true;
    if (right.trgWordClass < trgWordClass)
      return false;
    if (trgWordClass < right.trgWordClass)
      return true;
    return false;
  }

  bool operator==(const HeadDistortionKey& right) const
  {
    return srcWordClass == right.srcWordClass && trgWordClass == right.trgWordClass;
  }
};

struct HeadDistortionKeyHash
{
public:
  std::size_t operator()(const HeadDistortionKey& key) const
  {
    return (size_t)((size_t)256 * key.srcWordClass) + key.trgWordClass;
  }
};

class HeadDistortionTable
{
public:
  void setNumerator(WordClassIndex srcWordClass, WordClassIndex trgWordClass, int dj, float f);
  float getNumerator(WordClassIndex srcWordClass, WordClassIndex trgWordClass, int dj, bool& found) const;

  void setDenominator(WordClassIndex srcWordClass, WordClassIndex trgWordClass, float f);
  float getDenominator(WordClassIndex srcWordClass, WordClassIndex trgWordClass, bool& found) const;

  void set(WordClassIndex srcWordClass, WordClassIndex trgWordClass, int dj, float num, float den);

  void reserveSpace(WordClassIndex srcWordClass, WordClassIndex trgWordClass);

  bool load(const char* tableFile, int verbose = 0);
  bool print(const char* tableFile) const;

  void clear();

private:
  typedef OrderedVector<int, float> NumeratorsElem;
  typedef std::unordered_map<HeadDistortionKey, NumeratorsElem, HeadDistortionKeyHash> Numerators;
  typedef std::unordered_map<HeadDistortionKey, float, HeadDistortionKeyHash> Denominators;

  Numerators numerators;
  Denominators denominators;

  bool loadBin(const char* tableFile, int verbose);
  bool loadPlainText(const char* tableFile, int verbose);
  bool printBin(const char* tableFile) const;
  bool printPlainText(const char* tableFile) const;
};
