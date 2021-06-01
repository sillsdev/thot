#ifndef _HeadDistortionTable_h
#define _HeadDistortionTable_h

#if HAVE_CONFIG_H
#include "thot_config.h"
#endif

#include <unordered_map>
#include <vector>

#include "OrderedVector.h"
#include "PositionIndex.h"
#include "WordClassIndex.h"

struct HeadDistortionTableKey
{
public:
  WordClassIndex srcWordClass;
  WordClassIndex trgWordClass;

  bool operator<(const HeadDistortionTableKey& right) const
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

  bool operator==(const HeadDistortionTableKey& right) const
  {
    return srcWordClass == right.srcWordClass && trgWordClass == right.trgWordClass;
  }
};

struct HeadDistortionTableKeyHash
{
public:
  std::size_t operator()(const HeadDistortionTableKey& key) const
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

  void setNumeratorDenominator(WordClassIndex srcWordClass, WordClassIndex trgWordClass, int dj, float num, float den);

  void reserveSpace(WordClassIndex srcWordClass, WordClassIndex trgWordClass);

  bool load(const char* tableFile, int verbose = 0);
  bool print(const char* tableFile) const;

  void clear();

private:
  typedef OrderedVector<int, float> NumeratorsElem;
  typedef std::unordered_map<HeadDistortionTableKey, NumeratorsElem, HeadDistortionTableKeyHash> Numerators;
  typedef std::unordered_map<HeadDistortionTableKey, float, HeadDistortionTableKeyHash> Denominators;

  Numerators numerators;
  Denominators denominators;

  bool loadBin(const char* tableFile, int verbose);
  bool loadPlainText(const char* tableFile, int verbose);
  bool printBin(const char* tableFile) const;
  bool printPlainText(const char* tableFile) const;
};

#endif
