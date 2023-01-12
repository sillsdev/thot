#pragma once

#include "nlp_common/PositionIndex.h"

#include <unordered_map>
#include <vector>

struct AlignmentKey
{
public:
  PositionIndex j;
  PositionIndex slen;
  PositionIndex tlen;

  bool operator<(const AlignmentKey& right) const
  {
    if (right.j < j)
      return 0;
    if (j < right.j)
      return 1;
    if (right.slen < slen)
      return 0;
    if (slen < right.slen)
      return 1;
    if (right.tlen < tlen)
      return 0;
    if (tlen < right.tlen)
      return 1;
    return 0;
  }

  bool operator==(const AlignmentKey& right) const
  {
    return (j == right.j && slen == right.slen && tlen == right.tlen);
  }
};

struct AlignmentKeyHash
{
public:
  std::size_t operator()(const AlignmentKey& a1) const
  {
    return (std::size_t)(16384 * a1.j) + ((std::size_t)256 * a1.slen) + a1.tlen;
  }
};

class AlignmentTable
{
public:
  void setNumerator(PositionIndex j, PositionIndex slen, PositionIndex tlen, PositionIndex i, float f);
  float getNumerator(PositionIndex j, PositionIndex slen, PositionIndex tlen, PositionIndex i, bool& found) const;

  void setDenominator(PositionIndex j, PositionIndex slen, PositionIndex tlen, float f);
  float getDenominator(PositionIndex j, PositionIndex slen, PositionIndex tlen, bool& found) const;

  void set(PositionIndex j, PositionIndex slen, PositionIndex tlen, PositionIndex i, float num, float den);

  void reserveSpace(PositionIndex j, PositionIndex slen, PositionIndex tlen);

  bool load(const char* aligNumDenFile, int verbose = 0);

  bool print(const char* aligNumDenFile) const;

  void clear();

private:
  typedef std::vector<float> NumeratorsElem;
  typedef std::unordered_map<AlignmentKey, NumeratorsElem, AlignmentKeyHash> Numerators;
  typedef std::unordered_map<AlignmentKey, float, AlignmentKeyHash> Denominators;

  Numerators numerators;
  Denominators denominators;

  bool loadBin(const char* aligNumDenFile, int verbose);
  bool loadPlainText(const char* aligNumDenFile, int verbose);
  bool printBin(const char* aligNumDenFile) const;
  bool printPlainText(const char* aligNumDenFile) const;
};
