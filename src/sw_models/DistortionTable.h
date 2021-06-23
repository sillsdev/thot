#pragma once

#include "nlp_common/AwkInputStream.h"
#include "nlp_common/ErrorDefs.h"
#include "nlp_common/PositionIndex.h"

#include <unordered_map>
#include <vector>

struct DistortionKey
{
public:
  PositionIndex i;
  PositionIndex slen;
  PositionIndex tlen;

  bool operator<(const DistortionKey& right) const
  {
    if (right.i < i)
      return 0;
    if (i < right.i)
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

  bool operator==(const DistortionKey& right) const
  {
    return (i == right.i && slen == right.slen && tlen == right.tlen);
  }
};

struct DistortionKeyHash
{
public:
  std::size_t operator()(const DistortionKey& a1) const
  {
    return (size_t)(16384 * a1.i) + ((size_t)256 * a1.slen) + a1.tlen;
  }
};

class DistortionTable
{
public:
  void setNumerator(PositionIndex i, PositionIndex slen, PositionIndex tlen, PositionIndex j, float f);
  float getNumerator(PositionIndex i, PositionIndex slen, PositionIndex tlen, PositionIndex j, bool& found) const;

  void setDenominator(PositionIndex i, PositionIndex slen, PositionIndex tlen, float f);
  float getDenominator(PositionIndex i, PositionIndex slen, PositionIndex tlen, bool& found) const;

  void set(PositionIndex i, PositionIndex slen, PositionIndex tlen, PositionIndex j, float num, float den);

  void reserveSpace(PositionIndex i, PositionIndex slen, PositionIndex tlen);

  bool load(const char* distortionNumDenFile, int verbose = 0);

  bool print(const char* distortionNumDenFile) const;

  void clear();

private:
  typedef std::vector<float> NumeratorsElem;
  typedef std::unordered_map<DistortionKey, NumeratorsElem, DistortionKeyHash> Numerators;
  typedef std::unordered_map<DistortionKey, float, DistortionKeyHash> Denominators;

  Numerators numerators;
  Denominators denominators;

  bool loadBin(const char* distortionNumDenFile, int verbose);
  bool loadPlainText(const char* distortionNumDenFile, int verbose);
  bool printBin(const char* distortionNumDenFile) const;
  bool printPlainText(const char* distortionNumDenFile) const;
};
