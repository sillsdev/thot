#pragma once

#include "nlp_common/PositionIndex.h"

#include <vector>

struct CeptNode
{
public:
  PositionIndex prev;
  PositionIndex next;
};

class AlignmentInfo
{
public:
  AlignmentInfo(PositionIndex slen, PositionIndex tlen)
      : slen(slen), tlen(tlen), alignment(tlen, 0), positionSum(slen + 1, 0), fertility(slen + 1, 0),
        heads(slen + 1, 0), ceptNodes(tlen + 1), prob(-1.0)
  {
    fertility[0] = tlen;
    for (PositionIndex j = 1; j <= tlen; ++j)
    {
      if (j > 1)
        ceptNodes[j].prev = j - 1;
      if (j < tlen)
        ceptNodes[j].next = j + 1;
    }
    heads[0] = 1;
  }

  PositionIndex getSourceLength() const
  {
    return slen;
  }

  PositionIndex getTargetLength() const
  {
    return tlen;
  }

  const std::vector<PositionIndex>& getAlignment() const
  {
    return alignment;
  }

  void setAlignment(const std::vector<PositionIndex>& alignment)
  {
    for (PositionIndex j = 1; j <= tlen; ++j)
      set(j, alignment[j - 1]);
  }

  PositionIndex get(PositionIndex j) const
  {
    return alignment[j - 1];
  }

  void set(PositionIndex j, PositionIndex i)
  {
    PositionIndex iOld = alignment[j - 1];
    positionSum[iOld] -= j;

    PositionIndex prev = ceptNodes[j].prev;
    PositionIndex next = ceptNodes[j].next;
    if (next > 0)
      ceptNodes[next].prev = prev;
    if (prev > 0)
      ceptNodes[prev].next = next;
    else
      heads[iOld] = next;

    next = heads[i];
    prev = 0;
    while (next > 0 && next < j)
    {
      prev = next;
      next = ceptNodes[next].next;
    }

    ceptNodes[j].prev = prev;
    ceptNodes[j].next = next;
    if (prev > 0)
      ceptNodes[prev].next = j;
    else
      heads[i] = j;
    if (next > 0)
      ceptNodes[next].prev = j;

    fertility[iOld]--;
    positionSum[i] += j;
    fertility[i]++;
    alignment[j - 1] = i;
  }

  PositionIndex getFertility(PositionIndex i) const
  {
    return fertility[i];
  }

  bool isHead(PositionIndex j) const
  {
    PositionIndex i = get(j);
    return heads[i] == j;
  }

  PositionIndex getCenter(PositionIndex i) const
  {
    if (i == 0)
      return 0;

    return (positionSum[i] + fertility[i] - 1) / fertility[i];
  }

  PositionIndex getPrevCept(PositionIndex i) const
  {
    if (i == 0)
      return 0;
    PositionIndex k = i - 1;
    while (k > 0 && fertility[k] == 0)
      k--;
    return k;
  }

  PositionIndex getNextCept(PositionIndex i) const
  {
    PositionIndex k = i + 1;
    while (k < slen + 1 && fertility[k] == 0)
      k++;
    return k;
  }

  PositionIndex getPrevInCept(PositionIndex j) const
  {
    return ceptNodes[j].prev;
  }

  double getProb() const
  {
    return prob;
  }

  void setProb(double p)
  {
    prob = p;
  }

private:
  PositionIndex slen;
  PositionIndex tlen;
  std::vector<PositionIndex> alignment;
  std::vector<PositionIndex> positionSum;
  std::vector<PositionIndex> fertility;
  std::vector<PositionIndex> heads;
  std::vector<CeptNode> ceptNodes;
  double prob;
};

