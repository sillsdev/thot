#pragma once

#include "nlp_common/Bitset.h"
#include "nlp_common/WordAlignmentMatrix.h"
#include "phrase_models/BpSet.h"
#include "phrase_models/CellID.h"
#include "phrase_models/PhraseDefs.h"
#include "phrase_models/PhraseExtractParameters.h"
#include "phrase_models/PhraseExtractionCell.h"
#include "phrase_models/PhrasePair.h"
#include "phrase_models/SrfBisegm.h"
#include "phrase_models/SrfNodeInfoMap.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <vector>

/*
 * Defines the PhraseExtractionTable class for extracting all
 * consistent phrases from valid segmentations given a phrase pair and
 * its word alignment matrix.
 */
class PhraseExtractionTable
{
public:
  PhraseExtractionTable(void);
  void extractConsistentPhrases(PhraseExtractParameters phePars, const std::vector<std::string>& _ns,
                                const std::vector<std::string>& _t, const WordAlignmentMatrix& _alig,
                                std::vector<PhrasePair>& outvph);
  double segmBasedExtraction(PhraseExtractParameters phePars, const std::vector<std::string>& _ns,
                             const std::vector<std::string>& _t, const WordAlignmentMatrix& _alig,
                             std::vector<PhrasePair>& outvph, int verbose = 0);
  void clear(void);
  ~PhraseExtractionTable();

private:
  std::vector<std::vector<PhraseExtractionCell>> pecMatrix;

  std::vector<std::string> ns;
  std::vector<std::string> t;
  WordAlignmentMatrix alig;
  unsigned int nslen;
  unsigned int tlen;
  Bitset<MAX_SENTENCE_LENGTH> zFertBitset;
  Bitset<MAX_SENTENCE_LENGTH> spurBitset;

  int maxTrgPhraseLength;
  int maxSrcPhraseLength;
  bool countSpurious;
  bool monotone;

  void extractConsistentPhrasesOld(PhraseExtractParameters phePars, const std::vector<std::string>& _ns,
                                   const std::vector<std::string>& _t, const WordAlignmentMatrix& _alig,
                                   std::vector<PhrasePair>& outvph);
  void extractConsistentPhrasesOch(PhraseExtractParameters phePars, const std::vector<std::string>& _ns,
                                   const std::vector<std::string>& _t, const WordAlignmentMatrix& _alig,
                                   std::vector<PhrasePair>& outvph);
  double gen01RandNum(void);
  void obtainConsistentPhrases(void);
  void obtainBpSet(BpSet& bpSet);
  double srfPhraseExtract(const BpSet& bpSet, BpSet& C);
  double srfPhraseExtractRec(const BpSet& bpSet, const Bitset<MAX_SENTENCE_LENGTH>& SP,
                             const Bitset<MAX_SENTENCE_LENGTH>& TP, BpSet& C);
  double srfPhraseExtractDp(const BpSet& bpSet, BpSet& C, int verbose = false);
  double approxSrfPhraseExtract(const BpSet& bpSet, BpSet& C, int verbose = false);
  void srfPhrExtrEstBisegLenRand(const BpSet& bpSet, SrfNodeInfoMap& sniMap);
  void srfPhrExtrEstBisegLenRandRec(const BpSet& bpSet, const Bitset<MAX_SENTENCE_LENGTH>& SP,
                                    const Bitset<MAX_SENTENCE_LENGTH>& TP, const SrfNodeKey& snk,
                                    SrfNodeInfoMap& sniMap);
  void fillSrfNodeInfoMap(const BpSet& bpSet, SrfNodeInfoMap& sniMap, bool calcCSet = true);
  void fillSrfNodeInfoMapRec(const BpSet& bpSet, const Bitset<MAX_SENTENCE_LENGTH>& SP,
                             const Bitset<MAX_SENTENCE_LENGTH>& TP, SrfNodeInfoMap& sniMap, bool calcCSet = true);
  double bisegmRandWalk(const BpSet& bpSet, const SrfNodeInfoMap& sniMap, BpSet& C);
  bool bisegmRandWalkRec(const BpSet& bpSet, const Bitset<MAX_SENTENCE_LENGTH>& SP,
                         const Bitset<MAX_SENTENCE_LENGTH>& TP, const SrfBisegm& sb, const SrfNodeInfoMap& sniMap,
                         SrfBisegm& result);
  void obtainPhrPairVecFromBpSet(const BpSet& bpSet, std::vector<PhrasePair>& outvph, double logNumSegms = 0);
  void createVectorWithConsPhrases(std::vector<PhrasePair>& consistentPhrases);
  void getSegmentationsForEachCell(void);
  void getSegmentationsForEachCellFast(void);
  bool validCoverageForCell(Bitset<MAX_SENTENCE_LENGTH>& c, unsigned int x, unsigned int y);
  bool validSegmentationForCell(const std::vector<CellID>& cidVec, Bitset<MAX_SENTENCE_LENGTH>& zFertBits,
                                unsigned int x, unsigned int y, unsigned int first = 0);
  Bitset<MAX_SENTENCE_LENGTH> zeroFertBitset(WordAlignmentMatrix& waMatrix);
  Bitset<MAX_SENTENCE_LENGTH> spuriousWordsBitset(WordAlignmentMatrix& waMatrix);
  bool existCellAlig(const std::vector<CellAlignment>& cellAligs, CellAlignment calig);
  unsigned int leftMostPosInCell(unsigned int x, unsigned int y);
  unsigned int rightMostPosInCell(unsigned int x, unsigned int y);
  bool sourcePosInCell(unsigned int j, unsigned int x, unsigned int y);
  unsigned int trgPhraseLengthInCell(unsigned int x, unsigned int y);
  unsigned int trgPhraseLengthInCellNonSpurious(unsigned int x, unsigned int y, Bitset<MAX_SENTENCE_LENGTH>& spurBits);
};
