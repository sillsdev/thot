#include "stack_dec/TranslationMetadata.h"

#include "nlp_common/StrProcUtils.h"
#include "stack_dec/PhrScoreInfo.h"

#include <gtest/gtest.h>

class TranslationMetadataTest : public testing::Test
{
protected:
  std::string getXmlString()
  {
    return "<phr_pair_annot><src_segm>First</src_segm><trg_segm>premier</trg_segm></phr_pair_annot> and only "
           "<phr_pair_annot><src_segm>T-shirt with</src_segm><trg_segm>t-shirt avec</trg_segm></phr_pair_annot> "
           "<phr_pair_annot><src_segm>logo</src_segm><trg_segm>Logo</trg_segm></phr_pair_annot> "
           "<phr_pair_annot><src_segm>22.9cm</src_segm><trg_segm>22.9cm</trg_segm></phr_pair_annot> "
           "<phr_pair_annot><src_segm>2x5</src_segm><trg_segm>2x5</trg_segm></phr_pair_annot>";
  }

  TranslationMetadata<PhrScoreInfo> metadata;
};

TEST_F(TranslationMetadataTest, safeRecallingObtainTransConstraints)
{
  metadata.clear();

  // Call
  metadata.obtainTransConstraints(getXmlString(), 0);
  std::vector<std::string> srcSentVec1 = metadata.getSrcSentVec();

  // Call again without cleaning
  metadata.obtainTransConstraints(getXmlString(), 0);
  std::vector<std::string> srcSentVec2 = metadata.getSrcSentVec();

  EXPECT_EQ(srcSentVec1, srcSentVec2);
}

TEST_F(TranslationMetadataTest, getSrcSentVec)
{
  // Prepare expected answer
  const std::string expectedSrcSent = "First and only T-shirt with logo 22.9cm 2x5";

  metadata.clear();
  metadata.obtainTransConstraints(getXmlString(), 0);

  std::vector<std::string> srcSentVec = metadata.getSrcSentVec();

  EXPECT_EQ(StrProcUtils::stringToStringVector(expectedSrcSent).size(), srcSentVec.size());
  EXPECT_EQ(expectedSrcSent, StrProcUtils::stringVectorToString(srcSentVec));
}

TEST_F(TranslationMetadataTest, getTransForSrcPhr)
{
  metadata.obtainTransConstraints(getXmlString(), 0);
  std::vector<std::string> translation = metadata.getTransForSrcPhr(std::make_pair(1, 1));

  std::vector<std::string> expectedTranslation;
  expectedTranslation.push_back("premier");

  EXPECT_EQ(expectedTranslation, translation);
}

TEST_F(TranslationMetadataTest, getConstrainedSrcPhrases)
{
  // Prepare expected data structure
  std::set<std::pair<PositionIndex, PositionIndex>> expectedConstraints;
  expectedConstraints.insert(std::make_pair(1, 1));
  expectedConstraints.insert(std::make_pair(4, 5));
  expectedConstraints.insert(std::make_pair(6, 6));
  expectedConstraints.insert(std::make_pair(7, 7));
  expectedConstraints.insert(std::make_pair(8, 8));

  // Obtain constraints for a given xml string
  metadata.obtainTransConstraints(getXmlString(), 0);
  std::set<std::pair<PositionIndex, PositionIndex>> constraints = metadata.getConstrainedSrcPhrases();

  EXPECT_EQ(expectedConstraints, constraints);
}

TEST_F(TranslationMetadataTest, srcPhrAffectedByConstraint)
{
  bool isConstrained;
  // Extract constraints
  metadata.obtainTransConstraints(getXmlString(), 0);

  isConstrained = metadata.srcPhrAffectedByConstraint(std::make_pair(1, 1));
  EXPECT_TRUE(isConstrained);

  isConstrained = metadata.srcPhrAffectedByConstraint(std::make_pair(1, 2));
  EXPECT_TRUE(isConstrained);

  isConstrained = metadata.srcPhrAffectedByConstraint(std::make_pair(2, 2));
  EXPECT_FALSE(isConstrained);

  isConstrained = metadata.srcPhrAffectedByConstraint(std::make_pair(5, 7));
  EXPECT_TRUE(isConstrained);

  isConstrained = metadata.srcPhrAffectedByConstraint(std::make_pair(6, 8));
  EXPECT_TRUE(isConstrained);
}

TEST_F(TranslationMetadataTest, translationSatisfyingSrcPhrConstraints)
{
  // Extract constraints
  metadata.obtainTransConstraints(getXmlString(), 0);

  // Prepare parameters
  SourceSegmentation sourceSegmentation;
  sourceSegmentation.push_back(std::make_pair(1, 1));
  sourceSegmentation.push_back(std::make_pair(2, 3));
  sourceSegmentation.push_back(std::make_pair(4, 5));
  sourceSegmentation.push_back(std::make_pair(6, 6));
  sourceSegmentation.push_back(std::make_pair(7, 7));
  sourceSegmentation.push_back(std::make_pair(8, 8));

  std::vector<PositionIndex> targetSegmentCuts;
  targetSegmentCuts.push_back(1);
  targetSegmentCuts.push_back(3);
  targetSegmentCuts.push_back(5);
  targetSegmentCuts.push_back(6);
  targetSegmentCuts.push_back(7);
  targetSegmentCuts.push_back(8);

  std::vector<std::string> targetWordVec;
  targetWordVec.push_back("premier");
  targetWordVec.push_back("et");
  targetWordVec.push_back("Only");
  targetWordVec.push_back("t-shirt");
  targetWordVec.push_back("avec");
  targetWordVec.push_back("Logo");
  targetWordVec.push_back("22.9cm");
  targetWordVec.push_back("2x5");

  // Valid translation - respects all constraints
  bool isSatisfied = metadata.translationSatisfiesConstraints(sourceSegmentation, targetSegmentCuts, targetWordVec);
  EXPECT_TRUE(isSatisfied);
}

TEST_F(TranslationMetadataTest, translationViolatingSrcSegmentationConstraints)
{
  // Extract constraints
  metadata.obtainTransConstraints(getXmlString(), 0);

  // Prepare parameters
  SourceSegmentation sourceSegmentation;
  sourceSegmentation.push_back(std::make_pair(1, 2));
  sourceSegmentation.push_back(std::make_pair(3, 3));
  sourceSegmentation.push_back(std::make_pair(4, 5));
  sourceSegmentation.push_back(std::make_pair(6, 6));
  sourceSegmentation.push_back(std::make_pair(7, 7));
  sourceSegmentation.push_back(std::make_pair(8, 8));

  std::vector<PositionIndex> targetSegmentCuts;
  targetSegmentCuts.push_back(1);
  targetSegmentCuts.push_back(3);
  targetSegmentCuts.push_back(5);
  targetSegmentCuts.push_back(6);
  targetSegmentCuts.push_back(7);
  targetSegmentCuts.push_back(8);

  std::vector<std::string> targetWordVec;
  targetWordVec.push_back("premier");
  targetWordVec.push_back("et");
  targetWordVec.push_back("Only");
  targetWordVec.push_back("t-shirt");
  targetWordVec.push_back("avec");
  targetWordVec.push_back("Logo");
  targetWordVec.push_back("22.9cm");
  targetWordVec.push_back("2x5");

  // Valid translation - respects all constraints
  bool isSatisfied = metadata.translationSatisfiesConstraints(sourceSegmentation, targetSegmentCuts, targetWordVec);
  EXPECT_FALSE(isSatisfied);
}

TEST_F(TranslationMetadataTest, translationViolatingTrgSegmentationConstraints)
{
  // Extract constraints
  metadata.obtainTransConstraints(getXmlString(), 0);

  // Prepare parameters
  SourceSegmentation sourceSegmentation;
  sourceSegmentation.push_back(std::make_pair(1, 1));
  sourceSegmentation.push_back(std::make_pair(2, 3));
  sourceSegmentation.push_back(std::make_pair(4, 5));
  sourceSegmentation.push_back(std::make_pair(6, 6));
  sourceSegmentation.push_back(std::make_pair(7, 7));
  sourceSegmentation.push_back(std::make_pair(8, 8));

  std::vector<PositionIndex> targetSegmentCuts;
  targetSegmentCuts.push_back(2);
  targetSegmentCuts.push_back(3);
  targetSegmentCuts.push_back(5);
  targetSegmentCuts.push_back(6);
  targetSegmentCuts.push_back(7);
  targetSegmentCuts.push_back(8);

  std::vector<std::string> targetWordVec;
  targetWordVec.push_back("premier");
  targetWordVec.push_back("et");
  targetWordVec.push_back("Only");
  targetWordVec.push_back("t-shirt");
  targetWordVec.push_back("avec");
  targetWordVec.push_back("Logo");
  targetWordVec.push_back("22.9cm");
  targetWordVec.push_back("2x5");

  // Valid translation - respects all constraints
  bool isSatisfied = metadata.translationSatisfiesConstraints(sourceSegmentation, targetSegmentCuts, targetWordVec);
  EXPECT_FALSE(isSatisfied);
}

TEST_F(TranslationMetadataTest, translationViolatingWordSelectionConstraints)
{
  // Extract constraints
  metadata.obtainTransConstraints(getXmlString(), 0);

  // Prepare parameters
  SourceSegmentation sourceSegmentation;
  sourceSegmentation.push_back(std::make_pair(1, 1));
  sourceSegmentation.push_back(std::make_pair(2, 3));
  sourceSegmentation.push_back(std::make_pair(4, 5));
  sourceSegmentation.push_back(std::make_pair(6, 6));
  sourceSegmentation.push_back(std::make_pair(7, 7));
  sourceSegmentation.push_back(std::make_pair(8, 8));

  std::vector<PositionIndex> targetSegmentCuts;
  targetSegmentCuts.push_back(1);
  targetSegmentCuts.push_back(3);
  targetSegmentCuts.push_back(5);
  targetSegmentCuts.push_back(6);
  targetSegmentCuts.push_back(7);
  targetSegmentCuts.push_back(8);

  std::vector<std::string> targetWordVec;
  targetWordVec.push_back("premier");
  targetWordVec.push_back("XXXXXXXX");
  targetWordVec.push_back("Only");
  targetWordVec.push_back("t-shirt");
  targetWordVec.push_back("avec");
  targetWordVec.push_back("Logo");
  targetWordVec.push_back("22.9cm");
  targetWordVec.push_back("2x5");

  // Valid translation - respects all constraints
  bool isSatisfied = metadata.translationSatisfiesConstraints(sourceSegmentation, targetSegmentCuts, targetWordVec);
  EXPECT_TRUE(isSatisfied);
}
