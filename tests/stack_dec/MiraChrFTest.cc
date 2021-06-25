#include "stack_dec/MiraChrF.h"

#include <cmath>
#include <gtest/gtest.h>

class MiraChrFTest : public testing::Test
{
protected:
  void SetUp() override
  {
    system_sentences.push_back("colourless green ideas sleep furiously");
    system_sentences.push_back("");
    system_sentences.push_back("colourless green ideas sleep furiously");
    system_sentences.push_back("");
    system_sentences.push_back(".");
    system_sentences.push_back("colourless green ideas sleep furiously");
    system_sentences.push_back("colorless greeny idea sleeps furious");
    system_sentences.push_back("áéíóúûüмариночкалучшевсех");

    reference_sentences.push_back("colourless green ideas sleep furiously");
    reference_sentences.push_back("colourless green ideas sleep furiously");
    reference_sentences.push_back("");
    reference_sentences.push_back("");
    reference_sentences.push_back(".");
    reference_sentences.push_back("colourless green ideas sleepfuriously");
    reference_sentences.push_back("colourless green ideas sleep furiously");
    reference_sentences.push_back("áéíóúûüмариночкалучшевсех");
  }

  MiraChrF chrf_metric;
  std::vector<std::string> system_sentences;
  std::vector<std::string> reference_sentences;
};

// This test works with the following metric parameters defined in chrf.h:
// MAX_NGRAM_LENGTH 4
// BETA 3
// CONSIDER_WHITESPACE true

TEST_F(MiraChrFTest, sentenceLevel)
{
  double score;
  // Candidate and reference are exactly the same
  chrf_metric.sentScore(system_sentences[0], reference_sentences[0], score);
  EXPECT_EQ(score, 1.0);

  // Reference is empty, candidate is not
  chrf_metric.sentScore(system_sentences[1], reference_sentences[1], score);
  EXPECT_EQ(score, 0.0);

  // Candidate is empty, reference is not
  chrf_metric.sentScore(system_sentences[2], reference_sentences[2], score);
  EXPECT_EQ(score, 0.0);

  // Both candidate and reference are empty
  chrf_metric.sentScore(system_sentences[3], reference_sentences[3], score);
  EXPECT_EQ(score, 1.0);

  // Candidate and reference contain only one exactly matching character
  chrf_metric.sentScore(system_sentences[4], reference_sentences[4], score);
  EXPECT_EQ(score, 1.0);

  // Candidate and reference differ only in white space
  chrf_metric.sentScore(system_sentences[5], reference_sentences[5], score);
  EXPECT_EQ(floor(score * 100) / 100, 0.95);

  // Candidate and reference differ in word forms
  chrf_metric.sentScore(system_sentences[6], reference_sentences[6], score);
  EXPECT_EQ(floor(score * 100) / 100, 0.74);

  // Candidate and reference contain non-latin characters and characters with diacritics
  chrf_metric.sentScore(system_sentences[7], reference_sentences[7], score);
  EXPECT_EQ(score, 1.0);
}

TEST_F(MiraChrFTest, corpusLevel)
{
  double score;
  chrf_metric.corpusScore(system_sentences, reference_sentences, score);
  EXPECT_EQ(floor(score * 100) / 100, 0.71);
}