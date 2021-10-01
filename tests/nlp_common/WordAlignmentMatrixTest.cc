#include "nlp_common/WordAlignmentMatrix.h"

#include <gtest/gtest.h>

std::pair<WordAlignmentMatrix, WordAlignmentMatrix> createMatrices()
{
  WordAlignmentMatrix x{7, 9};
  x.set(0, 0);
  x.set(1, 5);
  x.set(2, 1);
  x.set(3, 2);
  x.set(3, 3);
  x.set(3, 4);
  x.set(4, 5);
  x.set(5, 3);

  WordAlignmentMatrix y{7, 9};
  y.set(0, 0);
  y.set(1, 1);
  y.set(2, 1);
  y.set(3, 4);
  y.set(4, 6);
  y.set(6, 8);

  return std::make_pair(std::move(x), std::move(y));
}

TEST(WordAlignmentMatrixTest, intersection)
{
  WordAlignmentMatrix x, y;
  std::tie(x, y) = createMatrices();

  x &= y;

  WordAlignmentMatrix expected{7, 9};
  expected.set(0, 0);
  expected.set(2, 1);
  expected.set(3, 4);
  EXPECT_EQ(x, expected);
}

TEST(WordAlignmentMatrixTest, union)
{
  WordAlignmentMatrix x, y;
  std::tie(x, y) = createMatrices();

  x |= y;

  WordAlignmentMatrix expected{7, 9};
  expected.set(0, 0);
  expected.set(1, 1);
  expected.set(1, 5);
  expected.set(2, 1);
  expected.set(3, 2);
  expected.set(3, 3);
  expected.set(3, 4);
  expected.set(4, 5);
  expected.set(4, 6);
  expected.set(5, 3);
  expected.set(6, 8);
  EXPECT_EQ(x, expected);
}

TEST(WordAlignmentMatrixTest, symmetr1)
{
  WordAlignmentMatrix x, y;
  std::tie(x, y) = createMatrices();

  x.symmetr1(y);

  WordAlignmentMatrix expected{7, 9};
  expected.set(0, 0);
  expected.set(1, 1);
  expected.set(2, 1);
  expected.set(3, 2);
  expected.set(3, 3);
  expected.set(3, 4);
  expected.set(4, 5);
  expected.set(4, 6);
  expected.set(6, 8);
  EXPECT_EQ(x, expected);
}

TEST(WordAlignmentMatrixTest, grow)
{
  WordAlignmentMatrix x, y;
  std::tie(x, y) = createMatrices();

  x.grow(y);

  WordAlignmentMatrix expected{7, 9};
  expected.set(0, 0);
  expected.set(1, 1);
  expected.set(2, 1);
  expected.set(3, 2);
  expected.set(3, 3);
  expected.set(3, 4);
  EXPECT_EQ(x, expected);
}

TEST(WordAlignmentMatrixTest, growDiag)
{
  WordAlignmentMatrix x, y;
  std::tie(x, y) = createMatrices();

  x.growDiag(y);

  WordAlignmentMatrix expected{7, 9};
  expected.set(0, 0);
  expected.set(1, 1);
  expected.set(2, 1);
  expected.set(3, 2);
  expected.set(3, 3);
  expected.set(3, 4);
  expected.set(4, 5);
  expected.set(4, 6);
  EXPECT_EQ(x, expected);
}

TEST(WordAlignmentMatrixTest, growDiagFinal)
{
  WordAlignmentMatrix x, y;
  std::tie(x, y) = createMatrices();

  x.growDiagFinal(y);

  WordAlignmentMatrix expected{7, 9};
  expected.set(0, 0);
  expected.set(1, 1);
  expected.set(2, 1);
  expected.set(3, 2);
  expected.set(3, 3);
  expected.set(3, 4);
  expected.set(4, 5);
  expected.set(4, 6);
  expected.set(5, 3);
  expected.set(6, 8);
  EXPECT_EQ(x, expected);
}

TEST(WordAlignmentMatrixTest, growDiagFinalAnd)
{
  WordAlignmentMatrix x, y;
  std::tie(x, y) = createMatrices();

  x.growDiagFinalAnd(y);

  WordAlignmentMatrix expected{7, 9};
  expected.set(0, 0);
  expected.set(1, 1);
  expected.set(2, 1);
  expected.set(3, 2);
  expected.set(3, 3);
  expected.set(3, 4);
  expected.set(4, 5);
  expected.set(4, 6);
  expected.set(6, 8);
  EXPECT_EQ(x, expected);
}
