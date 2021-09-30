#include "sw_models/FastAlignModel.h"

#include "TestUtils.h"

#include <gtest/gtest.h>

TEST(FastAlignModelTest, trainEmpty)
{
  FastAlignModel model;
  EXPECT_NO_THROW(train(model));
}
