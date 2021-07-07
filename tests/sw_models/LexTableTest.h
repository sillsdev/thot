#pragma once

#include "nlp_common/MathDefs.h"
#include "sw_models/LexTable.h"

#include <gtest/gtest.h>

template <class T>
LexTable* CreateLexTable();

template <class T>
class LexTableTest : public testing::Test
{
protected:
  LexTableTest() : table(CreateLexTable<T>())
  {
  }

  ~LexTableTest() override
  {
    delete table;
  }

  T* getTable()
  {
    return dynamic_cast<T*>(table);
  }

  LexTable* table;
};

TYPED_TEST_SUITE_P(LexTableTest);

TYPED_TEST_P(LexTableTest, getSetDenominator)
{
  bool found;
  WordIndex s = 20;
  float denom = 1.22;

  this->table->clear();

  this->table->getDenominator(s, found);
  EXPECT_FALSE(found); // Element should not be found

  this->table->setDenominator(s, denom);
  float restoredDenom = this->table->getDenominator(s, found);
  EXPECT_TRUE(found); // Element should be found
  EXPECT_NEAR(denom, restoredDenom, EPSILON);
}

TYPED_TEST_P(LexTableTest, getSetNumerator)
{
  bool found;
  WordIndex s = 14;
  WordIndex t = 10;
  float numer = 15.7;

  this->table->clear();

  this->table->getNumerator(s, t, found);
  EXPECT_FALSE(found); // Element should not be found

  this->table->setNumerator(s, t, numer);
  float restoredNumer = this->table->getNumerator(s, t, found);
  EXPECT_TRUE(found); // Element should be found
  EXPECT_NEAR(numer, restoredNumer, EPSILON);
}

TYPED_TEST_P(LexTableTest, set)
{
  bool found;
  WordIndex s = 14;
  WordIndex t = 9;
  float numer = 1.9;
  float denom = 9.1;

  this->table->clear();

  this->table->getNumerator(s, t, found);
  EXPECT_FALSE(found); // Element should not be found
  this->table->getDenominator(s, found);
  EXPECT_FALSE(found); // Element should not be found

  this->table->set(s, t, numer, denom);

  float restoredNumer = this->table->getNumerator(s, t, found);
  EXPECT_TRUE(found); // Element should be found
  EXPECT_NEAR(numer, restoredNumer, EPSILON);

  float restoredDenom = this->table->getDenominator(s, found);
  EXPECT_TRUE(found); // Element should be found
  EXPECT_NEAR(denom, restoredDenom, EPSILON);
}

TYPED_TEST_P(LexTableTest, getTransForSource)
{
  bool found;

  WordIndex s1 = 1;
  WordIndex t1_1 = 2;
  WordIndex t1_2 = 3;

  WordIndex s2 = 9;
  WordIndex t2 = 11;

  this->table->clear();

  // Fill structure with data
  this->table->set(s1, t1_1, 2.2, 3.3);
  this->table->set(s1, t1_2, 4.4, 5.5);
  this->table->set(s2, t2, 22.1, 22.7);

  // Query structure and validate results
  std::set<WordIndex> transSet;

  // s1
  std::set<WordIndex> s1Set;
  s1Set.insert(t1_1);
  s1Set.insert(t1_2);

  found = this->table->getTransForSource(s1, transSet);
  EXPECT_TRUE(found);
  EXPECT_EQ((size_t)2, transSet.size());
  EXPECT_EQ(transSet, s1Set);

  // s2
  std::set<WordIndex> s2Set;
  s2Set.insert(t2);

  found = this->table->getTransForSource(s2, transSet);
  EXPECT_TRUE(found);
  EXPECT_EQ((size_t)1, transSet.size());
  EXPECT_EQ(transSet, s2Set);
}

REGISTER_TYPED_TEST_SUITE_P(LexTableTest, getSetDenominator, getSetNumerator, set, getTransForSource);
