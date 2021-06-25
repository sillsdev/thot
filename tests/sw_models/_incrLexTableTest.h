#pragma once

#include "nlp_common/MathDefs.h"
#include "sw_models/_incrLexTable.h"

#include <gtest/gtest.h>

template <class T>
_incrLexTable* CreateIncrLexTable();

template <class T>
class _incrLexTableTest : public testing::Test
{
protected:
  _incrLexTableTest() : table(CreateIncrLexTable<T>())
  {
  }
  ~_incrLexTableTest() override
  {
    delete table;
  }

  T* getTable()
  {
    return dynamic_cast<T*>(table);
  }

  _incrLexTable* table;
};

TYPED_TEST_SUITE_P(_incrLexTableTest);

TYPED_TEST_P(_incrLexTableTest, getSetLexDenom)
{
  bool found;
  WordIndex s = 20;
  float denom = 1.22;

  this->table->clear();

  this->table->getLexDenom(s, found);
  EXPECT_FALSE(found); // Element should not be found

  this->table->setLexDenom(s, denom);
  float restoredDenom = this->table->getLexDenom(s, found);
  EXPECT_TRUE(found); // Element should be found
  EXPECT_NEAR(denom, restoredDenom, EPSILON);
}

TYPED_TEST_P(_incrLexTableTest, getSetLexNumer)
{
  bool found;
  WordIndex s = 14;
  WordIndex t = 10;
  float numer = 15.7;

  this->table->clear();

  this->table->getLexNumer(s, t, found);
  EXPECT_FALSE(found); // Element should not be found

  this->table->setLexNumer(s, t, numer);
  float restoredNumer = this->table->getLexNumer(s, t, found);
  EXPECT_TRUE(found); // Element should be found
  EXPECT_NEAR(numer, restoredNumer, EPSILON);
}

TYPED_TEST_P(_incrLexTableTest, setLexNumerDenom)
{
  bool found;
  WordIndex s = 14;
  WordIndex t = 9;
  float numer = 1.9;
  float denom = 9.1;

  this->table->clear();

  this->table->getLexNumer(s, t, found);
  EXPECT_FALSE(found); // Element should not be found
  this->table->getLexDenom(s, found);
  EXPECT_FALSE(found); // Element should not be found

  this->table->setLexNumDen(s, t, numer, denom);

  float restoredNumer = this->table->getLexNumer(s, t, found);
  EXPECT_TRUE(found); // Element should be found
  EXPECT_NEAR(numer, restoredNumer, EPSILON);

  float restoredDenom = this->table->getLexDenom(s, found);
  EXPECT_TRUE(found); // Element should be found
  EXPECT_NEAR(denom, restoredDenom, EPSILON);
}

TYPED_TEST_P(_incrLexTableTest, getTransForSource)
{
  bool found;

  WordIndex s1 = 1;
  WordIndex t1_1 = 2;
  WordIndex t1_2 = 3;

  WordIndex s2 = 9;
  WordIndex t2 = 11;

  this->table->clear();

  // Fill structure with data
  this->table->setLexNumDen(s1, t1_1, 2.2, 3.3);
  this->table->setLexNumDen(s1, t1_2, 4.4, 5.5);
  this->table->setLexNumDen(s2, t2, 22.1, 22.7);

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

REGISTER_TYPED_TEST_SUITE_P(_incrLexTableTest, getSetLexDenom, getSetLexNumer, setLexNumerDenom, getTransForSource);
