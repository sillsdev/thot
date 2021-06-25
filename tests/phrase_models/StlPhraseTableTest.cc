#include "phrase_models/StlPhraseTable.h"

#include "_phraseTableTest.h"

#include <gtest/gtest.h>

template <>
BasePhraseTable* CreatePhraseTable<StlPhraseTable>()
{
  return new StlPhraseTable;
}

INSTANTIATE_TYPED_TEST_SUITE_P(StlPhraseTableTest, _phraseTableTest, StlPhraseTable);

class StlPhraseTableTest : public _phraseTableTest<StlPhraseTable>
{
};

TEST_F(StlPhraseTableTest, addSrcTrgInfo)
{
  /* TEST:
      Check if two keys were added (for (s, t) and (t, s) vectors)
      and if their values are the same
  */
  bool found;

  std::vector<WordIndex> s = getVector("jezioro Skiertag");
  std::vector<WordIndex> t = getVector("Skiertag lake");

  Count c = Count(1);

  getTable()->clear();
  getTable()->addSrcInfo(s, c);
  getTable()->addSrcTrgInfo(s, t, c);

  Count src_trg_count = getTable()->cSrcTrg(s, t);
  Count trg_src_count = getTable()->getSrcTrgInfo(s, t, found);

  EXPECT_TRUE(found);
  EXPECT_NEAR(1, src_trg_count.get_c_s(), EPSILON);
  EXPECT_NEAR(src_trg_count.get_c_s(), trg_src_count.get_c_s(), EPSILON);
}

TEST_F(StlPhraseTableTest, iteratorsLoop)
{
  /* TEST:
      Check basic implementation of iterators - functions
      begin(), end() and operators (++ postfix, *).
  */
  std::vector<WordIndex> s = getVector("jezioro Skiertag");
  std::vector<WordIndex> t = getVector("Skiertag lake");
  std::vector<WordIndex> empty;

  getTable()->clear();
  getTable()->incrCountsOfEntry(s, t, Count(1));

  EXPECT_TRUE(getTable()->begin() != getTable()->end());
  EXPECT_TRUE(getTable()->begin() == getTable()->begin());

  int i = 0;
  const int MAX_ITER = 10;

  // Construct dictionary to record results returned by iterator
  // Dictionary structure: (key, (total count value, number of occurences))
  std::map<StlPhraseTable::PhraseInfoElementKey, std::pair<int, int>> d;
  StlPhraseTable::PhraseInfoElementKey s_key = std::make_pair(s, empty);
  StlPhraseTable::PhraseInfoElementKey t_key = std::make_pair(empty, t);
  StlPhraseTable::PhraseInfoElementKey st_key = std::make_pair(s, t);
  d[s_key] = std::make_pair(0, 0);
  d[t_key] = std::make_pair(0, 0);
  d[st_key] = std::make_pair(0, 0);

  for (StlPhraseTable::const_iterator iter = getTable()->begin(); iter != getTable()->end() && i < MAX_ITER;
       iter++, i++)
  {
    StlPhraseTable::PhraseInfoElement x = *iter;
    d[x.first].first += x.second;
    d[x.first].second++;
  }

  // Check if element returned by iterator is correct
  EXPECT_EQ((size_t)3, d.size());
  EXPECT_EQ(1, d[s_key].first);
  EXPECT_EQ(1, d[s_key].second);
  EXPECT_EQ(1, d[t_key].first);
  EXPECT_EQ(1, d[t_key].second);
  EXPECT_EQ(1, d[st_key].first);
  EXPECT_EQ(1, d[st_key].second);

  EXPECT_EQ(3, i);
}

TEST_F(StlPhraseTableTest, iteratorsOperatorsPlusPlusStar)
{
  /* TEST:
    Check basic implementation of iterators - function
    begin() and operators (++ prefix, ++ postfix, *, ->).
  */
  bool found = true;

  std::vector<WordIndex> s = getVector("zamek krzyzacki w Malborku");
  std::vector<WordIndex> t = getVector("teutonic castle in Malbork");
  std::vector<WordIndex> empty;

  getTable()->clear();
  getTable()->incrCountsOfEntry(s, t, Count(2));

  // Construct dictionary to record results returned by iterator
  // Dictionary structure: (key, (total count value, number of occurences))
  std::map<StlPhraseTable::PhraseInfoElementKey, std::pair<int, int>> d;
  StlPhraseTable::PhraseInfoElementKey s_key = std::make_pair(s, empty);
  StlPhraseTable::PhraseInfoElementKey t_key = std::make_pair(empty, t);
  StlPhraseTable::PhraseInfoElementKey st_key = std::make_pair(s, t);
  d[s_key] = std::make_pair(0, 0);
  d[t_key] = std::make_pair(0, 0);
  d[st_key] = std::make_pair(0, 0);

  for (StlPhraseTable::const_iterator iter = getTable()->begin(); iter != getTable()->end(); found = (iter++))
  {
    EXPECT_TRUE(found);
    StlPhraseTable::PhraseInfoElement x = *iter;
    d[x.first].first += x.second;
    d[x.first].second++;
  }

  // Iterating beyond the last element should return FALSE value
  EXPECT_FALSE(found);

  // Check if element returned by iterator is correct
  EXPECT_EQ((size_t)3, d.size());
  EXPECT_EQ(2, d[s_key].first);
  EXPECT_EQ(1, d[s_key].second);
  EXPECT_EQ(2, d[t_key].first);
  EXPECT_EQ(1, d[t_key].second);
  EXPECT_EQ(2, d[st_key].first);
  EXPECT_EQ(1, d[st_key].second);
}

TEST_F(StlPhraseTableTest, iteratorsOperatorsEqualNotEqual)
{
  /* TEST:
    Check basic implementation of iterators - operators == and !=
  */
  std::vector<WordIndex> s = getVector("kemping w Kretowinach");
  std::vector<WordIndex> t = getVector("camping Kretowiny");

  getTable()->clear();
  getTable()->incrCountsOfEntry(s, t, Count(1));

  StlPhraseTable::const_iterator iter1 = getTable()->begin();
  iter1++;
  StlPhraseTable::const_iterator iter2 = getTable()->begin();

  EXPECT_TRUE(iter1 == iter1);
  EXPECT_FALSE(iter1 != iter1);
  EXPECT_FALSE(iter1 == iter2);
  EXPECT_TRUE(iter1 != iter2);
}
