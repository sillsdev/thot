#include "phrase_models/HatTriePhraseTable.h"

#include "_phraseTableTest.h"

#include <gtest/gtest.h>

template <>
BasePhraseTable* CreatePhraseTable<HatTriePhraseTable>()
{
  return new HatTriePhraseTable;
}

INSTANTIATE_TYPED_TEST_SUITE_P(HatTriePhraseTableTest, _phraseTableTest, HatTriePhraseTable);

class HatTriePhraseTableTest : public _phraseTableTest<HatTriePhraseTable>
{
};

TEST_F(HatTriePhraseTableTest, addSrcTrgInfo)
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

TEST_F(HatTriePhraseTableTest, iteratorsLoop)
{
  /* TEST:
      Check basic implementation of iterators - functions
      begin(), end() and operators (++ postfix, *).

      Note that iterator returns only target elements.
  */
  std::vector<WordIndex> s1 = getVector("Uniwersytet Gdanski");
  std::vector<WordIndex> t1 = getVector("Gdansk University");
  std::vector<WordIndex> s2 = getVector("Politechnika Gdanska");
  std::vector<WordIndex> t2 = getVector("Gdansk University of Technology");
  std::vector<WordIndex> empty;

  getTable()->clear();
  getTable()->incrCountsOfEntry(s1, t1, Count(40));
  getTable()->incrCountsOfEntry(s2, t2, Count(60));
  getTable()->incrCountsOfEntry(s1, t2, Count(0));
  getTable()->incrCountsOfEntry(s2, t1, Count(1));

  EXPECT_TRUE(getTable()->begin() != getTable()->end());
  EXPECT_TRUE(getTable()->begin() == getTable()->begin());

  int i = 0;
  const int MAX_ITER = 10;

  // Construct dictionary to record results returned by iterator
  // Dictionary structure: (key, (total count value, number of occurences))
  std::map<std::vector<WordIndex>, std::pair<Count, int>> d;
  d[t1] = std::make_pair(0, 0);
  d[t2] = std::make_pair(0, 0);

  for (HatTriePhraseTable::const_iterator iter = getTable()->begin(); iter != getTable()->end() && i < MAX_ITER;
       iter++, i++)
  {
    HatTriePhraseTable::PhraseInfoElement x = *iter;

    EXPECT_TRUE(x.first == t1 || x.first == t2) << "Phrase returned by iterator is not the one of expected targets";

    d[x.first].first += x.second;
    d[x.first].second++;
  }

  // Check if element returned by iterator is correct
  EXPECT_NEAR(41, d[t1].first.get_c_s(), EPSILON);
  EXPECT_EQ(1, d[t1].second);
  EXPECT_NEAR(60, d[t2].first.get_c_s(), EPSILON);
  EXPECT_EQ(1, d[t2].second);

  EXPECT_EQ(2, i);
}

TEST_F(HatTriePhraseTableTest, iteratorsOperatorsPlusPlusStar)
{
  /* TEST:
    Check basic implementation of iterators - function
    begin() and operators (++ prefix, ++ postfix, *, ->).

    Note that iterator returns only target elements.
  */
  bool found;

  std::vector<WordIndex> s1 = getVector("Uniwersytet Gdanski");
  std::vector<WordIndex> t1 = getVector("Gdansk University");
  std::vector<WordIndex> s2 = getVector("Politechnika Gdanska");
  std::vector<WordIndex> t2 = getVector("Gdansk University of Technology");
  std::vector<WordIndex> s3 = getVector("Gdanski Uniwersytet Medyczny");
  std::vector<WordIndex> t3 = getVector("Gdansk Medical University");
  std::vector<WordIndex> empty;

  getTable()->clear();
  getTable()->incrCountsOfEntry(s1, t1, Count(40));
  getTable()->incrCountsOfEntry(s2, t2, Count(60));
  getTable()->incrCountsOfEntry(s1, t2, Count(0));
  getTable()->incrCountsOfEntry(s2, t1, Count(1));
  getTable()->incrCountsOfEntry(s3, t3, Count(50));

  // Check correctness of iterators
  EXPECT_TRUE(getTable()->begin() != getTable()->end());
  EXPECT_TRUE(getTable()->begin() == getTable()->begin());

  // Check if the results returned by iterator are correct
  // and operators work as expected
  HatTriePhraseTable::const_iterator iter = getTable()->begin();
  EXPECT_EQ(t2, iter->first);
  EXPECT_NEAR(60, iter->second.get_c_s(), EPSILON);

  found = ++iter;
  EXPECT_TRUE(found);
  EXPECT_EQ(t1, (*iter).first);
  EXPECT_NEAR(41, (*iter).second.get_c_s(), EPSILON);

  found = (iter++);
  EXPECT_TRUE(found);
  EXPECT_EQ(t3, (*iter).first);
  EXPECT_NEAR(50, (*iter).second.get_c_s(), EPSILON);

  found = ++iter;
  EXPECT_FALSE(found);
  EXPECT_TRUE(iter == getTable()->end());
}

TEST_F(HatTriePhraseTableTest, iteratorsOperatorsEqualNotEqual)
{
  /* TEST:
    Check basic implementation of iterators - operators == and !=

    Note that iterator returns only target elements.
  */
  std::vector<WordIndex> s = getVector("kemping w Kretowinach");
  std::vector<WordIndex> t = getVector("camping Kretowiny");

  getTable()->clear();
  getTable()->incrCountsOfEntry(s, t, Count(1));

  HatTriePhraseTable::const_iterator iter1 = getTable()->begin();
  iter1++;
  HatTriePhraseTable::const_iterator iter2 = getTable()->begin();

  EXPECT_TRUE(iter1 == iter1);
  EXPECT_FALSE(iter1 != iter1);
  EXPECT_FALSE(iter1 == iter2);
  EXPECT_TRUE(iter1 != iter2);
}
