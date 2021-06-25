#pragma once

#include "phrase_models/BasePhraseTable.h"

#include <gtest/gtest.h>

template <class T>
BasePhraseTable* CreatePhraseTable();

template <class T>
class _phraseTableTest : public testing::Test
{
protected:
  _phraseTableTest() : table(CreatePhraseTable<T>())
  {
  }
  ~_phraseTableTest() override
  {
    delete table;
  }

  T* getTable()
  {
    return dynamic_cast<T*>(table);
  }

  std::vector<WordIndex> getVector(std::string phrase)
  {
    std::vector<WordIndex> v;

    for (unsigned int i = 0; i < phrase.size(); i++)
    {
      v.push_back(phrase[i]);
    }

    return (v);
  }

  BasePhraseTable* table;
};

TYPED_TEST_SUITE_P(_phraseTableTest);

TYPED_TEST_P(_phraseTableTest, storeAndRestore)
{
  std::vector<WordIndex> s1 = this->getVector("Morag city");
  std::vector<WordIndex> s2 = this->getVector("Narie lake");
  Count cs1 = Count(5);
  Count cs2 = Count(2);
  this->table->clear();
  this->table->addSrcInfo(s1, cs1);
  this->table->addSrcInfo(s2, cs2);

  bool found;
  Count s1_count = this->table->getSrcInfo(s1, found);
  Count s2_count = this->table->getSrcInfo(s2, found);

  EXPECT_NEAR(5, s1_count.get_c_s(), EPSILON);
  EXPECT_NEAR(2, s2_count.get_c_s(), EPSILON);
}

TYPED_TEST_P(_phraseTableTest, addTableEntry)
{
  std::vector<WordIndex> s = this->getVector("Narie lake");
  std::vector<WordIndex> t = this->getVector("jezioro Narie");
  Count s_count = Count(3);
  Count t_count = Count(2);
  PhrasePairInfo ppi(s_count, t_count);

  this->table->clear();
  this->table->addTableEntry(s, t, ppi);

  EXPECT_NEAR(3, this->table->cSrc(s).get_c_s(), EPSILON);
  EXPECT_NEAR(2, this->table->cTrg(t).get_c_s(), EPSILON);
  EXPECT_NEAR(2, this->table->cSrcTrg(s, t).get_c_st(), EPSILON);
}

TYPED_TEST_P(_phraseTableTest, incCountsOfEntry)
{
  std::vector<WordIndex> s = this->getVector("Narie lake");
  std::vector<WordIndex> t = this->getVector("jezioro Narie");
  Count c_init = Count(3);
  Count c = Count(17);

  this->table->clear();
  this->table->addSrcInfo(s, c_init);
  this->table->incrCountsOfEntry(s, t, c);

  EXPECT_NEAR(20, this->table->cSrc(s).get_c_s(), EPSILON);
  EXPECT_NEAR(17, this->table->cTrg(t).get_c_s(), EPSILON);
  EXPECT_NEAR(17, this->table->cSrcTrg(s, t).get_c_st(), EPSILON);
}

TYPED_TEST_P(_phraseTableTest, getEntriesForTarget)
{
  BasePhraseTable::SrcTableNode node;
  std::vector<WordIndex> s1_1 = this->getVector("Pasleka river");
  std::vector<WordIndex> s1_2 = this->getVector("Pasleka");
  std::vector<WordIndex> t1_1 = this->getVector("rzeka Pasleka");
  std::vector<WordIndex> t1_2 = this->getVector("Pasleka");
  std::vector<WordIndex> s2 = this->getVector("river");
  std::vector<WordIndex> t2 = this->getVector("rzeka");
  Count c = Count(1);

  this->table->clear();
  this->table->incrCountsOfEntry(s1_1, t1_1, c);
  this->table->incrCountsOfEntry(s1_2, t1_1, c);
  this->table->incrCountsOfEntry(s1_1, t1_2, c);
  this->table->incrCountsOfEntry(s2, t2, c);

  bool result;
  // Looking for phrases for which 'rzeka Pasleka' is translation
  result = this->table->getEntriesForTarget(t1_1, node);
  EXPECT_TRUE(result);
  EXPECT_EQ((size_t)2, node.size());
  EXPECT_NEAR(2, node[s1_1].first.get_c_s(), EPSILON);
  EXPECT_NEAR(1, node[s1_1].second.get_c_st(), EPSILON);
  EXPECT_NEAR(1, node[s1_2].first.get_c_s(), EPSILON);
  EXPECT_NEAR(1, node[s1_2].second.get_c_st(), EPSILON);

  // Looking for phrases for which 'Pasleka' is translation
  result = this->table->getEntriesForTarget(t1_2, node);
  EXPECT_TRUE(result);
  EXPECT_EQ((size_t)1, node.size());
  EXPECT_NEAR(2, node[s1_1].first.get_c_s(), EPSILON);
  EXPECT_NEAR(1, node[s1_1].second.get_c_st(), EPSILON);

  // Looking for phrases for which 'rzeka' is translation
  result = this->table->getEntriesForTarget(t2, node);
  EXPECT_TRUE(result);
  EXPECT_EQ((size_t)1, node.size());
  EXPECT_NEAR(1, node[s2].first.get_c_s(), EPSILON);
  EXPECT_NEAR(1, node[s2].second.get_c_st(), EPSILON);

  // 'xyz'' key shoud not be found
  result = this->table->getEntriesForTarget(this->getVector("xyz"), node);
  EXPECT_FALSE(result);
}

TYPED_TEST_P(_phraseTableTest, retrievingSubphrase)
{
  /* TEST:
     Accessing element with the subphrase should return count 0
  */
  bool found;
  std::vector<WordIndex> s = this->getVector("Hello");
  std::vector<WordIndex> t1 = this->getVector("Buenos Dias");
  std::vector<WordIndex> t2 = this->getVector("Buenos");

  Count c = Count(1);

  this->table->clear();
  this->table->addSrcInfo(s, c);
  this->table->incrCountsOfEntry(s, t1, c);
  c = this->table->getSrcTrgInfo(s, t2, found);

  EXPECT_FALSE(found);
  EXPECT_NEAR(0, c.get_c_s(), EPSILON);
}

TYPED_TEST_P(_phraseTableTest, retrieveNonLeafPhrase)
{
  /* TEST:
     Phrases with count > 0 and not stored in the leaves
     should be also retrieved
  */
  bool found;
  BasePhraseTable::SrcTableNode node;
  std::vector<WordIndex> s = this->getVector("Hello");
  std::vector<WordIndex> t1 = this->getVector("Buenos Dias");
  std::vector<WordIndex> t2 = this->getVector("Buenos");

  Count c = Count(1);

  this->table->clear();
  this->table->incrCountsOfEntry(s, t1, c);
  this->table->incrCountsOfEntry(s, t2, c);

  // Check phrases and their counts
  // Phrase pair 1
  c = this->table->getSrcTrgInfo(s, t1, found);

  EXPECT_TRUE(found);
  EXPECT_NEAR(1, c.get_c_s(), EPSILON);
  // Phrase pair 2
  c = this->table->getSrcTrgInfo(s, t2, found);

  EXPECT_TRUE(found);
  EXPECT_NEAR(1, c.get_c_s(), EPSILON);

  // Looking for phrases for which 'Buenos' is translation
  found = this->table->getEntriesForTarget(t2, node);
  EXPECT_TRUE(found);
  EXPECT_NEAR(1, node.size(), EPSILON);
}

TYPED_TEST_P(_phraseTableTest, getEntriesForSource)
{
  /* TEST:
     Find translations for the source phrase
  */
  bool found;
  BasePhraseTable::TrgTableNode node;
  std::vector<WordIndex> s1 = this->getVector("jezioro Narie");
  std::vector<WordIndex> t1_1 = this->getVector("Narie lake");
  std::vector<WordIndex> t1_2 = this->getVector("Narie");
  std::vector<WordIndex> s2 = this->getVector("jezioro Skiertag");
  std::vector<WordIndex> t2_1 = this->getVector("Skiertag");
  std::vector<WordIndex> s3 = this->getVector("jezioro Jeziorak");
  std::vector<WordIndex> t3_1 = this->getVector("Jeziorak lake");
  std::vector<WordIndex> t3_2 = this->getVector("Jeziorak");

  Count c = Count(1);

  // Prepare data struture
  this->table->clear();
  // Add Narie phrases
  this->table->incrCountsOfEntry(s1, t1_1, c);
  this->table->incrCountsOfEntry(s1, t1_2, c);
  // Add Skiertag phrases
  this->table->incrCountsOfEntry(s2, t2_1, c);
  // Add Jeziorak phrases
  this->table->incrCountsOfEntry(s3, t3_1, c);
  this->table->incrCountsOfEntry(s3, t3_2, c);

  // Looking for translations
  // Narie phrases
  found = this->table->getEntriesForSource(s1, node);
  EXPECT_TRUE(found);
  EXPECT_EQ((size_t)2, node.size());
  // Skiertag phrases
  found = this->table->getEntriesForSource(s2, node);
  EXPECT_TRUE(found);
  EXPECT_EQ((size_t)1, node.size());
  // Jeziorak phrases
  found = this->table->getEntriesForSource(s3, node);
  EXPECT_TRUE(found);
  EXPECT_EQ((size_t)2, node.size());
}

TYPED_TEST_P(_phraseTableTest, retrievingEntriesWithCountEqualZero)
{
  /* TEST:
     Function getEntriesForTarget for retrieving entries should skip
     elements with count equals 0
  */
  bool found;
  BasePhraseTable::SrcTableNode node;
  std::vector<WordIndex> s1 = this->getVector("Palac Dohnow");
  std::vector<WordIndex> s2 = this->getVector("Palac Dohnow w Moragu");
  std::vector<WordIndex> t = this->getVector("Dohn's Palace");

  this->table->clear();
  this->table->incrCountsOfEntry(s1, t, Count(1));
  this->table->incrCountsOfEntry(s2, t, Count(0));

  found = this->table->getEntriesForTarget(t, node);

  EXPECT_TRUE(found);
  EXPECT_EQ((size_t)1, node.size());
}

TYPED_TEST_P(_phraseTableTest, getNbestForTrg)
{
  /* TEST:
     Check if method getNbestForTrg returns correct elements
  */
  bool found;
  NbestTableNode<PhraseTransTableNodeData> node;
  NbestTableNode<PhraseTransTableNodeData>::iterator iter;

  // Fill phrase table with data
  std::vector<WordIndex> s1 = this->getVector("city hall");
  std::vector<WordIndex> s2 = this->getVector("city hall in Morag");
  std::vector<WordIndex> s3 = this->getVector("town hall");
  std::vector<WordIndex> s4 = this->getVector("town hall in Morag");
  std::vector<WordIndex> t = this->getVector("ratusz miejski w Moragu");

  this->table->clear();
  this->table->incrCountsOfEntry(s1, t, Count(4));
  this->table->incrCountsOfEntry(s2, t, Count(2));
  this->table->incrCountsOfEntry(s3, t, Count(3));
  this->table->incrCountsOfEntry(s4, t, Count(0));

  // Returned elements should not exceed number of elements
  // in the structure
  found = this->table->getNbestForTrg(t, node, 10);

  EXPECT_TRUE(found);
  EXPECT_EQ((size_t)3, node.size());

  // If there are more available elements, only elements
  // with the highest score should be returned
  found = this->table->getNbestForTrg(t, node, 2);

  EXPECT_TRUE(found);
  EXPECT_EQ((size_t)2, node.size());

  iter = node.begin();
  EXPECT_EQ(iter->second, s1);
  iter++;
  EXPECT_EQ(iter->second, s3);
}

TYPED_TEST_P(_phraseTableTest, addSrcTrgInfo)
{
  /* TEST:
     Check if two keys were added (for (s, t) and (t, s) vectors)
     and if their values are the same
  */
  std::vector<WordIndex> s = this->getVector("jezioro Skiertag");
  std::vector<WordIndex> t = this->getVector("Skiertag lake");

  Count c = Count(1);

  this->table->clear();
  this->table->addSrcInfo(s, c);
  this->table->addSrcTrgInfo(s, t, c);

  Count src_trg_count = this->table->cSrcTrg(s, t);

  EXPECT_NEAR(1, src_trg_count.get_c_s(), EPSILON);
}

TYPED_TEST_P(_phraseTableTest, pSrcGivenTrg)
{
  /* TEST:
     Check retrieving probabilities for phrases based on stored
     counts for a given target.
  */
  std::vector<WordIndex> s1 = this->getVector("Morag");
  std::vector<WordIndex> s2 = this->getVector("Gdansk");
  std::vector<WordIndex> t1 = this->getVector("Candas");
  std::vector<WordIndex> t2 = this->getVector("Aviles");

  // Fill phrase table with data
  this->table->incrCountsOfEntry(s1, t1, Count(3));
  this->table->incrCountsOfEntry(s2, t1, Count(7));
  this->table->incrCountsOfEntry(s1, t2, Count(1));
  this->table->incrCountsOfEntry(s2, t2, Count(2));

  // Check probabilities
  EXPECT_NEAR(0.3, this->table->pSrcGivenTrg(s1, t1), EPSILON);
  EXPECT_NEAR(0.7, this->table->pSrcGivenTrg(s2, t1), EPSILON);
  EXPECT_NEAR(1. / 3., this->table->pSrcGivenTrg(s1, t2), EPSILON);
  EXPECT_NEAR(2. / 3., this->table->pSrcGivenTrg(s2, t2), EPSILON);
}

TYPED_TEST_P(_phraseTableTest, pTrgGivenSrc)
{
  /* TEST:
     Check retrieving probabilities for phrases based on stored
     counts for a given source.
  */
  std::vector<WordIndex> s1 = this->getVector("Morag");
  std::vector<WordIndex> s2 = this->getVector("Gdansk");
  std::vector<WordIndex> t1 = this->getVector("Candas");
  std::vector<WordIndex> t2 = this->getVector("Aviles");

  // Fill phrase table with data
  this->table->incrCountsOfEntry(s1, t1, Count(10));
  this->table->incrCountsOfEntry(s1, t2, Count(12));

  this->table->incrCountsOfEntry(s2, t1, Count(11));
  this->table->incrCountsOfEntry(s2, t2, Count(13));

  // Check probabilities
  EXPECT_NEAR(10. / 22., this->table->pTrgGivenSrc(s1, t1), EPSILON);
  EXPECT_NEAR(12. / 22., this->table->pTrgGivenSrc(s1, t2), EPSILON);
  EXPECT_NEAR(11. / 24., this->table->pTrgGivenSrc(s2, t1), EPSILON);
  EXPECT_NEAR(13. / 24., this->table->pTrgGivenSrc(s2, t2), EPSILON);
}

TYPED_TEST_P(_phraseTableTest, addingSameSrcAndTrg)
{
  /* TEST:
     Check if the results are returned correctly when source
     and target has the same values.
  */
  std::vector<WordIndex> v1 = this->getVector("Morag");
  std::vector<WordIndex> v2 = this->getVector("~ \" ()( -");

  // Fill phrase table with data
  this->table->incrCountsOfEntry(v1, v1, Count(1));
  this->table->incrCountsOfEntry(v1, v2, Count(2));
  this->table->incrCountsOfEntry(v2, v1, Count(4));
  this->table->incrCountsOfEntry(v2, v2, Count(8));

  // Check probabilities
  EXPECT_NEAR(1 + 2, this->table->cSrc(v1).get_c_s(), EPSILON);
  EXPECT_NEAR(1 + 4, this->table->cTrg(v1).get_c_s(), EPSILON);
  EXPECT_NEAR(4 + 8, this->table->cSrc(v2).get_c_s(), EPSILON);
  EXPECT_NEAR(2 + 8, this->table->cTrg(v2).get_c_s(), EPSILON);
}

TYPED_TEST_P(_phraseTableTest, size)
{
  /* TEST:
     Check if number of elements in the phrase table is returned correctly
  */
  this->table->clear();
  EXPECT_EQ((size_t)0, this->table->size()); // Collection after cleaning should be empty

  // Fill phrase table with data
  this->table->incrCountsOfEntry(this->getVector("kemping w Kretowinach"), this->getVector("camping Kretowiny"),
                                 Count(1));
  this->table->incrCountsOfEntry(this->getVector("kemping w Kretowinach"), this->getVector("camping in Kretowiny"),
                                 Count(2));

  EXPECT_EQ((size_t)5, this->table->size());

  this->table->clear();
  EXPECT_EQ((size_t)0, this->table->size()); // Collection after cleaning should be empty

  this->table->incrCountsOfEntry(this->getVector("Pan Samochodzik"), this->getVector("Mr Car"), Count(1));
  this->table->incrCountsOfEntry(this->getVector("Pan Samochodzik"), this->getVector("Pan Samochodzik"), Count(4));
  this->table->incrCountsOfEntry(this->getVector("Pan Samochodzik"), this->getVector("Mister Automobile"), Count(20));
  this->table->incrCountsOfEntry(this->getVector("Pan Samochodzik"), this->getVector("Mr Automobile"), Count(24));

  EXPECT_EQ((size_t)9, this->table->size());

  this->table->incrCountsOfEntry(this->getVector("Pierwsza przygoda Pana Samochodzika"),
                                 this->getVector("First Adventure of Mister Automobile"), Count(5));
  this->table->incrCountsOfEntry(this->getVector("Pierwsza przygoda Pana Samochodzika"),
                                 this->getVector("First Adventure of Pan Samochodzik"), Count(7));

  EXPECT_EQ((size_t)(9 + 5), this->table->size());
}

TYPED_TEST_P(_phraseTableTest, subkeys)
{
  /* TEST:
     Check if subkeys are stored correctly
  */

  // Fill phrase table with data
  this->table->clear();

  // Define vectors
  std::vector<WordIndex> s1 = this->getVector("Pan Samochodzik");
  std::vector<WordIndex> t1_1 = this->getVector("Mr Car");
  std::vector<WordIndex> t1_2 = this->getVector("Pan");
  std::vector<WordIndex> t1_3 = this->getVector("Mr");

  std::vector<WordIndex> s2 = this->getVector("Pan");
  std::vector<WordIndex> t2_1 = this->getVector("Mister");
  std::vector<WordIndex> t2_2 = this->getVector("Mr");

  // Insert data to phrase table
  this->table->incrCountsOfEntry(s1, t1_1, Count(1));
  this->table->incrCountsOfEntry(s1, t1_2, Count(2));
  this->table->incrCountsOfEntry(s1, t1_3, Count(4));

  this->table->incrCountsOfEntry(s2, t2_1, Count(8));
  this->table->incrCountsOfEntry(s2, t2_2, Count(16));

  EXPECT_EQ((size_t)11, this->table->size());

  // Check count values
  EXPECT_NEAR(1 + 2 + 4, this->table->cSrc(s1).get_c_s(), EPSILON);
  EXPECT_NEAR(1, this->table->cTrg(t1_1).get_c_s(), EPSILON);
  EXPECT_NEAR(2, this->table->cTrg(t1_2).get_c_s(), EPSILON);
  EXPECT_NEAR(4 + 16, this->table->cTrg(t1_3).get_c_s(), EPSILON);
  EXPECT_NEAR(1, this->table->cSrcTrg(s1, t1_1).get_c_st(), EPSILON);
  EXPECT_NEAR(2, this->table->cSrcTrg(s1, t1_2).get_c_st(), EPSILON);
  EXPECT_NEAR(4, this->table->cSrcTrg(s1, t1_3).get_c_st(), EPSILON);

  EXPECT_NEAR(8 + 16, this->table->cSrc(s2).get_c_s(), EPSILON);
  EXPECT_NEAR(8, this->table->cTrg(t2_1).get_c_s(), EPSILON);
  EXPECT_NEAR(4 + 16, this->table->cTrg(t2_2).get_c_s(), EPSILON);
  EXPECT_NEAR(8, this->table->cSrcTrg(s2, t2_1).get_c_st(), EPSILON);
  EXPECT_NEAR(16, this->table->cSrcTrg(s2, t2_2).get_c_st(), EPSILON);
}

TYPED_TEST_P(_phraseTableTest, code32bitRange)
{
  /* TEST:
     Check if phrase table supports codes from positive integer range
  */
  this->table->clear();

  std::vector<WordIndex> minVector, maxVector;

  minVector.push_back(0);
  maxVector.push_back(0x7FFFFFFE);

  // Insert data to phrase table and check their correctness
  this->table->incrCountsOfEntry(minVector, maxVector, Count(20));
  EXPECT_EQ((size_t)3, this->table->size());
  EXPECT_NEAR(20, this->table->cSrcTrg(minVector, maxVector).get_c_st(), EPSILON);
}

TYPED_TEST_P(_phraseTableTest, byteMax)
{
  /* TEST:
     Check if items with maximum byte values are added correctly
  */
  this->table->clear();

  std::vector<WordIndex> s, t;
  s.push_back(201);
  s.push_back(8);
  t.push_back(255);

  // Insert data and check their correctness
  this->table->incrCountsOfEntry(s, t, Count(1));
  EXPECT_EQ((size_t)3, this->table->size());
  EXPECT_NEAR(1, this->table->cSrcTrg(s, t).get_c_st(), EPSILON);
}

TYPED_TEST_P(_phraseTableTest, byteMin)
{
  /* TEST:
     Check if items with minimum byte values are added correctly
  */
  this->table->clear();

  std::vector<WordIndex> s1, s2, t1, t2;
  // s1
  s1.push_back(4);
  // s2
  s2.push_back(0);
  s2.push_back(1);
  s2.push_back(0);
  // t1
  t1.push_back(0);
  t1.push_back(3);
  // t2
  t2.push_back(0);
  t2.push_back(3);
  t2.push_back(0);

  // Insert data and check their correctness
  this->table->incrCountsOfEntry(s1, t1, Count(1));
  this->table->incrCountsOfEntry(s2, t2, Count(1));
  EXPECT_EQ((size_t)6, this->table->size());
  EXPECT_NEAR(1, this->table->cSrcTrg(s2, t2).get_c_st(), EPSILON);

  bool found;
  BasePhraseTable::SrcTableNode node;
  found = this->table->getEntriesForTarget(t2, node);

  EXPECT_TRUE(found);
  EXPECT_EQ((size_t)1, node.size());
  EXPECT_NEAR(1, node[s2].first.get_c_s(), EPSILON);
  EXPECT_NEAR(1, node[s2].second.get_c_s(), EPSILON);
}

REGISTER_TYPED_TEST_SUITE_P(_phraseTableTest, storeAndRestore, addTableEntry, incCountsOfEntry, getEntriesForTarget,
                            retrievingSubphrase, retrieveNonLeafPhrase, getEntriesForSource,
                            retrievingEntriesWithCountEqualZero, getNbestForTrg, addSrcTrgInfo, pSrcGivenTrg,
                            pTrgGivenSrc, addingSameSrcAndTrg, size, subkeys, code32bitRange, byteMax, byteMin);
