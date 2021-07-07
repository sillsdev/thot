#include "LexTableTest.h"
#include "sw_models/MemoryLexTable.h"

#include <gtest/gtest.h>

template <>
LexTable* CreateLexTable<MemoryLexTable>()
{
  return new MemoryLexTable;
}

INSTANTIATE_TYPED_TEST_SUITE_P(MemoryLexTableTest, LexTableTest, MemoryLexTable);
