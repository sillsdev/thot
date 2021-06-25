#include "sw_models/IncrLexTable.h"

#include "_incrLexTableTest.h"

#include <gtest/gtest.h>

template <>
_incrLexTable* CreateIncrLexTable<IncrLexTable>()
{
  return new IncrLexTable;
}

INSTANTIATE_TYPED_TEST_SUITE_P(IncrLexTableTest, _incrLexTableTest, IncrLexTable);
