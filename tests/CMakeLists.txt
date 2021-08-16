add_executable(thot_test
    sw_models/TestUtils.h
    sw_models/TestUtils.cc
    sw_models/MemoryLexTableTest.cc
    sw_models/LexTableTest.h
    sw_models/IncrHmmAlignmentModelTest.cc
    sw_models/Ibm4AlignmentModelTest.cc
    stack_dec/TranslationMetadataTest.cc
    stack_dec/MiraChrFTest.cc
    stack_dec/KbMiraLlWuTest.cc
    phrase_models/StlPhraseTableTest.cc
    phrase_models/HatTriePhraseTableTest.cc
    phrase_models/_phraseTableTest.h
)

target_link_libraries(thot_test PRIVATE
    thot_lib
    gtest_main
)

include(GoogleTest)

gtest_discover_tests(thot_test)
