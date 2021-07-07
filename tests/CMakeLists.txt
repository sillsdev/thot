add_executable(thot_test
    phrase_models/_phraseTableTest.h
    phrase_models/HatTriePhraseTableTest.cc
    phrase_models/StlPhraseTableTest.cc
    stack_dec/KbMiraLlWuTest.cc
    stack_dec/MiraChrFTest.cc
    stack_dec/TranslationMetadataTest.cc
    sw_models/Ibm4AligModelTest.cc
    sw_models/LexTableTest.h
    sw_models/MemoryLexTableTest.cc
)

target_link_libraries(thot_test PRIVATE
    thot_lib
    gtest_main
)

include(GoogleTest)

gtest_discover_tests(thot_test)
