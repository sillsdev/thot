add_executable(thot_test
    _incrLexTableTest.h
    _phraseTableTest.h
    HatTriePhraseTableTest.cc
    IncrLexTableTest.cc
    KbMiraLlWuTest.cc
    MiraChrFTest.cc
    StlPhraseTableTest.cc
    TranslationMetadataTest.cc
)

target_include_directories(thot_test PRIVATE
    "${PROJECT_SOURCE_DIR}/src"
)

target_link_libraries(thot_test PRIVATE
    stack_dec
    phrase_models
    sw_models
    nlp_common
    gtest_main
)

include(GoogleTest)

gtest_discover_tests(thot_test)
