add_executable(thot_test
    phrase_models/_phraseTableTest.h
    phrase_models/HatTriePhraseTableTest.cc
    phrase_models/StlPhraseTableTest.cc
    stack_dec/KbMiraLlWuTest.cc
    stack_dec/MiraChrFTest.cc
    stack_dec/TranslationMetadataTest.cc
    sw_models/_incrLexTableTest.h
    sw_models/IncrLexTableTest.cc
)

target_include_directories(thot_test PRIVATE
    "${PROJECT_SOURCE_DIR}/include"
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
