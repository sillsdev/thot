# Diagnostic-only, off by default: links mimalloc into thot_test to measure whether
# the default allocator is a scaling bottleneck under concurrent eflomal training
# (see eflomal-optimizations.md, Phase 1 opt 1). Not used by thot_lib/the shipped
# library - this only swaps the allocator for benchmark runs of the test binary.
option(THOT_BENCH_MIMALLOC "Link mimalloc into thot_test for allocator-contention benchmarking" OFF)
if(THOT_BENCH_MIMALLOC)
    include(FetchContent)
    FetchContent_Declare(
        mimalloc
        GIT_REPOSITORY https://github.com/microsoft/mimalloc.git
        GIT_TAG v2.1.7
    )
    set(MI_BUILD_SHARED OFF CACHE BOOL "" FORCE)
    set(MI_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(MI_OVERRIDE ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(mimalloc)
endif()

add_executable(thot_test
    nlp_common/WordAlignmentMatrixTest.cc
    phrase_models/_phraseTableTest.h
    phrase_models/HatTriePhraseTableTest.cc
    phrase_models/StlPhraseTableTest.cc
    stack_dec/KbMiraLlWuTest.cc
    stack_dec/MiraChrFTest.cc
    stack_dec/PhrLocalSwLiTmTest.cc
    stack_dec/TranslationMetadataTest.cc
    sw_models/EflomalAlignmentModelTest.cc
    sw_models/FastAlignModelTest.cc
    sw_models/Ibm4AlignmentModelTest.cc
    sw_models/IncrHmmAlignmentModelTest.cc
    sw_models/LexTableTest.h
    sw_models/MemoryLexTableTest.cc
    sw_models/SymmetrizedAlignmentModelTest.cc
    sw_models/TestUtils.cc
    sw_models/TestUtils.h
)

target_link_libraries(thot_test PRIVATE
    thot_lib
    gtest_main
)
if(THOT_BENCH_MIMALLOC)
    target_link_libraries(thot_test PRIVATE mimalloc-static)
    target_compile_definitions(thot_test PRIVATE THOT_BENCH_MIMALLOC)
    target_include_directories(thot_test PRIVATE ${mimalloc_SOURCE_DIR}/include)
endif()

include(GoogleTest)

gtest_discover_tests(thot_test)
