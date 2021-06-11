add_library(phrase_models STATIC
    _incrPhraseModel.h _incrPhraseModel.cc
    _wbaIncrPhraseModel.h _wbaIncrPhraseModel.cc
    AligInfo.h
    AlignmentContainer.h AlignmentContainer.cc
    AlignmentExtractor.h AlignmentExtractor.cc
    BaseCountPhraseModel.h
    BaseIncrPhraseModel.h BaseIncrPhraseModel.cc
    BasePhraseModel.h BasePhraseModel.cc
    BasePhrasePairFilter.h
    BasePhraseTable.h
    BpSet.h BpSet.cc
    BpSetInfo.h
    CategPhrasePairFilter.h CategPhrasePairFilter.cc
    CellAlignment.h
    CellID.h
    HatTriePhraseTable.h HatTriePhraseTable.cc
    IncrPhraseModel.h IncrPhraseModel.cc
    PhraseDefs.h
    PhraseExtractionCell.h
    PhraseExtractionTable.h PhraseExtractionTable.cc
    PhraseExtractParameters.h
    PhraseExtractUtils.h PhraseExtractUtils.cc
    PhraseId.h
    PhrasePair.h
    PhrasePairInfo.h
    PhraseSortCriterion.h
    PhraseTransTableNodeData.h
    SegLenTable.h SegLenTable.cc
    SentSegmentation.h
    SrcSegmLenTable.h SrcSegmLenTable.cc
    SrfBisegm.h
    SrfNodeInfo.h
    SrfNodeInfoMap.h SrfNodeInfoMap.cc
    SrfNodeKey.h SrfNodeKey.cc
    StlPhraseTable.h StlPhraseTable.cc
    StrictCategPhrasePairFilter.h StrictCategPhrasePairFilter.cc
    TrgCutsTable.h TrgCutsTable.cc
    TrgSegmLenTable.h TrgSegmLenTable.cc
    VecUnsignedIntSortCriterion.h
    WbaIncrPhraseModel.h WbaIncrPhraseModel.cc
)

target_include_directories(phrase_models PUBLIC
    "${PROJECT_SOURCE_DIR}/include"
    "${PROJECT_SOURCE_DIR}/src/nlp_common"
    "${PROJECT_SOURCE_DIR}/src/sw_models"
)

target_compile_definitions(phrase_models PUBLIC
    THOT_USE_HAT_TRIE_PHRASE_TABLE
)
