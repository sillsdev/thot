add_library(sw_models STATIC
    _incrHmmAligModel.h _incrHmmAligModel.cc
    _incrHmmP0AligModel.h _incrHmmP0AligModel.cc
    _incrLexTable.h
    _incrSwAligModel.h _incrSwAligModel.cc
    _sentLengthModel.h _sentLengthModel.cc
    _swAligModel.h _swAligModel.cc
    anjiMatrix.h anjiMatrix.cc
    anjm1ip_anjiMatrix.h anjm1ip_anjiMatrix.cc
    ashPidxPairHashF.h
    aSource.h aSource.cc
    aSourceHashF.h
    aSourceHmm.h aSourceHmm.cc
    BaseSentenceHandler.h
    BaseSentLengthModel.h
    BaseStepwiseAligModel.h
    BaseSwAligModel.h BaseSwAligModel.cc
    BestLgProbForTrgWord.h
    CachedHmmAligLgProb.h CachedHmmAligLgProb.cc
    DoubleMatrix.h DoubleMatrix.cc
    FastAlignModel.h FastAlignModel.cc
    HmmAligInfo.h
    IncrHmmAligModel.h IncrHmmAligModel.cc
    IncrHmmAligTable.h IncrHmmAligTable.cc
    IncrHmmP0AligModel.h IncrHmmP0AligModel.cc
    IncrIbm1AligModel.h IncrIbm1AligModel.cc
    IncrIbm2AligModel.h IncrIbm2AligModel.cc
    IncrIbm2AligTable.h IncrIbm2AligTable.cc
    IncrLexTable.h IncrLexTable.cc
    LexAuxVar.h
    LightSentenceHandler.h LightSentenceHandler.cc
    SentPairCont.h
    SmoothedIncrIbm1AligModel.h SmoothedIncrIbm1AligModel.cc
    SmoothedIncrIbm2AligModel.h SmoothedIncrIbm2AligModel.cc
    SwDefs.h
    WeightedIncrNormSlm.h WeightedIncrNormSlm.cc
)

target_include_directories(sw_models PUBLIC 
    "${PROJECT_SOURCE_DIR}/src/nlp_common"
)

target_link_libraries(sw_models PRIVATE
    OpenMP::OpenMP_CXX
)
