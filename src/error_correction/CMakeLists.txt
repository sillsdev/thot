add_library(error_correction STATIC
    _editDist.h
    _editDistBasedEcm.h _editDistBasedEcm.cc
    BaseEcmForWg.h
    BaseEcModelForNbUcat.h
    BaseEditDist.h
    BaseErrorCorrectionModel.h BaseErrorCorrectionModel.cc
    BaseWgProcessorForAnlp.h
    EditDistForStr.h EditDistForStr.cc
    EditDistForVec.h
    EditDistForVecString.h EditDistForVecString.cc
    HypStateIndex.h
    NbestCorrections.h
    NbSearchHighLevelHyp.h
    NbSearchHyp.h
    NbSearchStack.h	NbSearchStack.cc
    NonPbEcModelForNbUcat.h NonPbEcModelForNbUcat.cc
    PfsmEcm.h PfsmEcm.cc
    PfsmEcmForWg.h PfsmEcmForWg.cc
    PfsmEcmForWgEsi.h
    PrefAlignInfo.h
    RejectedWordsSet.h
    WgHandler.h WgHandler.cc
    WgProcessorForAnlp.h
    WordAndCharLevelOps.h
    WordGraph.h WordGraph.cc
    WordGraphArc.h
    WordGraphArcId.h
    WordGraphStateData.h
)

target_include_directories(error_correction PUBLIC
    "${PROJECT_SOURCE_DIR}/src/nlp_common"
)
