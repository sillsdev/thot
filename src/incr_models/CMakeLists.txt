add_library(incr_models STATIC
    _incrEncCondProbModel.h
    _incrJelMerNgramLM.h
    _incrNgramLM.h
    BaseIncrCondProbModel.h
    BaseIncrCondProbTable.h
    BaseIncrEncCondProbModel.h
    BaseIncrEncoder.h
    BaseWordPenaltyModel.h
    im_pair.h
    IncrCondProbTable.h
    IncrEncoder.h
    IncrJelMerNgramLM.h IncrJelMerNgramLM.cc
    IncrNgramLM.h IncrNgramLM.cc
    lm_ienc.h lm_ienc.cc
    vecx_x_incr_cptable.h
    vecx_x_incr_ecpm.h
    vecx_x_incr_enc.h
    WordPenaltyModel.h WordPenaltyModel.cc
    WordPredictor.h WordPredictor.cc
)

target_include_directories(incr_models PUBLIC
    "${PROJECT_SOURCE_DIR}/src/nlp_common"
    "${PROJECT_SOURCE_DIR}/src/downhill_simplex"
)
