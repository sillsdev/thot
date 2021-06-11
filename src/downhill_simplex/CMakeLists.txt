add_library(downhill_simplex STATIC
    step_by_step_dhs.h step_by_step_dhs.c
)

target_include_directories(downhill_simplex PUBLIC
    "${PROJECT_SOURCE_DIR}/src/nlp_common"
)
