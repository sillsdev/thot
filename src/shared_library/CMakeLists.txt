add_library(thot SHARED
    thot.h thot.cc
)

target_include_directories(thot PRIVATE
    "${PROJECT_SOURCE_DIR}/src/downhill_simplex"
    "${PROJECT_SOURCE_DIR}/src/error_correction"
    "${PROJECT_SOURCE_DIR}/src/incr_models"
    "${PROJECT_SOURCE_DIR}/src/nlp_common"
    "${PROJECT_SOURCE_DIR}/src/phrase_models"
    "${PROJECT_SOURCE_DIR}/src/stack_dec"
    "${PROJECT_SOURCE_DIR}/src/sw_models"
)

target_link_libraries(thot PRIVATE
    stack_dec
    phrase_models
    error_correction
    incr_models
    sw_models
    downhill_simplex
    nlp_common
    OpenMP::OpenMP_CXX
    ${GMP_LIBRARIES}
)

set_target_properties(thot PROPERTIES DEFINE_SYMBOL "THOT_EXPORTING")
if(WIN32)
    if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
        set_target_properties(thot PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/win-x64")
    else()
        set_target_properties(thot PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/win-x86")
    endif()
elseif(APPLE)
    set_target_properties(thot PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/osx-x64")
else()
    set_target_properties(thot PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/linux-x64")
endif()
