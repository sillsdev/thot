add_library(thot SHARED
    thot.h thot.cc
)

target_link_libraries(thot PUBLIC
    thot_lib
)