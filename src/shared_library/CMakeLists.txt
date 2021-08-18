add_library(thot SHARED
    thot.h thot.cc
)

target_link_libraries(thot PUBLIC
    thot_lib
)

if(APPLE)
    add_custom_command(
        TARGET thot POST_BUILD
        COMMAND install_name_tool -add_rpath /opt/local/lib/libomp $<TARGET_FILE:thot>
        COMMAND install_name_tool -add_rpath /usr/local/opt/libomp/lib $<TARGET_FILE:thot>
        COMMAND install_name_tool -change /usr/local/opt/libomp/lib/libomp.dylib @rpath/libomp.dylib $<TARGET_FILE:thot>
        VERBATIM
    )
endif()