pybind11_add_module(thot_module
    module.cc
)

set_target_properties(thot_module PROPERTIES OUTPUT_NAME thot)

target_link_libraries(thot_module PUBLIC
    thot_lib
)

if(APPLE)
    add_custom_command(
        TARGET thot_module POST_BUILD
        COMMAND install_name_tool -add_rpath /opt/local/lib/libomp $<TARGET_FILE:thot_module>
        COMMAND install_name_tool -add_rpath /usr/local/opt/libomp/lib $<TARGET_FILE:thot_module>
        COMMAND install_name_tool -change /usr/local/opt/libomp/lib/libomp.dylib @rpath/libomp.dylib $<TARGET_FILE:thot_module>
        VERBATIM
    )
endif()
