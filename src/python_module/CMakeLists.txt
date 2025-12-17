if (CROSS_COMPILE) 
    if(PYTHON_IS_DEBUG)
        set_property(
            TARGET pybind11::pybind11
            APPEND
            PROPERTY INTERFACE_COMPILE_DEFINITIONS Py_DEBUG)
    endif()

    set_property(
        TARGET pybind11::pybind11
        APPEND
        PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${Python_ROOT_DIR}/include")
    if (MSVC)
        set_property(
            TARGET pybind11::module
            APPEND
            PROPERTY INTERFACE_LINK_LIBRARIES "${Python_ROOT_DIR}/libs/python${Python_VERSION}${Python_SUFFIX}.lib")
    endif()

    add_library(thot_module MODULE module.cc)

    target_link_libraries(thot_module PRIVATE pybind11::module pybind11::lto pybind11::windows_extras)
    set_target_properties(
        thot_module
        PROPERTIES PREFIX ""
                   DEBUG_POSTFIX "${Python_MODULE_DEBUG_POSTFIX}"
                   SUFFIX "${Python_MODULE_EXTENSION}")
    if(NOT MSVC AND NOT ${CMAKE_BUILD_TYPE} MATCHES Debug|RelWithDebInfo)
        # Strip unnecessary sections of the binary on Linux/macOS
        pybind11_strip(thot_module)
    endif()

    set_target_properties(thot_module PROPERTIES CXX_VISIBILITY_PRESET "hidden"
                                                 CUDA_VISIBILITY_PRESET "hidden")
else()
    pybind11_add_module(thot_module
        module.cc
    )
endif()

set_target_properties(thot_module PROPERTIES OUTPUT_NAME thot)

if (CMAKE_COMPILER_IS_GNUCC AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.5)
    set_target_properties(thot_module PROPERTIES INTERPROCEDURAL_OPTIMIZATION FALSE)
endif()

target_link_libraries(thot_module PUBLIC
    thot_lib
)

if(APPLE)
    add_custom_command(
        TARGET thot_module POST_BUILD
        COMMAND install_name_tool -add_rpath /opt/local/lib/libomp $<TARGET_FILE:thot_module>
        COMMAND install_name_tool -add_rpath /usr/local/opt/libomp/lib $<TARGET_FILE:thot_module>
        COMMAND install_name_tool -add_rpath /opt/homebrew/opt/libomp/lib $<TARGET_FILE:thot_module>
        COMMAND install_name_tool -change /usr/local/opt/libomp/lib/libomp.dylib @rpath/libomp.dylib $<TARGET_FILE:thot_module>
        COMMAND install_name_tool -change /opt/homebrew/opt/libomp/lib/libomp.dylib @rpath/libomp.dylib $<TARGET_FILE:thot_module>
        VERBATIM
    )
endif()
