pybind11_add_module(thot_module
    module.cc
)

set_target_properties(thot_module PROPERTIES OUTPUT_NAME thot)

target_link_libraries(thot_module PUBLIC
    thot_lib
)
