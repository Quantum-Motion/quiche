nanobind_add_module(
    bindings
    NOMINSIZE
    STABLE_ABI
    bindings.cpp
    quiche.cpp
    quest.cpp
)
target_link_libraries(bindings PRIVATE quiche::quiche)
