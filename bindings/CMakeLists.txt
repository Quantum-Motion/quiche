find_package(Python 3.12 COMPONENTS Interpreter Development.Module Development.SABIModule REQUIRED)

# Use git source to preserve necessary submodules
FetchContent_Declare(
    nanobind
    GIT_REPOSITORY https://github.com/wjakob/nanobind
    GIT_TAG 11087dd7707c80ef85802c661ad459ace5bef671    # v2.10.0
    GIT_PROGRESS TRUE
    FIND_PACKAGE_ARGS 2.10
)
FetchContent_MakeAvailable(nanobind)

# --------------------------------------------------------------------------------

if(APPLE)
    set(RPATH_ORIGIN "@loader_path")
else()
    set(RPATH_ORIGIN "$ORIGIN")
endif()

# --------------------------------------------------------------------------------

nanobind_add_module(
    bindings
    NOMINSIZE
    STABLE_ABI
    bindings.cpp
    quiche.cpp
    quest.cpp
)

target_link_libraries(bindings PRIVATE quiche)

set_target_properties(bindings PROPERTIES INSTALL_RPATH "${RPATH_ORIGIN}/libs")
set_target_properties(quiche QuEST PROPERTIES INSTALL_RPATH "${RPATH_ORIGIN}")

# Note: The Python build backend will set CMAKE_INSTALL_PREFIX to the temporary wheel
# staging directory. Therefore, these paths should be set relative to the site-packages
# root.
install(TARGETS QuEST quiche
    LIBRARY DESTINATION quiche/libs
    RUNTIME DESTINATION quiche/libs
)
install(TARGETS bindings LIBRARY DESTINATION quiche)

set(PYTHON_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../python/src/quiche")
nanobind_add_stub(
    bindings_stub
    INSTALL_TIME
    MODULE quiche.bindings
    PYTHON_PATH $<TARGET_FILE_DIR:bindings>
    RECURSIVE
    OUTPUT_PATH "${PYTHON_SOURCE_DIR}"
    OUTPUT
        bindings/__init__.pyi
        bindings/quest_bindings.pyi
        bindings/quiche_bindings.pyi
)
