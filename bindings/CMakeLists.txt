find_package(Python 3.12 COMPONENTS Interpreter Development.Module REQUIRED)

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

set(BUILD_RPATH_USE_ORIGIN ON)
if(APPLE)
    set(RPATH_ORIGIN "@loader_path")
else()
    set(RPATH_ORIGIN "$ORIGIN")
endif()

set(RPATH_BINDINGS "${RPATH_ORIGIN}/libs")

# --------------------------------------------------------------------------------

nanobind_add_module(
    bindings
    bindings.cpp
    quiche.cpp
    quest.cpp
    NOMINSIZE
    NB_SHARED
    STABLE_ABI
)

target_link_libraries(bindings PRIVATE quiche)
set_target_properties(bindings PROPERTIES INSTALL_RPATH "${RPATH_BINDINGS}")

# Note: The Python build backend will set CMAKE_INSTALL_PREFIX to the temporary wheel
# staging directory. Therefore, these paths should be set relative to the site-packages
# root.
install(TARGETS nanobind QuEST quiche LIBRARY DESTINATION quiche/libs)
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
