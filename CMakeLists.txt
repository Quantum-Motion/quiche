cmake_minimum_required(VERSION 3.26)

project(QuicheProject
    VERSION 0.0.1
    DESCRIPTION "QUantum Integrated CHEmistry toolkit"
    LANGUAGES CXX C # QUICHE itself only uses C++ but QuEST requires C
)

include(FetchContent)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

if(PROJECT_IS_TOP_LEVEL AND NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

if(PROJECT_IS_TOP_LEVEL)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

# --------------------------------------------------------------------------------
# Quiche config

message("
    ==================
    QUICHE config
    ==================
    ")

option(QUICHE_BUILD_EXAMPLES "Build the QUICHE example scripts." OFF)
message(STATUS "Build QUICHE examples: ${QUICHE_BUILD_EXAMPLES}")

option(QUICHE_BUILD_TESTS "Build the QUICHE test suite." OFF)
message(STATUS "Build QUICHE tests: ${QUICHE_BUILD_TESTS}")

option(QUICHE_BUILD_BINDINGS "Build the QUICHE Python bindings." OFF)
message(STATUS "Build QUICHE bindings: ${QUICHE_BUILD_BINDINGS}")

option(QUICHE_INSTALL "Install the QUICHE C++ library." ${PROJECT_IS_TOP_LEVEL})
message(STATUS "Install QUICHE C++ library: ${QUICHE_INSTALL}")

if(PROJECT_IS_TOP_LEVEL)
    option(BUILD_SHARED_LIBS "Build shared libraries." ON)
    message(STATUS "Build with shared libraries: ${BUILD_SHARED_LIBS}")
endif()

if(QUICHE_BUILD_BINDINGS AND NOT BUILD_SHARED_LIBS)
    message(FATAL_ERROR "QUICHE_BUILD_BINDINGS requires BUILD_SHARED_LIBS=ON.")
endif()

# --------------------------------------------------------------------------------
# Define dependencies

# QUICHE requires private QuEST headers, which are only exposed via the build
# interface and are not installed, so QuEST must come from FetchContent.
# Do not add FIND_PACKAGE_ARGS here.
FetchContent_Declare(
    QuEST
    URL https://github.com/QuEST-Kit/QuEST/archive/refs/tags/v4.2.0.tar.gz
    URL_HASH SHA256=2c812a7ec4d727e0947ffd0daf05452963c3f1c10e428c8bc30c35164921fcba
    OVERRIDE_FIND_PACKAGE
)

# Use git source to preserve required submodules
FetchContent_Declare(
    nanobind
    GIT_REPOSITORY https://github.com/wjakob/nanobind
    GIT_TAG 22acf11a2dac212e857f0d06ce80f808cac8fdad    # v2.11.0
    GIT_PROGRESS TRUE
    FIND_PACKAGE_ARGS 2.11
)

FetchContent_Declare(
    Catch2
    URL https://github.com/catchorg/Catch2/archive/refs/tags/v3.13.0.tar.gz
    URL_HASH SHA256=650795f6501af514f806e78c554729847b98db6935e69076f36bb03ed2e985ef
    FIND_PACKAGE_ARGS 3.13
)

# --------------------------------------------------------------------------------
# Fetch dependencies

message("
    ==================
    QuEST config
    ==================
    ")

FetchContent_MakeAvailable(QuEST)


if(QUICHE_BUILD_BINDINGS)
    find_package(Python 3.12
        COMPONENTS
        Interpreter
        Development.Module
        Development.SABIModule
        REQUIRED
    )
    FetchContent_MakeAvailable(nanobind)
endif()

if(QUICHE_BUILD_TESTS)
    FetchContent_MakeAvailable(Catch2)

    if(catch2_SOURCE_DIR)
        list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras")
    endif()

    include(CTest)
    include(Catch)
endif()

# --------------------------------------------------------------------------------

add_library(quiche)
add_library(quiche::quiche ALIAS quiche)

target_sources(quiche PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/cpp/include
    FILES
        ${PROJECT_SOURCE_DIR}/cpp/include/quiche/mappings.hpp
        ${PROJECT_SOURCE_DIR}/cpp/include/quiche/qdrift.hpp
        ${PROJECT_SOURCE_DIR}/cpp/include/quiche/qpe.hpp
        ${PROJECT_SOURCE_DIR}/cpp/include/quiche/qubitisation.hpp
        ${PROJECT_SOURCE_DIR}/cpp/include/quiche/quest-patches.hpp
        ${PROJECT_SOURCE_DIR}/cpp/include/quiche/utils.hpp
)

target_compile_features(quiche PUBLIC cxx_std_17)
target_link_libraries(quiche PUBLIC QuEST::QuEST)
set_target_properties(quiche PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

add_subdirectory(cpp)

if(QUICHE_BUILD_BINDINGS)
    add_subdirectory(bindings)
endif()

# --------------------------------------------------------------------------------
# Install and package setup

if(APPLE)
    set(RPATH_ORIGIN "@loader_path")
else()
    set(RPATH_ORIGIN "$ORIGIN")
endif()

set_target_properties(quiche PROPERTIES INSTALL_RPATH "${RPATH_ORIGIN}")

if(QUICHE_INSTALL)
    # Install library and public headers
    install(TARGETS quiche
        EXPORT quicheTargets
        FILE_SET HEADERS
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )

    # Install CMake target metadata
    install(EXPORT quicheTargets
        FILE quicheTargets.cmake
        NAMESPACE quiche::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/quiche
    )

    # Generate package config file
    configure_package_config_file(
        ${PROJECT_SOURCE_DIR}/cmake/quicheConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/quicheConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/quiche
    )

    # Generate package version file
    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/quicheConfigVersion.cmake
        COMPATIBILITY SameMinorVersion # update to SameMajorVersion post-1.0
    )

    # Install package files
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/quicheConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/quicheConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/quiche
    )

endif()

if(QUICHE_BUILD_BINDINGS)

    # For Linux and Mac can place in libs and locate via INSTALL_RPATH,
    # for Windows simply place beside bindings since don't have RPATH
    set_target_properties(bindings PROPERTIES INSTALL_RPATH "${RPATH_ORIGIN}/libs")
    install(TARGETS QuEST quiche
        LIBRARY DESTINATION quiche/libs COMPONENT quiche_wheel EXCLUDE_FROM_ALL
        RUNTIME DESTINATION quiche COMPONENT quiche_wheel EXCLUDE_FROM_ALL
    )

    install(TARGETS bindings
        LIBRARY DESTINATION quiche COMPONENT quiche_wheel EXCLUDE_FROM_ALL
    )

    # Generate stubs into the source tree for development builds
    nanobind_add_stub(bindings_stub
        INSTALL_TIME
        COMPONENT quiche_wheel
        MODULE quiche.bindings
        PYTHON_PATH $<TARGET_FILE_DIR:bindings>
        RECURSIVE
        OUTPUT_PATH "${PROJECT_SOURCE_DIR}/python/src/quiche"
        OUTPUT
            bindings/__init__.pyi
            bindings/quest_bindings.pyi
            bindings/quiche_bindings.pyi
    )

endif()
