cmake_minimum_required(VERSION 3.24)

project(QuicheProject
    VERSION 0.0.1
    DESCRIPTION "QUantum Integrated CHEmistry toolkit")

# Enforce C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# --------------------------------------------------------------------------------

message("
    =================
    QUEST config
    ==================
    ")

# NOTE: quiche requires private QuEST headers from quest/src/core/. These are only
# exposed via the build interface and are not installed, so QuEST must come from
# FetchContent -- do not add FIND_PACKAGE_ARGS here.
FetchContent_Declare(
    QuEST
    URL https://github.com/QuEST-Kit/QuEST/archive/refs/tags/v4.2.0.tar.gz
    URL_HASH SHA256=2c812a7ec4d727e0947ffd0daf05452963c3f1c10e428c8bc30c35164921fcba
    OVERRIDE_FIND_PACKAGE
)
FetchContent_MakeAvailable(QuEST)

# --------------------------------------------------------------------------------

message("
    =================
    QUICHE config
    ==================
    ")

# --------------------------------------------------------------------------------

add_subdirectory(cpp)

# --------------------------------------------------------------------------------

option(QUICHE_BUILD_BINDINGS "Build the python bindings." OFF)
message(STATUS "Build QUICHE bindings: ${QUICHE_BUILD_BINDINGS}")

if(QUICHE_BUILD_BINDINGS)
    add_subdirectory(bindings)
endif()
