cmake_minimum_required(VERSION 3.21)

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

find_package(QuEST QUIET)
if(NOT QuEST_FOUND)
    message(STATUS "QuEST not found, fetching from GitHub.")
    FetchContent_Declare(
        QuEST
        GIT_REPOSITORY https://github.com/QuEST-Kit/QuEST
        GIT_TAG v4.2.0
        GIT_PROGRESS TRUE
    )
    FetchContent_MakeAvailable(QuEST)
endif()

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
