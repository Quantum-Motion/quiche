find_package(Catch2 3 QUIET)
if(NOT Catch2_FOUND)
    message(STATUS "Catch2 not found, fetching from GitHub.")
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.13.0
        GIT_PROGRESS TRUE
    )
    FetchContent_MakeAvailable(Catch2)
    list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
endif ()

# --------------------------------------------------------------------------------

add_executable(
    tests
    main.cpp
    mappings.cpp
    qpe.cpp
    qubitisation.cpp
)
target_link_libraries(tests PRIVATE quiche Catch2::Catch2)

include(CTest)
include(Catch)
catch_discover_tests(tests)
