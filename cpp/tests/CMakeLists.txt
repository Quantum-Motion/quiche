FetchContent_Declare(
    Catch2
    URL https://github.com/catchorg/Catch2/archive/refs/tags/v3.13.0.tar.gz
    URL_HASH SHA256=650795f6501af514f806e78c554729847b98db6935e69076f36bb03ed2e985ef
    FIND_PACKAGE_ARGS 3.13
)
FetchContent_MakeAvailable(Catch2)

if(catch2_SOURCE_DIR)
    list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras")
endif()

# --------------------------------------------------------------------------------

add_executable(tests main.cpp)
target_link_libraries(tests PRIVATE quiche::quiche Catch2::Catch2)

add_subdirectory(unit)
add_subdirectory(integration)

include(CTest)
include(Catch)
catch_discover_tests(tests)
