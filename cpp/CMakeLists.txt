add_subdirectory(src)

# --------------------------------------------------------------------------------

option(QUICHE_BUILD_TESTS "Build the test suite." OFF)
message(STATUS "Build QUICHE tests: ${QUICHE_BUILD_TESTS}")

if(QUICHE_BUILD_TESTS)
    add_subdirectory(tests)
endif()

# --------------------------------------------------------------------------------

option(QUICHE_BUILD_EXAMPLES "Build the example scripts." OFF)
message(STATUS "Build QUICHE examples: ${QUICHE_BUILD_EXAMPLES}")

if(QUICHE_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
