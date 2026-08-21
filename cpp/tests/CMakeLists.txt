add_executable(tests main.cpp)
target_link_libraries(tests PRIVATE quiche::quiche Catch2::Catch2)

add_subdirectory(unit)
add_subdirectory(integration)

catch_discover_tests(tests)
