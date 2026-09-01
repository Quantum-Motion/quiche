add_executable(
    tests
    main.cpp
    mappings.cpp
    qpe.cpp
    qubitisation.cpp
)
target_link_libraries(tests PRIVATE quiche::quiche Catch2::Catch2)
catch_discover_tests(tests)
