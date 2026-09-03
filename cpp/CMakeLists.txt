add_subdirectory(src)

if(QUICHE_BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(QUICHE_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
