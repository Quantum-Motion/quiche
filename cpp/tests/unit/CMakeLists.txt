target_sources(
    tests
    PRIVATE
    mappings.cpp
    qpe.cpp
    qubitisation.cpp
)

if(QUICHE_BUILD_HAMLIB)
    target_sources(tests PRIVATE hamlib.cpp)
endif()
