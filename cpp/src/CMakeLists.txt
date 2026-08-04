add_library(
    quiche
    SHARED
    mappings.cpp
    qdrift.cpp
    qpe.cpp
    qubitisation.cpp
    quest-patches.cpp
    utils.cpp
)
add_library(quiche::quiche ALIAS quiche)

target_include_directories(quiche PUBLIC ${PROJECT_SOURCE_DIR}/cpp/include)
target_link_libraries(quiche PUBLIC QuEST::QuEST)

# --------------------------------------------------------------------------------

option(QUICHE_BUILD_HAMLIB "Build the Hamlib parsing tools." OFF)
message(STATUS "Build QUICHE Hamlib tools: ${QUICHE_BUILD_HAMLIB}")


if(QUICHE_BUILD_HAMLIB)
    find_package(HDF5 COMPONENTS CXX)
    if (NOT HDF5_FOUND)
        message(STATUS "HDF5 not found, fetching from GitHub.")
        FetchContent_Declare(
            HDF5
            GIT_REPOSITORY https://github.com/HDFGroup/hdf5
            GIT_TAG 2.0.0
            GIT_PROGRESS TRUE
        )
        FetchContent_MakeAvailable(HDF5)
    endif()
    target_sources(quiche PRIVATE hamlib.cpp)
    target_link_libraries(quiche PUBLIC HDF5::HDF5)
endif()
