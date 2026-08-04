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
target_compile_features(quiche PUBLIC cxx_std_17)
target_link_libraries(quiche PUBLIC QuEST::QuEST)

# --------------------------------------------------------------------------------

option(QUICHE_BUILD_HAMLIB "Build the Hamlib parsing tools." OFF)
message(STATUS "Build QUICHE Hamlib tools: ${QUICHE_BUILD_HAMLIB}")

if(QUICHE_BUILD_HAMLIB)
    FetchContent_Declare(
        HDF5
        URL https://github.com/HDFGroup/hdf5/releases/download/2.0.0/hdf5-2.0.0.tar.gz
        URL_HASH SHA256=f4c2edc5668fb846627182708dbe1e16c60c467e63177a75b0b9f12c19d7efed
        FIND_PACKAGE_ARGS 2.0.0 COMPONENTS CXX
    )

    # Only consulted during source build. Enable bare minimum HDF5 C++ library components.
    set(HDF5_BUILD_CPP_LIB ON)
    set(HDF5_ENABLE_ZLIB_SUPPORT ON)
    set(BUILD_STATIC_LIBS OFF)
    set(HDF5_BUILD_HL_LIB OFF)
    set(HDF5_BUILD_TOOLS OFF)
    set(HDF5_BUILD_UTILS OFF)
    set(HDF5_BUILD_EXAMPLES OFF)
    set(BUILD_TESTING OFF)

    FetchContent_MakeAvailable(HDF5)

    add_library(quiche_hdf5 INTERFACE)
    add_library(quiche::hdf5 ALIAS quiche_hdf5)

    # Try to match against different possible HDF5 targets
    if(TARGET HDF5::HDF5)
        target_link_libraries(quiche_hdf5 INTERFACE HDF5::HDF5)
    elseif(TARGET hdf5::hdf5_cpp)
        target_link_libraries(quiche_hdf5 INTERFACE hdf5::hdf5_cpp)
    elseif(TARGET hdf5_cpp-shared)
        target_link_libraries(quiche_hdf5 INTERFACE hdf5_cpp-shared)
    else()
        message(FATAL_ERROR "HDF5 found but no known C++ target")
    endif()

    target_sources(quiche PRIVATE hamlib.cpp)
    target_link_libraries(quiche PUBLIC quiche::hdf5)

endif()
