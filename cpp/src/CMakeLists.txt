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

set_target_properties(quiche PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
