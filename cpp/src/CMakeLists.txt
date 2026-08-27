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

target_include_directories(quiche PUBLIC ${PROJECT_SOURCE_DIR}/cpp/include)
target_link_libraries(quiche PUBLIC QuEST)
