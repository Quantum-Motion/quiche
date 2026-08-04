if(QUICHE_BUILD_HAMLIB)
    add_executable(hamlib hamlib.cpp)
    target_link_libraries(hamlib PRIVATE quiche::quiche)
endif()

add_executable(qdrift qdrift.cpp)
target_link_libraries(qdrift PRIVATE quiche::quiche)

add_executable(qpe-iterative qpe-iterative.cpp)
target_link_libraries(qpe-iterative PRIVATE quiche::quiche)

add_executable(qpe-kitaev qpe-kitaev.cpp)
target_link_libraries(qpe-kitaev PRIVATE quiche::quiche)

add_executable(qpe-naive qpe-naive.cpp)
target_link_libraries(qpe-naive PRIVATE quiche::quiche)

add_executable(qpe-qubitised-optimised qpe-qubitised-optimised.cpp)
target_link_libraries(qpe-qubitised-optimised PRIVATE quiche::quiche)

add_executable(qpe-qubitised qpe-qubitised.cpp)
target_link_libraries(qpe-qubitised PRIVATE quiche::quiche)

add_executable(qpe-textbook qpe-textbook.cpp)
target_link_libraries(qpe-textbook PRIVATE quiche::quiche)
