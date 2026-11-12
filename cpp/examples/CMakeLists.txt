add_executable(hamiltonian-simulation hamiltonian-simulation.cpp)
target_link_libraries(hamiltonian-simulation PRIVATE quiche::quiche)

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

add_executable(qpe-single-ancilla qpe-single-ancilla.cpp)
target_link_libraries(qpe-single-ancilla PRIVATE quiche::quiche)

add_executable(qpe-textbook qpe-textbook.cpp)
target_link_libraries(qpe-textbook PRIVATE quiche::quiche)

add_executable(qpe-textbook-simulations qpe-textbook-simulations.cpp)
target_link_libraries(qpe-textbook-simulations PRIVATE quiche::quiche)
