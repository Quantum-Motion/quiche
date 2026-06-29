/*
 * Copyright 2026 Quantum Motion Technologies Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/** @file
 * Demo app using helpers to load Hamlib files.
 *
 * @author Vasco Ferreira
 */

#include <iostream>
#include <string>

#include "quiche/hamlib.hpp"
#include "quiche/utils.hpp"

const int verbosity = 2;
const int maxTerms = 2500;
const int maxQubits = 15;

int main() {
    std::string folderPath = std::string(getenv("HOME")) + "/Downloads/hamlib/chemistry/electronic/standard/";
    std::string fileName = "H2.hdf5";

    H5::H5File file(folderPath + fileName, H5F_ACC_RDONLY);

    H5::Group rootGroup = file.openGroup("/");
    std::vector<std::string> keys = hamlib::getKeys(rootGroup);
    std::cout << "Keys: " << formatVector(keys) << '\n';
    rootGroup.close();

    for (auto &key : keys) {
        std::cout << "------------------------------\n"
                  << "Dataset " << key << " from " << fileName << '\n';

        size_t found = key.find("molec");
        if (found != std::string::npos) {
            std::cout << "Skipping molecular dataset.\n";
            continue;
        }

        H5::DataSet dataSet = file.openDataSet(key);

        auto numTerms = hamlib::getNumericAttribute<size_t>(dataSet, "terms");
        auto numQubits = hamlib::getNumericAttribute<size_t>(dataSet, "nqubits");

        std::cout << "Number of terms: " << numTerms << '\n' << "Number of qubits: " << numQubits << '\n';

        if (numTerms > maxTerms) {
            std::cout << "Number of terms exceeds maximum, skipping Hamiltonian.\n";
            dataSet.close();
            continue;
        }

        if (numQubits > maxQubits) {
            std::cout << "Number of qubits exceeds maximum, skipping Hamiltonian.\n";
            dataSet.close();
            continue;
        }

        std::string string = hamlib::getHamiltonianString(dataSet);
        auto [coeffs, paulis, targets] = hamlib::getPauliStrings(string);

        if (coeffs.size() != numTerms || paulis.size() != numTerms || targets.size() != numTerms) {
            dataSet.close();
            file.close();
            throw std::runtime_error("Mismatch between number of parsed terms and "
                                     "expected number of terms.");
        }

        if (verbosity >= 2) {
            std::cout << "Original string: " << string << '\n';
        }

        if (verbosity >= 1) {
            std::cout << "Parsed coeffs: " << formatVector(coeffs) << '\n'
                      << "Parsed paulis: " << formatVector(paulis) << '\n'
                      << "Parsed targets: " << formatVector(targets) << '\n';
        }

        dataSet.close();
    }

    file.close();

    return 0;
}
