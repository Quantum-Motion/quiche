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

#include "quiche/hamlib.hpp"

#include <regex>

namespace hamlib {

std::vector<std::string> getKeys(const H5::Group &group) {
    auto numObjects = group.getNumObjs();

    std::vector<std::string> keys;
    keys.reserve(numObjects);

    for (auto i = 0; i < numObjects; i++) {
        keys.push_back(group.getObjnameByIdx(i));
    }

    return keys;
}

std::string getHamiltonianString(const H5::DataSet &dataSet) {
    H5::DataSpace dataSpace = dataSet.getSpace();
    H5::StrType strType = dataSet.getStrType();

    std::string data;
    dataSet.read(data, strType, dataSpace);

    return data;
}

std::pair<std::string, std::vector<int>> parseStringToPaulisAndTargets(const std::string &str) {
    std::string paulis;
    std::vector<int> targets;

    std::regex re(R"(([XYZ])(\d+))");
    for (std::sregex_iterator it(str.begin(), str.end(), re), end; it != end; it++) {
        auto pauli = it->str(1);
        auto target = std::stoi(it->str(2));

        paulis.append(pauli);
        targets.push_back(target);
    }

    if (paulis.size() == 0 && targets.size() == 0) {
        paulis = "I";
        targets = {0};
    }

    return {paulis, targets};
}

// This relies on the fact that Hamlib coefficients (seem to) come in two formats
// <real float>+0.0j and <real float> to do a quick-and-dirty parsing of the real
// part. This is quite brittle though, e.g. it would fail with floats in engineering
// notation with a positive exponent, which I haven't seen arise, but still not ideal.
// It should be replaced with a more robust regex approach.
double getRealCoeff(const std::string &str) {
    std::string realString = str;

    size_t pos = str.find('+');
    if (pos != std::string::npos) {
        realString = realString.substr(0, pos);
    }

    double real = std::stod(realString);
    return real;
}

std::tuple<std::vector<double>, std::vector<std::string>, std::vector<std::vector<int>>>
getPauliStrings(const std::string &str) {
    std::vector<double> coeffs;
    std::vector<std::string> paulis;
    std::vector<std::vector<int>> targets;

    std::regex re(R"(\(?(.*?)\)?\s+\[(.*?)\])");
    for (std::sregex_iterator it(str.begin(), str.end(), re), end; it != end; it++) {

        auto coeff = getRealCoeff(it->str(1));
        auto [pauli, target] = parseStringToPaulisAndTargets(it->str(2));

        coeffs.push_back(coeff);
        paulis.push_back(pauli);
        targets.push_back(target);
    }

    return {coeffs, paulis, targets};
}

PauliStrSum getHamiltonian(const std::string &filePath, const std::string &key) {
    H5::H5File file(filePath, H5F_ACC_RDONLY);
    H5::DataSet dataSet = file.openDataSet(key);

    auto numTerms = getNumericAttribute<qindex>(dataSet, "terms");
    std::string string = getHamiltonianString(dataSet);
    auto [coeffs, paulis, targets] = getPauliStrings(string);

    if (coeffs.size() != numTerms || paulis.size() != numTerms || targets.size() != numTerms) {
        throw std::runtime_error("Mismatch between number of parsed terms and expected number of terms.");
    }

    std::vector<PauliStr> strings(numTerms);
    std::vector<qcomp> coeffsCast(numTerms);

    for (size_t i = 0; i < numTerms; i++) {
        strings[i] = getPauliStr(paulis[i], targets[i]);
        coeffsCast[i] = qcomp(coeffs[i], 0);
    }

    return createPauliStrSum(strings, coeffsCast);
}

} // namespace hamlib
