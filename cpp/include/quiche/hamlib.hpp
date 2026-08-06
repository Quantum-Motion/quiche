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

/**
 * @file hamlib.hpp
 * @brief Parsing logic for Hamlib HDF5 files.
 * @author Vasco Ferreira
 */

#ifndef HAMLIB_HPP
#define HAMLIB_HPP

#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <H5Cpp.h>

#include <quest.h>

namespace hamlib {

template <typename T>
T getNumericAttribute(const H5::DataSet &dataSet, const std::string &key) {
    static_assert(std::is_arithmetic_v<T>, "Error: getNumericAttribute requires a numeric type");
    H5::Attribute attribute = dataSet.openAttribute(key);
    H5::DataType dataType = attribute.getDataType();

    T result;
    attribute.read(dataType, &result);

    return result;
}

std::vector<std::string> getKeys(const H5::Group &group);

std::string getHamiltonianString(const H5::DataSet &dataSet);

std::pair<std::string, std::vector<int>> parseStringToPaulisAndTargets(const std::string &str);

double getRealCoeff(const std::string &str);

std::tuple<std::vector<double>, std::vector<std::string>, std::vector<std::vector<int>>>
getPauliStrings(const std::string &str);

PauliStrSum getHamiltonian(const std::string &filePath, const std::string &key);

} // namespace hamlib

#endif // HAMLIB_HPP
