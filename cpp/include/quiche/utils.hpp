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
 * @file utils.hpp
 * @brief Helpers and utilities for QUICHE.
 * @author Vasco Ferreira
 */

#ifndef UTILS_HPP
#define UTILS_HPP

#include <sstream>
#include <string>
#include <vector>

template <typename T>
std::string formatVector(const T &value) {
    std::ostringstream oss;
    oss << value;
    return oss.str();
}

template <typename T>
std::string formatVector(const std::vector<T> &vec) {
    std::string str;
    str += "[";
    for (size_t i = 0; i < vec.size(); i++) {
        str += formatVector(vec[i]);
        if (i < vec.size() - 1) {
            str += ", ";
        }
    }
    str += "]";
    return str;
}

#endif // UTILS_HPP
