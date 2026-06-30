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

#include "quiche/mappings.hpp"

#include <stdexcept>

#include <quest/src/core/bitwise.hpp>

// Note qindex = long long int

namespace {

// binary string with all ones in the lowest n bits
inline qindex getLSBMask(int n) { return (1ULL << n) - 1; }

// Fenwick Tree basic operations
inline qindex getParentNode(qindex i) { return i | (i + 1); }

inline qindex getNodeRangeStart(qindex i) { return i & (i + 1); }

inline qindex getNodeRangeEnd(qindex i) { return i; }

std::vector<qindex> getUpdateSet(int index, int size) {

    // U(i) = qubits that are dependent on the occupation of orbital i
    //      = set of all ancestors of node i in the Fenwick tree

    // NB: depending on the convention i is or isn't included in U(i).
    // We use the latter convention and DON'T include i in U(i).

    std::vector<qindex> set;

    // NB: by convention we DON'T include i in U(i)
    qindex currentNode = getParentNode(index);
    while (currentNode < size) {
        set.push_back(currentNode);
        currentNode = getParentNode(currentNode);
    }

    return set;
}

std::vector<qindex> getFlipSet(int index) {

    // F(i) = qubits that determine whether orbitzal i and qubit i have the same or flipped parity
    //      = set of children of node i in the Fenwick tree

    std::vector<qindex> set;

    int start = getNodeRangeStart(index);
    int size = getNodeRangeEnd(index) - getNodeRangeStart(index) + 1; // = 2^k

    while (size > 1) {
        size /= 2;
        start += size;
        set.push_back(start - 1);
    }

    return set;
};

std::vector<qindex> getParitySet(int index) {

    // P(i) = qubits that determine the parity of the set of orbitals up to, but excluding i
    //      = minimal set of nodes covering the range [0, i)

    std::vector<qindex> set;

    qindex j = index - 1;
    while (j >= 0) {
        set.push_back(j);
        j = getNodeRangeStart(j) - 1;
    }

    // note that indices will be in decreasing size
    return set;
}

void validateOccupationVector(const std::vector<int> &occupation) {
    bool isBinary = std::all_of(occupation.begin(), occupation.end(), [](int x) { return x == 0 || x == 1; });
    if (!isBinary) {
        throw std::invalid_argument("Occupation vector entries must be 0 or 1.");
    }
}

} // namespace

namespace mappings {

std::vector<int> getQubitBasisStateJW(std::vector<int> occupation) {
    validateOccupationVector(occupation);
    return occupation;
}

std::vector<int> getQubitBasisStateBK(const std::vector<int> &occupation) {
    validateOccupationVector(occupation);
    std::vector<int> out = occupation;

    for (int i = 0; i < occupation.size(); i++) {
        for (int j : getUpdateSet(i, occupation.size())) {
            out[j] ^= occupation[i];
        }
    }

    return out;
}

std::vector<int> getQubitBasisStateParity(std::vector<int> occupation) {
    validateOccupationVector(occupation);

    std::partial_sum(occupation.begin(), occupation.end(), occupation.begin(),
                     [](int x, int y) { return (x + y) % 2; });

    return occupation;
}

qindex getHartreeFockStateJW(int numElectrons) {
    // Jordan-Wigner = occupation basis string
    return getLSBMask(numElectrons);
}

qindex getHartreeFockStateBK(int numElectrons, int numQubits) {
    qindex bits = getLSBMask(numElectrons);

    for (int i = 0; i < numElectrons; i++)
        for (qindex j : getUpdateSet(i, numQubits))
            bits = flipBit(bits, j);

    return bits;
}

qindex getHartreeFockStateParity(int numElectrons, int numQubits) {
    qindex bits = getLSBMask(numElectrons);

    // calculate prefix sum
    bits ^= (bits << 1);
    bits ^= (bits << 2);
    bits ^= (bits << 4);
    bits ^= (bits << 8);
    bits ^= (bits << 16);
    bits ^= (bits << 32);

    // truncate bit string to number of qubits
    return bits & getLSBMask(numQubits);
}

} // namespace mappings
