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

#include <nanobind/nanobind.h>

namespace nb = nanobind;

// Forward declaration for submodule constructors
void init_quest_bindings(nb::module_ &);
void init_quiche_bindings(nb::module_ &);

NB_MODULE(bindings, m) {
    init_quest_bindings(m);
    init_quiche_bindings(m);
}
