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

#pragma once

#include <optional>
#include <stdexcept>
#include <utility>

#include <quest.h>

namespace bindings::resources {

template <class Q>
inline constexpr auto destructor = nullptr;
template <>
inline constexpr auto destructor<Qureg> = &destroyQureg;
template <>
inline constexpr auto destructor<PauliStrSum> = &destroyPauliStrSum;
template <>
inline constexpr auto destructor<KrausMap> = &destroyKrausMap;
template <>
inline constexpr auto destructor<SuperOp> = &destroySuperOp;
template <>
inline constexpr auto destructor<CompMatr> = &destroyCompMatr;
template <>
inline constexpr auto destructor<DiagMatr> = &destroyDiagMatr;
template <>
inline constexpr auto destructor<FullStateDiagMatr> = &destroyFullStateDiagMatr;

template <class Q>
inline constexpr bool is_quest_resource_v = destructor<Q> != nullptr;

template <class R>
class CheckedResource {
    std::optional<R> resource;

  public:
    explicit CheckedResource(R r) noexcept : resource(r) {}

    CheckedResource(CheckedResource &&other) noexcept : resource(std::exchange(other.resource, std::nullopt)) {}

    CheckedResource(const CheckedResource &) = delete;
    CheckedResource &operator=(const CheckedResource &) = delete;
    CheckedResource &operator=(CheckedResource &&) = delete;

    ~CheckedResource() { destroy(); }

    R get() const {
        if (!resource)
            throw std::runtime_error("Attempted to access a QuEST resource that has already been destroyed.");
        if (!isQuESTEnvInit())
            throw std::runtime_error("Attempted to access a QuEST resource after the QuEST environment was finalised.");
        return *resource;
    }

    void destroy() {
        if (!resource)
            return;
        R r = *std::exchange(resource, std::nullopt);
        if (isQuESTEnvInit())
            destructor<R>(r);
    }
};

} // namespace bindings::resources
