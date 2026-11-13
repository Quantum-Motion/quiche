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

#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <quest.h>

#include "resources.hpp"

namespace bindings::resources {

// Define unwrapping for non-resource type (move)
template <class T, class = void>
struct checked_arg {
    using Wrapped = T;

    static T &&unwrap(Wrapped &v) noexcept { return std::move(v); }
};

// Define unwrapping for resource type
template <class T>
struct checked_arg<T, std::enable_if_t<is_quest_resource_v<T>>> {
    using Wrapped = CheckedResource<T> &;

    static T unwrap(Wrapped handle) { return handle.get(); }
};

// Define unwrapping for vector of resource type objects
template <class T>
struct checked_arg<std::vector<T>, std::enable_if_t<is_quest_resource_v<T>>> {
    using Wrapped = std::vector<CheckedResource<T> *>;

    static std::vector<T> unwrap(Wrapped &handles) {
        std::vector<T> out;
        out.reserve(handles.size());
        for (auto *h : handles) {
            if (!h)
                throw std::runtime_error("None where a QuEST resource was expected.");
            out.push_back(h->get());
        }
        return out;
    }
};

template <class Ret, class... Args, class Func>
auto checkedImpl(Func fn) {
    return [fn](typename checked_arg<Args>::Wrapped... args) {
        if constexpr (is_quest_resource_v<Ret>)
            return CheckedResource<Ret>(fn(checked_arg<Args>::unwrap(args)...));
        else
            return fn(checked_arg<Args>::unwrap(args)...);
    };
}

// Lambdas: call using lambda object (to keep captured values), use &operator() for type deduction
template <class Func, class Class, class Ret, class... Args>
auto checkedLambda(Func fn, Ret (Class::*)(Args...) const) {
    return checkedImpl<Ret, Args...>(fn);
}

// Lambdas: use &operator() for type deduction
template <class Fn, class = std::enable_if_t<std::is_class_v<Fn>>>
auto checked(Fn fn) {
    return checkedLambda(fn, &Fn::operator());
}

// Functions: deduce types from function pointer
template <class Ret, class... Args>
auto checked(Ret (*fn)(Args...)) {
    return checkedImpl<Ret, Args...>(fn);
}

} // namespace bindings::resources
