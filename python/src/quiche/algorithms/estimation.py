# Copyright 2026 Quantum Motion Technologies Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Quantum phase estimation algorithms."""

from pydantic.dataclasses import dataclass

type EstimationAlgorithm = TextbookQPE | IterativeQPE | KitaevQPE | NaiveQPE


@dataclass(frozen=True)
class TextbookQPE:
    pass


@dataclass(frozen=True)
class IterativeQPE:
    pass


@dataclass(frozen=True)
class KitaevQPE:
    pass


@dataclass(frozen=True)
class NaiveQPE:
    pass
