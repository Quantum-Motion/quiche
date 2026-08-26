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

"""Common interface for objects dispatching to the Qualtran and QuEST backends."""

from abc import ABC, abstractmethod

from qualtran import Bloq

from quiche.quest import QuestRoutine


class Spec(ABC):
    """Interface for objects that lower to a Qualtran Bloq or QuEST routine."""

    @abstractmethod
    def to_qualtran(self) -> Bloq:
        """Build the Qualtran Bloq for this spec."""

    @abstractmethod
    def to_quest(self) -> QuestRoutine:
        """Build the QuEST routine for this spec."""
