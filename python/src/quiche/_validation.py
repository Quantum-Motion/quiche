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


"""Input validation shared by the algorithm method objects."""


def require_exactly_one(**options: object) -> None:
    """Raise unless exactly one of the keyword arguments is not `None`."""
    given = [name for name, value in options.items() if value is not None]
    if len(given) != 1:
        names = " or ".join(f"`{name}`" for name in options)
        msg = f"Exactly one of {names} must be given, got {len(given)}."
        raise ValueError(msg)
