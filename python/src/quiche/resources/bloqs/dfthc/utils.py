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

"""Common utilities used for DFTHC resource implementation."""

from qualtran import Bloq, SoquetT


# Adapted from the (unexported) closure in Qualtran's LCUBlockEncoding decomposition.
def extract_soqs(bloq: Bloq, soqs: dict[str, SoquetT]) -> dict[str, SoquetT]:
    """Pop and return the entries of `soqs` that match `bloq`'s input registers."""
    return {reg.name: soqs.pop(reg.name) for reg in bloq.signature.lefts()}
