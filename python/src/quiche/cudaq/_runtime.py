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

"""Deferred CUDA-Q imports and the load-order guard for the CUDA-Q backend."""

import sys
import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from types import ModuleType

# CUDA-Q exposes no stable public Python type for a compiled kernel object. This
# module never imports cudaq itself, so the alias is safe to import eagerly.
type CudaqKernel = Any

_LOAD_ORDER_WARNING = (
    "cudaq is being imported after quiche's compiled QuEST extension is already "
    "loaded. cudaq's MLIR bindings must be the first native extension loaded in "
    "the process, or their initialisation can break - import cudaq before "
    "importing quiche to avoid this."
)


def load_cudaq() -> "tuple[ModuleType, ModuleType]":
    """
    Import `cudaq` and `cudaq_algorithms` on first use.

    Both stay soft, optional dependencies of quiche: this is the only place the
    CUDA-Q backend imports them, and it is never reached at `import quiche` time
    (mirrors the lazy `import cudaq` inside `PauliSum.from_cudaq`).

    Warns, rather than failing opaquely inside CUDA-Q's MLIR initialisation, when
    the documented load-order collision with quiche's compiled extension is about
    to happen - see the comment in `python/tests/conftest.py`.
    """
    if "cudaq" not in sys.modules and "quiche.bindings" in sys.modules:
        warnings.warn(_LOAD_ORDER_WARNING, RuntimeWarning, stacklevel=2)

    try:
        import cudaq  # noqa: PLC0415
        import cudaq_algorithms  # noqa: PLC0415
    except ImportError as exc:
        msg = "The CUDA-Q backend requires the `cudaq` and `cudaq-algorithms` packages."
        raise ImportError(msg) from exc

    return cudaq, cudaq_algorithms
