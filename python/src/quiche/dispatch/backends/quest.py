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

"""Methods to generate the QuEST implementation of a QPESpec object."""

from __future__ import annotations

from functools import partial
from typing import Any, assert_never

from quiche import QPESpec, algorithms
from quiche.bindings import quiche_bindings as bindings
from quiche.dispatch.backends import UnsupportedAlgorithmError
from quiche.simulation import QuESTRoutine, QuregLayout, QuregOp
from quiche.simulation.utils import _with_quest_hamiltonian


def get_stateprep_op(spec: QPESpec) -> QuregOp:
    """Pattern matching and construction for QuEST state preparation."""
    match spec.stateprep:
        case algorithms.IdentityStateprep():
            return lambda _qureg: None
        case algorithms.BitstringStateprep(bitstring=bitstring):
            return partial(bindings.initClassicalState, state=bitstring)
        case _:
            assert_never(spec.stateprep)


def _get_simulation_kwargs(
    spec: QPESpec,
    layout: QuregLayout,
) -> dict[str, Any]:
    match spec.simulation:
        case algorithms.QDRIFT(num_samples=num_samples, seed=seed, time=time):
            return {"reps": num_samples, "time": time, "seed": seed}
        case algorithms.Trotter(time=time, order=order, num_steps=num_steps):
            return {"order": order, "reps": num_steps, "time": time}
        case algorithms.Qubitisation(prep=prep):
            match prep:
                case algorithms.ExactRotationTree():
                    return {"qubitisation_ancillas": layout.simulation}
                case algorithms.QROMRotationTree() | algorithms.AliasSampling():
                    raise UnsupportedAlgorithmError(spec, QuESTBackend)
                case _:
                    assert_never(prep)
        case _:
            assert_never(spec.simulation)


def _get_estimation_kwargs(
    spec: QPESpec,
    layout: QuregLayout,
) -> dict[str, Any]:
    match spec.estimation:
        case algorithms.KitaevQPE(num_rounds=num_rounds):
            return {"num_bits": num_rounds, "ancilla_index": layout.qpe[0]}
        case algorithms.NaiveQPE():
            return {"ancilla_index": layout.qpe[0]}
        case algorithms.TextbookQPE():
            # TODO(Vasco): standardise C++ naming and remove this match
            match spec.simulation:
                case algorithms.Qubitisation():
                    return {"qpe_ancillas": layout.qpe}
                case algorithms.Trotter() | algorithms.QDRIFT():
                    return {"ancillas": layout.qpe}
                case _:
                    assert_never(spec.simulation)
        case _:
            raise UnsupportedAlgorithmError(spec, QuESTBackend)


# fmt: off
_BINDING_MAP = {
    (algorithms.KitaevQPE, algorithms.QDRIFT): bindings.getPhaseKitaevQDRIFT,
    (algorithms.KitaevQPE, algorithms.Trotter): bindings.getPhaseKitaevTrotter,
    (algorithms.NaiveQPE, algorithms.QDRIFT): bindings.getPhaseNaiveQDRIFT,
    (algorithms.NaiveQPE, algorithms.Trotter): bindings.getPhaseNaiveTrotter,
    (algorithms.TextbookQPE, algorithms.QDRIFT): bindings.getPhaseTextbookQDRIFT,
    (algorithms.TextbookQPE, algorithms.Qubitisation): bindings.getPhaseTextbookQubitised,  # ruff: ignore[line-too-long]
    (algorithms.TextbookQPE, algorithms.Trotter): bindings.getPhaseTextbookTrotter,
}
# fmt: on


def get_estimation_op(spec: QPESpec, layout: QuregLayout) -> QuregOp:
    """Pattern matching and construction for QuEST phase estimation."""
    key = (type(spec.estimation), type(spec.simulation))

    binding = _BINDING_MAP.get(key)
    if binding is None:
        raise UnsupportedAlgorithmError(spec, QuESTBackend)

    return _with_quest_hamiltonian(
        binding,
        hamiltonian=spec.simulation.hamiltonian,
        **_get_simulation_kwargs(spec, layout),
        **_get_estimation_kwargs(spec, layout),
    )


class QuESTBackend:
    """QuEST backend generating a statevector simulation."""

    def generate(self, spec: QPESpec) -> QuESTRoutine:
        """Build the QuEST routine implementing the given QPESpec."""
        layout = QuregLayout(
            num_data=spec.simulation.hamiltonian.num_qubits,
            num_qpe=spec.estimation.num_ancillas,
            num_simulation=spec.simulation.num_ancillas,
        )

        return QuESTRoutine(
            ops=(get_stateprep_op(spec), get_estimation_op(spec, layout)),
            layout=layout,
        )
