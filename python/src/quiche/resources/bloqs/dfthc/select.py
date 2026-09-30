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

"""SELECT routines for the DFTHC Hamiltonian block encoding."""

import attrs
from qualtran import (
    Bloq,
    BloqBuilder,
    CtrlSpec,
    QAny,
    QBit,
    QFxp,
    Register,
    Signature,
    SoquetT,
)
from qualtran.bloqs.basic_gates import CNOT, CZ, CSwap, Hadamard, Toffoli, XGate, ZGate
from qualtran.bloqs.bookkeeping import Partition
from qualtran.bloqs.chemistry.quad_fermion.givens_bloq import (
    RealGivensRotationByPhaseGradient,
)
from qualtran.resource_counting import (
    BloqCountDictT,
    SympySymbolAllocator,
)

from quiche.resources.bloqs.dfthc.utils import extract_soqs


@attrs.frozen
class GivensLadder(Bloq):
    """
    Ladder of nearest-neighbour real Givens rotations on the system register.

    Parameters
    ----------
    num_orbitals : int
        Number of spatial orbitals, :math:`N` in ref. [1].
    num_bits_phase_grad : int
        Total number of bits for persistent phase gradient register, :math:`b_{rot}` in
        ref. [1].

    Registers
    ---------
    rotations : QAny
        Givens rotation angles for the basis rotation.
    phase_grad : QAny
        Phase gradient register used throughout the circuit.
    system : QAny
        System register of the selected spin sector.

    References
    ----------
        [1] G. H. Low et al., "Fast Quantum Simulation of Electronic Structure by
            Spectral Amplification", Phys. Rev. X, vol. 15, no. 4, p. 041016, Oct. 2025.

    """

    num_orbitals: int

    num_bits_phase_grad: int

    @property
    def num_rotations(self) -> int:
        """Givens ladder length for one basis rotation, :math:`N - 1`."""
        return self.num_orbitals - 1

    @property
    def rotations_size(self) -> int:
        """Rotation register size, :math:`(N - 1) b_{rot}`."""
        return self.num_rotations * self.num_bits_phase_grad

    @property
    def signature(self) -> Signature:
        """Define input and/or output registers of the bloq."""
        return Signature(
            [
                Register("rotations", QAny(self.rotations_size)),
                Register("phase_grad", QAny(self.num_bits_phase_grad)),
                Register("system", QAny(self.num_orbitals)),
            ]
        )

    @property
    def partition(self) -> Partition:
        """Get the partition of `rotations` into the individual angles."""
        theta = QFxp(self.num_bits_phase_grad, self.num_bits_phase_grad)
        registers = (Register("theta", theta, (self.num_rotations,)),)
        return Partition(self.rotations_size, registers)

    @property
    def givens_rot(self) -> RealGivensRotationByPhaseGradient:
        """Get the Givens rotation applied at each step of the ladder."""
        return RealGivensRotationByPhaseGradient(
            phasegrad_bitsize=self.num_bits_phase_grad
        )

    def build_call_graph(self, ssa: SympySymbolAllocator) -> BloqCountDictT:  # noqa: ARG002
        """Build call graph for GivensLadder."""
        return {self.givens_rot: self.num_rotations}

    def build_composite_bloq(
        self,
        bb: BloqBuilder,
        **soqs: SoquetT,
    ) -> dict[str, SoquetT]:
        """Implement bloq decomposition into sub-bloqs."""
        thetas = bb.add(self.partition, x=soqs["rotations"])
        qs = bb.split(soqs["system"])

        for j in reversed(range(self.num_rotations)):
            qs[j], qs[j + 1], thetas[j], soqs["phase_grad"] = bb.add(
                self.givens_rot,
                target_i=qs[j],
                target_j=qs[j + 1],
                rom_data=thetas[j],
                phase_gradient=soqs["phase_grad"],
            )

        soqs["rotations"] = bb.add(self.partition.adjoint(), theta=thetas)
        soqs["system"] = bb.join(qs)

        return soqs


@attrs.frozen
class SpinSelectSwap(Bloq):
    """
    Swap the spin-down and spin-up system registers, controlled on the selected spin.

    Parameters
    ----------
    num_orbitals : int
        Number of spatial orbitals, :math:`N` in ref. [1].

    Registers
    ---------
    G_0 : QBit
        Qubit selecting between SF and :math:`D_1/Q_1`.

    spin_0 : QBit
        Spin control for SF terms.
    spin_1 : QBit
        Spin control for :math:`D_1/Q_1` terms.

    spin : QBit
        Selected spin, copied from spin_0 (SF) or spin_1 (:math:`D_1/Q_1`).

    system_down : QAny
        Spin-down system register.
    system_up : QAny
        Spin-up system register.

    References
    ----------
        [1] G. H. Low et al., "Fast Quantum Simulation of Electronic Structure by
            Spectral Amplification", Phys. Rev. X, vol. 15, no. 4, p. 041016, Oct. 2025.

    """

    num_orbitals: int

    @property
    def signature(self) -> Signature:
        """Define input and/or output registers of the bloq."""
        return Signature(
            [
                Register("G_0", QBit()),
                Register("spin_0", QBit()),
                Register("spin_1", QBit()),
                Register("spin", QBit()),
                Register("system_down", QAny(self.num_orbitals)),
                Register("system_up", QAny(self.num_orbitals)),
            ]
        )

    @property
    def toffoli_01(self) -> Bloq:
        """Get the Toffoli copying `spin_0` to `spin` if `G_0` = 0 (SF)."""
        return XGate().controlled(CtrlSpec(cvs=(0, 1)))

    @property
    def toffoli_11(self) -> Bloq:
        """Get the Toffoli copying `spin_1` to `spin` if `G_0` = 1 (:math:`D_1/Q_1`)."""
        return Toffoli()

    @property
    def cswap(self) -> CSwap:
        """Get the swap of the two spin sectors, controlled on the selected spin."""
        return CSwap(self.num_orbitals)

    def build_call_graph(self, ssa: SympySymbolAllocator) -> BloqCountDictT:  # noqa: ARG002
        """Build call graph for SpinSelectSwap."""
        return {self.toffoli_01: 1, self.toffoli_11: 1, self.cswap: 1}

    def build_composite_bloq(
        self,
        bb: BloqBuilder,
        **soqs: SoquetT,
    ) -> dict[str, SoquetT]:
        """Implement bloq decomposition into sub-bloqs."""
        (soqs["G_0"], soqs["spin_0"]), soqs["spin"] = bb.add(
            self.toffoli_01,
            ctrl=[soqs["G_0"], soqs["spin_0"]],
            q=soqs["spin"],
        )

        (soqs["G_0"], soqs["spin_1"]), soqs["spin"] = bb.add(
            self.toffoli_11,
            ctrl=[soqs["G_0"], soqs["spin_1"]],
            target=soqs["spin"],
        )

        soqs["spin"], soqs["system_down"], soqs["system_up"] = bb.add(
            self.cswap,
            ctrl=soqs["spin"],
            x=soqs["system_down"],
            y=soqs["system_up"],
        )

        return soqs


@attrs.frozen
class MajoranaDFTHC(Bloq):
    r"""
    Majorana operator for the DFTHC block encoding.

    Registers
    ---------
    G_0 : QBit
        Qubit selecting between SF and :math:`D_1/Q_1`.
    G_1 : QBit
        Qubit selecting between :math:`D_1` and :math:`Q_1`.
    non_identity : QBit
        Non-identity flag qubit.
    s : QBit
        Superposition qubit selecting :math:`X` or :math:`iY` for :math:`D_1/Q_1`,
        :math:`\varsigma` in ref. [2].
    target : QBit
        Qubit to apply Majorana operator to.

    Notes
    -----
    Polarity of `G_0`: we use `G_0` = 0 for SF and `G_0` = 1 for :math:`D_1/Q_1`, as in
    ref. [1] fig. 2 and appendix B7. Fig. 4 appears to use the opposite polarity. It is
    not followed here, so that this bloq stays consistent with the others.

    References
    ----------
        [1] G. H. Low et al., "Fast Quantum Simulation of Electronic Structure by
            Spectral Amplification", Phys. Rev. X, vol. 15, no. 4, p. 041016, Oct. 2025.
        [2] G. H. Low et al., "A Denser Planar Surface Code", May 28, 2026, arXiv:
            2605.30455.

    """

    @property
    def signature(self) -> Signature:
        """Define input and/or output registers of the bloq."""
        return Signature(
            [
                Register("G_0", QBit()),
                Register("G_1", QBit()),
                Register("non_identity", QBit()),
                Register("s", QBit()),
                Register("target", QBit()),
            ]
        )

    @property
    def ccz_01(self) -> Bloq:
        """Get the CCZ on `target` for SF (`G_0` = 0) and `non_identity` = 1."""
        return ZGate().controlled(CtrlSpec(cvs=(0, 1)))

    @property
    def ccz_11(self) -> Bloq:
        """Get the CCZ on `target` for :math:`D_1/Q_1` (`G_0` = 1) and `s` = 1."""
        return ZGate().controlled(CtrlSpec(cvs=(1, 1)))

    def build_call_graph(self, ssa: SympySymbolAllocator) -> BloqCountDictT:  # noqa: ARG002
        """Build call graph for MajoranaDFTHC."""
        return {self.ccz_01: 1, CNOT(): 1, self.ccz_11: 1, CZ(): 1}

    def build_composite_bloq(
        self,
        bb: BloqBuilder,
        **soqs: SoquetT,
    ) -> dict[str, SoquetT]:
        """Implement bloq decomposition into sub-bloqs."""
        (soqs["G_0"], soqs["non_identity"]), soqs["target"] = bb.add(
            self.ccz_01,
            ctrl=(soqs["G_0"], soqs["non_identity"]),
            q=soqs["target"],
        )

        soqs["G_0"], soqs["target"] = bb.add(
            CNOT(),
            ctrl=soqs["G_0"],
            target=soqs["target"],
        )

        (soqs["G_0"], soqs["s"]), soqs["target"] = bb.add(
            self.ccz_11,
            ctrl=(soqs["G_0"], soqs["s"]),
            q=soqs["target"],
        )
        soqs["G_1"], soqs["s"] = bb.add(CZ(), q1=soqs["G_1"], q2=soqs["s"])

        return soqs


@attrs.frozen
class RotatedMajorana(Bloq):
    r"""
    Majorana operator conjugated by a Givens rotation ladder.

    Parameters
    ----------
    num_orbitals : int
        Number of spatial orbitals, :math:`N` in ref. [1].
    num_bits_phase_grad : int
        Total number of bits for persistent phase gradient register, :math:`b_{rot}` in
        ref. [1].

    Registers
    ---------
    succ_outer : QBit
        Success flag for the uniform state preparation over the outer index,
        :math:`succ_{x_o}` in ref. [1].

    succ_inner : QBit
        Success flag for the uniform state preparation over the inner index,
        :math:`succ_b` in ref. [1].

    G_0 : QBit
        Qubit selecting between SF and :math:`D_1/Q_1`.
    G_1 : QBit
        Qubit selecting between :math:`D_1` and :math:`Q_1`.
    non_identity : QBit
        Non-identity flag qubit.
    s : QBit
        Superposition qubit selecting :math:`X` or :math:`iY` for :math:`D_1/Q_1`,
        :math:`\varsigma` in ref. [2].

    rotations : QAny
        Givens rotation angles for the basis rotation.

    phase_grad : QAny
        Phase gradient register used throughout the circuit.

    system : QAny
        System register of the selected spin sector.

    References
    ----------
        [1] G. H. Low et al., "Fast Quantum Simulation of Electronic Structure by
            Spectral Amplification", Phys. Rev. X, vol. 15, no. 4, p. 041016, Oct. 2025.
        [2] G. H. Low et al., "A Denser Planar Surface Code", May 28, 2026, arXiv:
            2605.30455.

    """

    num_orbitals: int

    num_bits_phase_grad: int

    @property
    def signature(self) -> Signature:
        """Define input and/or output registers of the bloq."""
        return Signature(
            [
                Register("succ_outer", QBit()),
                Register("succ_inner", QBit()),
                Register("G_0", QBit()),
                Register("G_1", QBit()),
                Register("non_identity", QBit()),
                Register("s", QBit()),
                Register("rotations", QAny(self.ladder.rotations_size)),
                Register("phase_grad", QAny(self.num_bits_phase_grad)),
                Register("system", QAny(self.num_orbitals)),
            ]
        )

    @property
    def ladder(self) -> GivensLadder:
        """Get the Givens rotation ladder used to conjugate the Majorana operator."""
        return GivensLadder(self.num_orbitals, self.num_bits_phase_grad)

    @property
    def cmajorana(self) -> Bloq:
        """Get the Majorana controlled on the two preparation-success flags."""
        return MajoranaDFTHC().controlled(CtrlSpec(cvs=(1, 1)))

    def build_call_graph(self, ssa: SympySymbolAllocator) -> BloqCountDictT:  # noqa: ARG002
        """Build call graph for RotatedMajorana."""
        return {self.ladder: 1, self.cmajorana: 1, self.ladder.adjoint(): 1}

    def build_composite_bloq(
        self,
        bb: BloqBuilder,
        **soqs: SoquetT,
    ) -> dict[str, SoquetT]:
        """Implement bloq decomposition into sub-bloqs."""
        soqs["rotations"], soqs["phase_grad"], soqs["system"] = bb.add(
            self.ladder,
            rotations=soqs["rotations"],
            phase_grad=soqs["phase_grad"],
            system=soqs["system"],
        )

        # The ladder leaves the selected spin orbital on qubit 0
        qs = bb.split(soqs["system"])
        out = bb.add_d(
            self.cmajorana,
            ctrl=[soqs["succ_outer"], soqs["succ_inner"]],
            G_0=soqs["G_0"],
            G_1=soqs["G_1"],
            non_identity=soqs["non_identity"],
            s=soqs["s"],
            target=qs[0],
        )
        qs[0] = out.pop("target")
        soqs["succ_outer"], soqs["succ_inner"] = out.pop("ctrl")
        soqs |= out
        soqs["system"] = bb.join(qs)

        soqs["rotations"], soqs["phase_grad"], soqs["system"] = bb.add(
            self.ladder.adjoint(),
            rotations=soqs["rotations"],
            phase_grad=soqs["phase_grad"],
            system=soqs["system"],
        )

        return soqs


@attrs.frozen
class SelectDFTHC(Bloq):
    r"""
    SELECT oracle for the DFTHC block encoding.

    Parameters
    ----------
    num_orbitals : int
        Number of spatial orbitals, :math:`N` in ref. [1].
    num_bits_phase_grad : int
        Total number of bits for persistent phase gradient register, :math:`b_{rot}` in
        ref. [1].

    Registers
    ---------
    succ_outer : QBit
        Success flag for the uniform state preparation over the outer index,
        :math:`succ_{x_o}` in ref. [1].
    succ_inner : QBit
        Success flag for the uniform state preparation over the inner index,
        :math:`succ_b` in ref. [1].

    G_0 : QBit
        Qubit selecting between SF and :math:`D_1/Q_1`.
    G_1 : QBit
        Qubit selecting between :math:`D_1` and :math:`Q_1`.

    rotations : QAny
        Givens rotation angles for the basis rotation.

    non_identity : QBit
        Non-identity flag qubit.

    spin_0 : QBit
        Spin control for SF terms.
    spin_1 : QBit
        Spin control for :math:`D_1/Q_1` terms.
    spin : QBit
        Selected spin, copied from spin_0 (SF) or spin_1 (:math:`D_1/Q_1`).
    s : QBit
        Superposition qubit selecting :math:`X` or :math:`iY` for :math:`D_1/Q_1`,
        :math:`\varsigma` in ref. [2].

    phase_grad : QAny
        Phase gradient register used throughout the circuit.
    system_down : QAny
        Spin-down system register.
    system_up : QAny
        Spin-up system register.

    References
    ----------
        [1] G. H. Low et al., "Fast Quantum Simulation of Electronic Structure by
            Spectral Amplification", Phys. Rev. X, vol. 15, no. 4, p. 041016, Oct. 2025.
        [2] G. H. Low et al., "A Denser Planar Surface Code", May 28, 2026, arXiv:
            2605.30455.

    """

    num_orbitals: int

    num_bits_phase_grad: int

    @property
    def num_rotations(self) -> int:
        """Givens ladder length for one basis rotation, :math:`N - 1`."""
        return self.num_orbitals - 1

    @property
    def rotations_size(self) -> int:
        """Rotation register size, :math:`(N - 1) b_{rot}`."""
        return self.num_rotations * self.num_bits_phase_grad

    @property
    def signature(self) -> Signature:
        """Define input and/or output registers of the bloq."""
        return Signature(
            [
                Register("succ_outer", QBit()),
                Register("succ_inner", QBit()),
                Register("G_0", QBit()),
                Register("G_1", QBit()),
                Register("rotations", QAny(self.rotations_size)),
                Register("non_identity", QBit()),
                Register("spin_0", QBit()),
                Register("spin_1", QBit()),
                Register("spin", QBit()),
                Register("s", QBit()),  # ς in the original reference
                Register("phase_grad", QAny(self.num_bits_phase_grad)),
                Register("system_down", QAny(self.num_orbitals)),
                Register("system_up", QAny(self.num_orbitals)),
            ]
        )

    @property
    def spin_swap(self) -> SpinSelectSwap:
        """Get the swap moving the selected spin sector into `system_down`."""
        return SpinSelectSwap(self.num_orbitals)

    @property
    def rotated_majorana(self) -> RotatedMajorana:
        """Get the Givens-rotated Majorana applied to the selected spin sector."""
        return RotatedMajorana(self.num_orbitals, self.num_bits_phase_grad)

    def build_call_graph(self, ssa: SympySymbolAllocator) -> BloqCountDictT:  # noqa: ARG002
        """Build call graph for SelectDFTHC."""
        return {
            self.spin_swap: 1,
            self.rotated_majorana: 1,
            self.spin_swap.adjoint(): 1,
            Hadamard(): 5,
        }

    def build_composite_bloq(
        self,
        bb: BloqBuilder,
        **soqs: SoquetT,
    ) -> dict[str, SoquetT]:
        """Implement bloq decomposition into sub-bloqs."""
        soqs["spin_0"] = bb.add(Hadamard(), q=soqs["spin_0"])
        soqs["spin_1"] = bb.add(Hadamard(), q=soqs["spin_1"])
        soqs["s"] = bb.add(Hadamard(), q=soqs["s"])

        soqs |= bb.add_d(self.spin_swap, **extract_soqs(self.spin_swap, soqs))

        # The Majorana acts on the selected spin sector, now held in `system_down`
        soqs["system"] = soqs.pop("system_down")
        soqs |= bb.add_d(
            self.rotated_majorana,
            **extract_soqs(self.rotated_majorana, soqs),
        )
        soqs["system_down"] = soqs.pop("system")

        soqs |= bb.add_d(
            self.spin_swap.adjoint(),
            **extract_soqs(self.spin_swap.adjoint(), soqs),
        )

        soqs["s"] = bb.add(Hadamard(), q=soqs["s"])
        soqs["spin_0"] = bb.add(Hadamard(), q=soqs["spin_0"])

        return soqs
