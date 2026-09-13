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

"""Helpers to read and parse FCIDUMP electronic integral files."""

import re
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from quiche.core.electronic import SecondQuantisedHamiltonian


def _strip_header_start(text: str) -> str:
    """Strip the '&FCI' marker from the start of an FCIDUMP."""
    # Split to first whitespace
    parts = text.split(maxsplit=1)

    if len(parts) < 2 or parts[0] != "&FCI":
        error_msg = "FCIDUMP must start with '&FCI'."
        raise ValueError(error_msg)

    return parts[1]


def _split_at_terminator(text: str) -> tuple[str, str]:
    """Split an FCIDUMP at the namelist terminator."""
    # Split to header terminator
    parts = re.split(r"/|&END", text, maxsplit=1)

    if len(parts) < 2:
        error_msg = "FCIDUMP header is not terminated."
        raise ValueError(error_msg)

    return parts[0], parts[1]


def _extract_namelist_and_integrals(text: str) -> tuple[str, str]:
    """Get the namelist and integral sections of an FCIDUMP."""
    return _split_at_terminator(_strip_header_start(text))


def _split_namelist(namelist: str) -> dict[str, list[str]]:
    """Split the namelist into keys and their values."""
    # Normalise commas
    namelist = namelist.replace(",", " ")

    # Capture namelist into: leading text, key, value, key, value, ...
    parts = re.split(r"([a-zA-Z]\w*)\s*=", namelist)

    if stray := parts[0].strip():
        error_msg = f"Unexpected text before the first namelist key: {stray!r}."
        raise ValueError(error_msg)

    return {
        key.upper(): value.split()
        for key, value in zip(parts[1::2], parts[2::2], strict=True)
    }


def _require_int(parameters: dict[str, list[str]], key: str) -> int:
    """Read a mandatory single-valued integer parameter."""
    error_msg = f"FCIDUMP header needs a single integer {key}."

    fields = parameters.get(key, [])
    if len(fields) != 1:
        raise ValueError(error_msg)

    try:
        return int(fields[0])
    except ValueError as exc:
        raise ValueError(error_msg) from exc


def _check_restricted(parameters: dict[str, list[str]]) -> None:
    """Ensure FCIDUMP corresponds to a spin-restricted file."""
    flags = [
        value[0].upper() for key in ("UHF", "IUHF") if (value := parameters.get(key))
    ]
    if any("1" in flag or "T" in flag for flag in flags):
        error_msg = "Spin-unrestricted (UHF) FCIDUMP files are not supported."
        raise NotImplementedError(error_msg)


def _check_orbitals(norb: int) -> None:
    """Validate the number of spatial orbitals is positive."""
    if norb < 1:
        error_msg = f"FCIDUMP header needs a positive NORB, got {norb}."
        raise ValueError(error_msg)


def _check_electrons(nelec: int, norb: int) -> None:
    """Validate the electron count fits the orbital space."""
    if not 0 <= nelec <= 2 * norb:
        error_msg = f"FCIDUMP header needs NELEC between 0 and {2 * norb}, got {nelec}."
        raise ValueError(error_msg)


def _parse_namelist(namelist: str) -> tuple[int, int]:
    """Parse the namelist into the orbital and electron counts."""
    parameters = _split_namelist(namelist)

    _check_restricted(parameters)

    norb = _require_int(parameters, "NORB")
    nelec = _require_int(parameters, "NELEC")

    _check_orbitals(norb)
    _check_electrons(nelec, norb)

    return norb, nelec


def _parse_integral_line(
    fields: list[str],
    norb: int,
) -> tuple[float, tuple[int, int, int, int]]:
    """Parse one integral into its coefficient and orbital indices."""
    error_msg = f"Malformed integral: {fields!r}."

    if len(fields) != 5:
        raise ValueError(error_msg)

    try:
        coefficient = float(fields[0])
        indices = tuple(int(field) for field in fields[1:])
    except ValueError as exc:
        raise ValueError(error_msg) from exc

    if any(not 0 <= index <= norb for index in indices):
        raise ValueError(error_msg)

    return coefficient, indices


def _parse_integrals(
    integrals: str,
    norb: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Construct the integral tensors and the core energy from the integrals."""
    one_body = np.zeros((norb, norb))
    two_body = np.zeros((norb,) * 4)
    core_energy = 0.0

    for line in integrals.splitlines():
        fields = line.split()

        if not fields:
            continue

        coefficient, (p, q, r, s) = _parse_integral_line(fields, norb)

        if p and q and r and s:
            # FCIDUMP orbital indices are 1-based
            (p, q, r, s) = (p - 1, q - 1, r - 1, s - 1)

            # Assumes real orbitals (8-fold symmetry)
            for orbit in (
                (p, q, r, s),
                (q, p, r, s),
                (p, q, s, r),
                (q, p, s, r),
                (r, s, p, q),
                (s, r, p, q),
                (r, s, q, p),
                (s, r, q, p),
            ):
                two_body[orbit] = coefficient

        elif p and q and not (r or s):
            (p, q) = (p - 1, q - 1)
            one_body[p, q] = one_body[q, p] = coefficient

        elif not (p or q or r or s):
            core_energy = coefficient

        else:
            error_msg = f"Unrecognised index pattern: {(p, q, r, s)}."
            raise ValueError(error_msg)

    return one_body, two_body, core_energy


def parse(text: str) -> SecondQuantisedHamiltonian:
    """Parse an FCIDUMP string into a SecondQuantisedHamiltonian."""
    namelist, integrals = _extract_namelist_and_integrals(text)
    norb, nelec = _parse_namelist(namelist)
    one_body, two_body, core_energy = _parse_integrals(integrals, norb)

    return SecondQuantisedHamiltonian(
        one_body=one_body,
        two_body=two_body,
        electrons=nelec,
        core_energy=core_energy,
    )


def read(path: str | Path) -> SecondQuantisedHamiltonian:
    """Read and parse an FCIDUMP file."""
    return parse(Path(path).read_text())
