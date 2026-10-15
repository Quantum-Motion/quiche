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

"""
Helpers to handle factorisation output database files.

References
----------
.. [Low2025] G. H. Low et al., "Fast Quantum Simulation of Electronic Structure by
    Spectral Amplification", Phys. Rev. X, vol. 15, no. 4, p. 041016, Oct. 2025.

"""

import io
import json
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np

from quiche.core.electronic import DFTHCHamiltonian


def load(path: str | Path, job_id: int) -> DFTHCHamiltonian:
    """
    Return a DFTHCHamiltonian that functions as input for the resource estimation.

    Parameters
    ----------
    path : str | Path
        The database file generated with external DFTHC code.
    job_id : int
        The ID of the job to extract.

    Returns
    -------
    DFTHCHamiltonian
        The electronic Hamiltonian in double-factorized tensor hypercontraction (DFTHC)
        form, containing the number of orbitals, number of electrons, number of ranks,
        number of bases, number of copies copies, together with the unit vectors,
        weight vectors, BLISS matrix, one-body term and constant
        shift. The unit and weight vectors are returned in RBC order.
        The originating ``job_id`` is stored on the object.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not point towards an existing file.
    ValueError
        If no row with the given ``job_id`` exists in the database.
    ValueError
        If the job exists but its status is not ``"completed"``.

    """
    path = Path(path)
    if not path.is_file():
        error_msg = f"Database file not found: {path}"
        raise FileNotFoundError(error_msg)

    # Open read-only so that sqlite3 never creates a file as a side effect.
    uri = f"{path.resolve().as_uri()}?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as con:
        row = con.execute(
            "SELECT status, inputs, results, data_blob FROM experiments WHERE id=?",
            (job_id,),
        ).fetchone()

    if row is None:
        error_msg = f"No job with id {job_id} in {path}"
        raise ValueError(error_msg)
    status, inputs, results, blob = row
    if status != "completed":
        error_msg = f"Job {job_id} has status '{status}', not 'completed'"
        raise ValueError(error_msg)

    outer_rank, copies, inner_rank = json.loads(inputs)["VLM"]
    results = json.loads(results)
    arrays = np.load(io.BytesIO(blob))

    # External code that generates the input uses convention VLM,
    # which corresponds to RCB.
    # Just for clarity's sake, we transpose and save in the RBC order,
    # as used in [Low2025]_.
    unit_vectors = np.asarray(arrays["R_vpm"], dtype=np.float64).transpose(0, 2, 1)
    weight_vectors = np.asarray(arrays["F_vlm"], dtype=np.float64).transpose(0, 2, 1)

    return DFTHCHamiltonian(
        num_orbitals=results["num_orb"],
        num_ranks=outer_rank,
        num_bases=inner_rank,
        num_copies=copies,
        unit_vectors=unit_vectors,
        weight_vectors=weight_vectors,
        bliss_matrix=np.asarray(arrays["B_bliss"], dtype=np.float64),
        factorised_one_body=np.asarray(arrays["h1_exact"], dtype=np.float64),
        const=float(arrays["const"]),
        num_electrons=results["num_elec"],
    )
