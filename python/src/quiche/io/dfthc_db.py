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
.. [Low2026] G. H. Low et al., "A Denser Planar Surface Code", May 28, 2026,
    arXiv: 2605.30455.

"""

import io
import json
import sqlite3
from pathlib import Path

import numpy as np

from quiche.core.electronic import DFTHCHamiltonian


def parse(path: str | Path, job_id: int) -> DFTHCHamiltonian:
    """
    Return a DFTHCHamiltonian that functions as input for the resource estimation.

    Parameters
    ----------
    path : str
        The database file generated with external DFTHC code.
    job_id : int
        The ID of the job you want to extract.
        This selects the row (e.g. one of the 4 jobs run in this example).

    """
    with sqlite3.connect(path) as con:
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
    t = np.load(io.BytesIO(blob))

    # External code that generates the input uses convention VLM,
    # which corresponds to RCB.
    # Just for clarity's sake, we transpose and save in the RBC order,
    # as used in [Low2025]_.
    u = np.asarray(t["R_vpm"], dtype=np.float64).transpose(0, 2, 1)
    w = np.asarray(t["F_vlm"], dtype=np.float64).transpose(0, 2, 1)

    return DFTHCHamiltonian(
        num_orbitals=results["num_orb"],
        num_ranks=outer_rank,
        num_bases=inner_rank,
        num_copies=copies,
        unit_vectors=u,
        weight_vectors=w,
        bliss_matrix=np.asarray(t["B_bliss"], dtype=np.float64),
        h1=np.asarray(t["h1_exact"], dtype=np.float64),
        const=float(t["const"]),
        num_electrons=results["num_elec"],
        job_id=job_id,
    )
