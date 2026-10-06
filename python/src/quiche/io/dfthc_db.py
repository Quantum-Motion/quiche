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

"""Helpers to handle factorisation output database files.

References
----------
    [1] G. H. Low et al., "Fast Quantum Simulation of Electronic Structure by
        Spectral Amplification", Phys. Rev. X, vol. 15, no. 4, p. 041016, Oct. 2025.
            
"""

from pathlib import Path

import numpy as np
import json
import io
import sqlite3

from quiche.core.electronic import FactorisedHamiltonian


def parse(path: str| Path, job_id: int) -> FactorisedHamiltonian:
    """
    Return a dataclass object that could function as a potential quiche input for one completed job saved in the database file.

    :param path:        The database file generated with external DFTHC code.
    :param job_id:      The ID of the job you want to extract. This selects the row (e.g. one of the 4 jobs run in this example).
    """

    with sqlite3.connect(path) as con:
        row = con.execute(
            "SELECT status, inputs, results, data_blob FROM experiments WHERE id=?", (job_id,)
        ).fetchone()
    if row is None:
        raise ValueError(f"No job with id {job_id} in {path}")
    status, inputs, results, blob = row
    if status != "completed":
        raise ValueError(f"Job {job_id} has status '{status}', not 'completed'")

    V, L, M = json.loads(inputs)["VLM"]
    results = json.loads(results)
    t = np.load(io.BytesIO(blob))

    # External code that generates the input uses convention VLM, which corresponds to RCB.
    # Just for clarity's sake, we transpose and save in the RBC order, as used in ref. [1].
    U = np.asarray(t["R_vpm"], dtype=np.float64).transpose(0, 2, 1) 
    W = np.asarray(t["F_vlm"], dtype=np.float64).transpose(0, 2, 1)

    return FactorisedHamiltonian(
        N=results["num_orb"], R=V, B=M, C=L,
        U=U, W=W,
        bliss_matrix=np.asarray(t["B_bliss"], dtype=np.float64),
        h1=np.asarray(t["h1_exact"], dtype=np.float64),
        const=float(t["const"]),
        electrons=results["num_elec"],
        job_id=job_id,
    )
