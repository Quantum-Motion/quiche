# %% tags=["remove-cell"]
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

# %% [markdown]
# # 2. Introduction to the Paulis module
#
# The Paulis module (`quiche.core.paulis`) contains the core engine to process the Pauli
# strings and their sums, necessary to construct the qubit Hamiltonians.
#
# The Paulis module contains three classes: `Pauli`, `PauliWord` and `PauliSum` to
# represent the relevant Pauli objects. Below, we are going to cover their example
# usage.

# %% [markdown]
# ## 2.1. `Pauli`: Non-identity single-qubit Pauli operator
#
# `Pauli` is used to represent the most basic unit of the Pauli operator, i.e. the
# single-qubit Pauli operators.

# %%
from quiche.core import Pauli

# %%
# Define the single-qubit Pauli X operator.
pauli_x = Pauli("X")
print("Pauli X matrix:\n", pauli_x.to_matrix())

# %% [markdown]
# We defined the single-qubit Pauli X operator and successfully printed the matrix
# representation of it. The same goes for the Pauli Y and Z operators.

# %% [markdown]
# ## 2.2. `PauliWord`: Multi-qubit tensor products of Pauli operators
#
# `PauliWord` is used to define a Pauli string of arbitrary length.

# %%
from quiche.core import PauliWord

# %%
# Construct a Pauli word.
pword = PauliWord(paulis=(Pauli.X, Pauli.Y, Pauli.Z), qubits=(0, 1, 2))
print("PauliWord:", pword)
pword = PauliWord(paulis=("X", "Y", "Z"), qubits=(0, 1, 2))
print("PauliWord:", pword)

# %% [markdown]
# ### 2.2.1. Convert to and from a string
# In QUICHE, we have two types of strings to represent Pauli strings, namely dense and
# sparse strings.
#
# - **Dense string**: Each character in the string should correspond to the (possibly
# identity) Pauli acting on the corresponding qubit, e.g.: `XIZY`
#
# Note that at least one Pauli must be non-identity in the dense string.
#
# - **Sparse string**: Each Pauli is written as `P(q)` for a (non-identity) Pauli `P`
# acting on (zero-indexed) qubit `q`, separated by spaces, e.g.: `X(0) Z(2)`
#
# Note that each qubit index may appear at most once. Untargeted qubits are implicitly
# treated as identities and must not be explicitly written.
#
# Below, we show how to use the strings to construct the `PauliWord` objects and vice
# versa.

# %%
# Convert to and from dense string formats (PauliWord).
str_dense = "XYZ"
pword = PauliWord.from_dense_string(str_dense, big_endian=True)
print("PauliWord:", pword)
print("Dense string:", pword.to_dense_string(big_endian=True))

# %%
# Convert to and from sparse string formats (PauliWord).
str_sparse = "X(0) Z(2)"
pword = PauliWord.from_sparse_string(str_sparse)
print("PauliWord:", pword)
print("Sparse string:", pword.to_sparse_string())

# %% [markdown]
# ### 2.2.2. Convert a Pauli word to a matrix (`np.ndarray`)
#
# `PauliWord` supports the conversion of the Pauli string into a matrix with the
# `.to_matrix` method. One can specify whether to ignore the idle qubits with the
# identity operations or not.

# %%
# Convert a Pauli word to a NumPy matrix with `to_matrix`.
str_sparse = "X(0)"
pword = PauliWord.from_sparse_string(str_sparse)
print("PauliWord:", pword)
print("NumPy matrix (full):\n", pword.to_matrix(length=2, ignore_idle_qubits=False))
print("NumPy matrix (ignore):\n", pword.to_matrix(length=2, ignore_idle_qubits=True))

# %% [markdown]
# ## 2.3. `PauliSum`: Linear combination of Pauli words
#
# `PauliSum` is used to define the sum of Pauli strings.

# %%
from quiche.core import PauliSum

# %%
# Construct a PauliSum object.
# Define the constituent PauliWord objects.
pwords = [
    PauliWord(paulis=("X", "Y", "Z"), qubits=(0, 1, 2)),
    PauliWord.from_dense_string("ZIZ", big_endian=True),
    PauliWord.from_sparse_string("X(0)"),
]

# Define the coefficients corresponding to the PauliWord objects.
coefficients = [0.3, 0.4, 0.1]
identity_coefficient = 0.05

psum = PauliSum(
    coefficients=coefficients, words=pwords, identity_coefficient=identity_coefficient
)
print("PauliSum:", psum)

# %% [markdown]
# ### 2.3.1. Convert to and from a string
#
# Similar to `PauliWord`, `PauliSum` supports the conversion to and from dense and
# sparse strings. Below, we show how to use the strings to construct the `PauliSum`
# objects and vice versa.

# %%
# Convert to and from dense string formats (PauliSum).
str_dense = """\
+10.0 III
+5.0 XIZ
-2.0 ZYI
"""

psum = PauliSum.from_dense_string(str_dense, big_endian=True)
print("PauliSum:", psum)

# %%
# Convert to and from sparse string formats (PauliSum).
str_sparse = """\
+10.0
+5.0 X(0) Z(2)
-2.0 Z(0) Y(1)
"""

psum = PauliSum.from_sparse_string(str_sparse)
print("PauliSum:", psum)

# %% [markdown]
# ### 2.3.2. Convert a `PauliSum` object to a matrix (`np.ndarray`)
# Similar to `PauliWord`, `PauliSum` supports the conversion of the sum of Pauli strings
# into a matrix with the `.to_matrix` method.

# %%
# Convert a PauliSum object to a NumPy matrix with `to_matrix`.
print("NumPy matrix:\n", psum.to_matrix())

# %% [markdown]
# Additionally, one can obtain the 1-norm of the `PauliSum` object.

# %%
# Obtain the 1-norm of the PauliSum object.
print("1-norm:", psum.lam)

# %% [markdown]
# ### 2.3.3. Special methods: `without_identity` and `split_identity`
#
# `PauliSum` has special methods to either delete or split the identity term from the
# other non-identity Pauli terms. Such methods are useful as the identity term often
# contributes only a global phase to the evolved state in Hamiltonian simulations.

# %%
# Strip the identity out of the PauliSum.
print("PauliSum without the identity:", psum.without_identity())

# %%
identity_coefficient, psum_no_identity = psum.split_identity()
print("Identity coefficient (split_identity):", identity_coefficient)
print("PauliSum without the identity (split_identity):", psum_no_identity)
