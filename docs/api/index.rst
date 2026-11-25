API reference
=============

The ``quiche`` package is organised into a small set of subpackages:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Subpackage
     - Purpose
   * - :doc:`algorithms <algorithms>`
     - Data structures defining state preparation, Hamiltonian simulation and phase estimation algorithms.
   * - :doc:`bindings <bindings>`
     - Python bindings to the compiled QuEST simulator and QUICHE backend.
   * - :doc:`chemistry <chemistry>`
     - Chemistry helpers including Hartree-Fock states and fermion-to-qubit state mappings.
   * - :doc:`core <core>`
     - Data structures for Hamiltonians, Pauli operators.
   * - :doc:`dispatch <dispatch>`
     - Specifying a calculation and dispatching it to a backend.
   * - :doc:`io <io>`
     - File parsers for Hamlib electronic Hamiltonians and FCIDUMP electronic integral files.
   * - :doc:`resources <resources>`
     - The Qualtran-based resource estimation backend: bloqs and logical cost functions.
   * - :doc:`simulation <simulation>`
     - The QuEST-based simulation backend.

.. toctree::
   :hidden:
   :maxdepth: 2

   algorithms
   bindings
   chemistry
   core
   dispatch
   io
   resources
   simulation
