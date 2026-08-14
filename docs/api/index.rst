API reference
=============

The ``quiche`` package is organised into a small set of subpackages:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Subpackage
     - Purpose
   * - :doc:`core <core>`
     - Data structures for Hamiltonians, Pauli operators, algorithm choices and error
       budgets.
   * - :doc:`chemistry & hamlib <chemistry>`
     - Chemistry helpers (Hartree-Fock states, fermion-to-qubit state mappings) and Hamlib
       file parsing.
   * - :doc:`dispatch <dispatch>`
     - Specifying a calculation and dispatching it to a backend, plus the error-budget
       logic that fixes routine parameters.
   * - :doc:`resources <resources>`
     - The Qualtran-based resource estimation backend: bloqs and logical cost functions.
   * - :doc:`simulation <simulation>`
     - The QuEST-based simulation backend.

.. toctree::
   :maxdepth: 2

   core
   chemistry
   dispatch
   resources
   simulation
