quiche.bindings
===============

QuEST bindings
--------------

.. automodule:: quiche.bindings.quest_bindings
   :undoc-members:

QUICHE bindings
---------------

.. `automodule` silently skips `nb_func` objects. As a workaround we manually call
   `autofunction` on each of the QUICHE binding functions. Eventually might be useful to
   replace with a custom Directive in `conf.py`.

.. currentmodule:: quiche.bindings.quiche_bindings

.. autofunction:: getHartreeFockStateBK
.. autofunction:: getHartreeFockStateJW
.. autofunction:: getHartreeFockStateParity
.. autofunction:: getPhaseIterativeQDRIFT
.. autofunction:: getPhaseIterativeTrotter
.. autofunction:: getPhaseKitaevQDRIFT
.. autofunction:: getPhaseKitaevTrotter
.. autofunction:: getPhaseNaiveQDRIFT
.. autofunction:: getPhaseNaiveTrotter
.. autofunction:: getPhaseTextbookQDRIFT
.. autofunction:: getPhaseTextbookQubitised
.. autofunction:: getPhaseTextbookQubitisedOptimised
.. autofunction:: getPhaseTextbookTrotter
.. autofunction:: initClassicalState
