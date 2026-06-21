API reference
=============

This page exposes the public HNEP API. Internal symbols (anything starting
with ``_``) are not part of the stability guarantee.

Top-level
---------

.. autofunction:: hnep.evaluate

.. autoclass:: hnep.HNEPResult
   :members:

.. autoclass:: hnep.ProbeResult
   :members:

.. autoclass:: hnep.QCTVerdict
   :members:

.. autoclass:: hnep.Thresholds
   :members:

Adapters
--------

.. autoclass:: hnep.Dataset
   :members:

.. autoclass:: hnep.ModelInterface
   :members:

.. autoclass:: hnep.FunctionalAdapter
   :members:

.. autoclass:: hnep.PrecomputedAdapter
   :members:

.. autoclass:: hnep.adapters.JaxFlaxAdapter
   :members:

.. autoclass:: hnep.adapters.PyTorchAdapter
   :members:

Probes
------

.. autoclass:: hnep.Probe
   :members:

.. autoclass:: hnep.SurrogationProbe
   :members:

.. autoclass:: hnep.InterventionProbe
   :members:

Classifiers
-----------

.. autoclass:: hnep.QCTClassifier
   :members:

Cost-utility
------------

.. autofunction:: hnep.compute_qus

.. autoclass:: hnep.PointMeasurement
   :members:

.. autofunction:: hnep.estimate_hardware_cost

Visualisations
--------------

.. autofunction:: hnep.plot_qct_plane

.. autofunction:: hnep.plot_convergent_validity_radar

.. autofunction:: hnep.plot_pareto_with_hardware_cost

Reports
-------

.. autofunction:: hnep.render_html_report
