Quickstart
==========

This page walks you through running your first HNEP evaluation in under
five minutes.

Install
-------

.. code-block:: bash

   pip install hnep

Optional framework extras:

.. code-block:: bash

   pip install "hnep[jax]"          # JAX / Flax models
   pip install "hnep[pytorch]"      # PyTorch models
   pip install "hnep[molecular]"    # RDKit utilities
   pip install "hnep[all]"          # everything

Wrap your model in an adapter
-----------------------------

Pick whichever adapter matches your situation:

* :class:`hnep.FunctionalAdapter` — most general. Provide three callables and
  HNEP probes the model through them.
* :class:`hnep.PrecomputedAdapter` — wrap cached numpy arrays plus a decoder
  function. Ideal when you have already extracted everything from a trained
  model into ``.npz`` files.
* :class:`hnep.adapters.JaxFlaxAdapter` — for Flax models conforming to the
  reference Hybrid-V1 surface.
* :class:`hnep.adapters.PyTorchAdapter` — skeleton template for PyTorch users.

The minimal interface a model needs to expose:

.. code-block:: python

   def predict(dataset, indices) -> np.ndarray: ...
   def extract_quantum_output(dataset, indices) -> np.ndarray: ...
   def predict_with_quantum_override(dataset, q_override, indices) -> np.ndarray: ...

Run a full evaluation
---------------------

.. code-block:: python

   import hnep

   result = hnep.evaluate(adapter, dataset)
   print(result.summary())

That call runs HNEP's default battery (surrogation + intervention),
classifies the result, and computes confidence intervals via bootstrap.

Inspect the result
------------------

.. code-block:: python

   print(result.qct_verdict)     # "Regularizer", "Genuine", "Ignored", ...
   print(result.qct_confidence)  # 0.0 – 1.0

   # Per-probe scores
   for name, probe in result.probes.items():
       print(f"{name}: {probe.primary_score:.3f} → {probe.verdict}")

Generate a report
-----------------

.. code-block:: python

   result.to_html("report.html")    # full HTML report with figures
   result.to_json("report.json")    # JSON for tooling
   result.to_csv("report.csv")      # flat per-probe table

The HTML report is a single self-contained file with embedded base64 figures
— easy to share, attach to a paper, or upload to GitHub.

Replay a run
------------

Every evaluation captures a manifest (``result.manifest``) recording the HNEP
version, timestamp, platform, thresholds, and probes that were run. Save it
alongside your model artifacts to make your evaluation reproducible:

.. code-block:: python

   import json
   json.dump(result.manifest, open("hnep_manifest.json", "w"), indent=2)

(``hnep replay`` CLI support lands in v0.2.)
