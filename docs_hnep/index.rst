HNEP — Hybrid Network Evaluation Protocol
==========================================

.. raw:: html

   <p style="font-size: 1.2em; color:#555; font-style: italic;">
   Does the quantum component in your hybrid model actually contribute
   meaningful computation, or could a classical surrogate replace it?
   </p>

Most QML benchmarks report end-task accuracy and call it a day.
**HNEP applies multiple independent probes** to your trained hybrid model and
returns a verdict you can defend.

Quick install
-------------

.. code-block:: bash

   pip install hnep

30-second example
-----------------

.. code-block:: python

   import hnep

   adapter = hnep.FunctionalAdapter(
       name="my_model",
       predict_fn=my_predict,
       extract_quantum_fn=my_extract_quantum,
       predict_with_override_fn=my_predict_override,
       quantum_dim=4,
   )

   result = hnep.evaluate(adapter, my_dataset)
   print(result.qct_verdict)        # "Regularizer" / "Genuine" / "Ignored" / "Dead Weight"
   result.to_html("report.html")

Contents
--------

.. toctree::
   :maxdepth: 2

   quickstart
   concepts
   tutorial
   api

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
