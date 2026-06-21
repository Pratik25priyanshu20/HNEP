Tutorial — wrapping your own model
===================================

This tutorial walks through the most common HNEP workflow: training a
hybrid model on your own data, then running HNEP on it.

There is **no requirement** that your model be molecular, quantum, or
related to the thesis benchmarks. Anything that has a "quantum branch"
producing a per-sample output vector is fair game.

Setting up the dataset
----------------------

HNEP datasets carry both inputs (classical descriptors) and targets, plus
explicit train/val/test indices:

.. code-block:: python

   import numpy as np
   from hnep import Dataset

   dataset = Dataset(
       inputs=np.array(...),          # (N, d_input)
       targets=np.array(...),         # (N,) or (N, k)
       train_idx=np.array(...),
       val_idx=np.array(...),
       test_idx=np.array(...),
       metadata={"name": "my_task"},
   )

The metadata dict is for *your* convenience — HNEP never requires anything
specific in it.

Recommended adapter — FunctionalAdapter
---------------------------------------

For first-time users this is the cleanest path. Write three callables that
match the HNEP interface:

.. code-block:: python

   def predict_fn(dataset, indices):
       """Return predictions in the original target scale."""
       X = dataset.inputs[indices]
       return my_model.predict(X)

   def extract_quantum_fn(dataset, indices):
       """Return the quantum branch's per-sample output, shape (n, q_dim)."""
       X = dataset.inputs[indices]
       return my_model.quantum_output(X)

   def predict_with_override_fn(dataset, q_override, indices):
       """Predict using the supplied quantum override instead of the
       model's own quantum output."""
       X = dataset.inputs[indices]
       return my_model.predict_with_quantum(X, q_override)

   adapter = hnep.FunctionalAdapter(
       name="MyHybridModel",
       predict_fn=predict_fn,
       extract_quantum_fn=extract_quantum_fn,
       predict_with_override_fn=predict_with_override_fn,
       quantum_dim=4,
   )

Run HNEP
--------

.. code-block:: python

   result = hnep.evaluate(adapter, dataset)
   print(result.summary())

The default probe battery runs surrogation + intervention with bootstrap
CIs. To use a custom set:

.. code-block:: python

   from hnep import SurrogationProbe, InterventionProbe
   from hnep.probes.surrogation import default_surrogate_ladder

   result = hnep.evaluate(adapter, dataset, probes=[
       SurrogationProbe(
           surrogates=default_surrogate_ladder()[:4],   # only 4 surrogates
           n_bootstrap=200,
       ),
       InterventionProbe(interventions=["zero_quantum", "random_noise"]),
   ])

Generate reports and figures
----------------------------

.. code-block:: python

   # HTML report (self-contained, base64 figures embedded)
   result.to_html("report.html")

   # JSON / CSV
   result.to_json("report.json")
   result.to_csv("report.csv")

   # Standalone figures
   from hnep import plot_qct_plane, plot_convergent_validity_radar
   plot_qct_plane(result).savefig("qct.pdf")
   plot_convergent_validity_radar(result).savefig("radar.pdf")

Comparing models
----------------

Overlay multiple models on the same QCT plane / radar:

.. code-block:: python

   result_a = hnep.evaluate(adapter_a, dataset)
   result_b = hnep.evaluate(adapter_b, dataset)

   fig = plot_qct_plane([result_a, result_b])
   fig.savefig("comparison.pdf")

Cost-utility comparison
-----------------------

Compute QUS for a quantum vs classical pairing:

.. code-block:: python

   from hnep import compute_qus, estimate_hardware_cost

   qus = compute_qus(
       quantum_model="Hybrid",
       classical_model="MPNN",
       r2_quantum=0.83, r2_classical=0.65,
       time_quantum_s=3000, time_classical_s=20,
   )
   print(qus.qus)        # 0.00012
   print(qus.verdict)    # "quantum buys R² marginally per-FLOP"

   # Hardware cost projection
   for estimate in estimate_hardware_cost():
       print(f"{estimate.provider}: ${estimate.estimated_usd_per_run:,.0f}/run")

Pareto plot:

.. code-block:: python

   from hnep import PointMeasurement, plot_pareto_with_hardware_cost
   pts = [
       PointMeasurement("GNN", accuracy=0.43, compute_cost=12),
       PointMeasurement("MPNN", accuracy=0.83, compute_cost=15),
       PointMeasurement("Hybrid", accuracy=0.83, compute_cost=7000,
                        is_quantum=True),
   ]
   plot_pareto_with_hardware_cost(pts).savefig("pareto.pdf")
