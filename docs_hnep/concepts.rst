Concepts
========

This page gives a brief tour of the ideas behind HNEP. If you have read the
thesis it will be familiar — the docstrings in :mod:`hnep` go deeper.

The two probes
--------------

HNEP combines **two independent probes** of quantum contribution.

**Surrogation.** Train a classical surrogate to predict the quantum branch's
output from the dataset's classical features. The Surrogation Score (SS) is
``1 − mean R²`` across quantum dims, summarised across an 8-surrogate ladder
(linear, KNN, trees, MLPs). Low SS = REPLACEABLE; high SS = NECESSARY.

**Intervention.** Substitute the quantum output at inference and measure the
relative R² drop on the test set. Four interventions are run:

* ``zero_quantum``  ↔ ``do(quantum = ∅)``
* ``random_noise``  ↔ ``do(quantum = noise(σ_q))``
* ``permute``       ↔ ``do(quantum = π(quantum))``
* ``constant``      ↔ ``do(quantum = mean(quantum))``

Large Δ on the headline ``zero_quantum`` means the decoder is structurally
dependent on quantum. The other three help distinguish *specific-information*
dependence from *placeholder* dependence — disagreement among them is the
most informative single signal this probe produces.

The QCT taxonomy
----------------

Cross-tabulating the two probes yields the 2×2 taxonomy:

+------------------+----------------------+---------------------+
|                  | LOAD-BEARING         | NOT LOAD-BEARING    |
+==================+======================+=====================+
| **NECESSARY**    | Genuine Contributor  | Unique-but-Ignored  |
+------------------+----------------------+---------------------+
| **REPLACEABLE**  | Architectural Reg.   | Dead Weight         |
+------------------+----------------------+---------------------+

When confidence intervals straddle a threshold, HNEP returns
``Inconclusive`` rather than guessing.

The classifier
--------------

:class:`hnep.QCTClassifier` is the small data class that turns probe results
into verdicts. Defaults match the thresholds used throughout the thesis
benchmarks:

* SS < 0.20 ⇒ REPLACEABLE
* relative R² drop ≥ 5% ⇒ LOAD-BEARING

You can override these via :class:`hnep.Thresholds`.

The killer figures
------------------

Three visualisations ship with v0.1.0:

* **QCT plane** (the HNEP logo) — SS on x, Δ on y, four shaded quadrants,
  confidence rectangles per model.
* **Convergent validity radar** — one axis per probe; jaggedness reveals
  disagreement between probes.
* **Pareto plot** — R² vs compute, log x-axis, with optional NISQ hardware
  cost annotations.

All three accept either a single :class:`HNEPResult` or a list (to overlay
multiple models in one chart).
