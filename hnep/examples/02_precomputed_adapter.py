"""Run HNEP on cached extractions — no model framework required.

A common workflow: you train your model on a GPU server, dump the test-set
extractions (gnn_emb, quantum_output, intact_predictions) plus the decoder
weights to disk, and analyse them on your laptop. PrecomputedAdapter wraps
those numpy arrays + a decoder callable directly.

    python -m hnep.examples.02_precomputed_adapter
"""

from __future__ import annotations

import numpy as np

import hnep
from hnep.adapters.base import Dataset


def main() -> None:
    rng = np.random.default_rng(0)
    N, D = 400, 16

    inputs = rng.normal(size=(N, D))
    targets = inputs @ rng.normal(size=D) + 0.1 * rng.normal(size=N)

    dataset = Dataset(
        inputs=inputs, targets=targets,
        train_idx=np.arange(0, 280),
        val_idx=np.arange(280, 340),
        test_idx=np.arange(340, 400),
        metadata={"name": "synthetic"},
    )

    # Cached extractions
    proj = rng.normal(size=(D, 4))
    quantum_outputs = inputs @ proj
    gnn_embeddings = rng.normal(size=(N, 32))

    decoder_w_g = rng.normal(size=32) * 0.05
    decoder_w_q = rng.normal(size=4) * 1.5

    def decoder(g_batch, q_batch):
        return g_batch @ decoder_w_g + q_batch @ decoder_w_q

    intact_predictions = decoder(gnn_embeddings, quantum_outputs)

    adapter = hnep.PrecomputedAdapter(
        name="CachedHybridModel",
        gnn_embeddings=gnn_embeddings,
        quantum_outputs=quantum_outputs,
        intact_predictions=intact_predictions,
        decoder_fn=decoder,
    )

    result = hnep.evaluate(adapter, dataset)
    print(result.summary())


if __name__ == "__main__":
    main()
