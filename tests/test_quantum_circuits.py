"""
Manual smoke test for quantum circuit predictions.
Requires trained checkpoint at experiments/checkpoints/quantum_vqc_full/best.npz
Run directly: python tests/test_quantum_circuits.py
"""

import numpy as np
import jax
import jax.numpy as jnp
from src.models.quantum.quantum_circuits import QuantumNeuralNetwork
from src.data.quantum_encoding import QuantumEncoder
from src.data.molecular_features import MolecularFeaturizer
from src.data.quantum_encoding import GraphToVectorFeatures


def main():
    # Load model/params correctly (unflatten)
    qnn = QuantumNeuralNetwork(n_qubits=4, n_layers=2, output_dim=1)
    dummy = qnn.initialize_parameters()
    leaves, treedef = jax.tree_util.tree_flatten(dummy)
    ckpt = np.load("experiments/checkpoints/quantum_vqc_full/best.npz", allow_pickle=True)
    params = jax.tree_util.tree_unflatten(
        treedef, [jnp.array(ckpt[f"param_{i}"]) for i in range(len(leaves))]
    )

    # Load encoder
    encoder = QuantumEncoder.load("data/processed/quantum_encoder.pkl")
    feat = MolecularFeaturizer()

    def predict(smiles):
        g = feat.smiles_to_graph(smiles)
        vec = GraphToVectorFeatures.aggregate_node_features({
            "node_features": g.node_features,
            "edge_index": g.edge_index,
            "edge_features": g.edge_features
        })
        qx = encoder.transform(vec)
        return float(qnn.forward(params, qx))

    for s in ["C", "CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]:
        print(s, predict(s))


if __name__ == "__main__":
    main()
