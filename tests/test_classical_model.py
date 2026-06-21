"""
Manual smoke test for classical model predictions.
Requires trained checkpoint at experiments/checkpoints/classical_gnn/best.npz
Run directly: python tests/test_classical_model.py
"""

import numpy as np
import jax
import jax.numpy as jnp
from src.models.classical.gnn_baseline import GNNPredictor
from src.data.molecular_features import MolecularFeaturizer
from pathlib import Path


def main():
    # Load the featurizer
    featurizer = MolecularFeaturizer()

    # Load the classical model weights (best model)
    checkpoint_path = "experiments/checkpoints/classical_gnn/best.npz"
    d = np.load(checkpoint_path, allow_pickle=True)
    params = d["param_0"]  # Assuming param_0 contains the model parameters

    # Initialize the classical model (ensure it's the same as before)
    model = GNNPredictor(
        node_feat_dim=featurizer.node_feat_dim,
        edge_feat_dim=featurizer.edge_feat_dim,
        hidden_dim=128,
        num_layers=3,
        output_dim=1,
    )

    # Define a few SMILES strings for testing
    smiles_list = [
        "CCO",  # Ethanol
        "c1ccccc1",  # Benzene
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
    ]

    # Convert SMILES to graphs
    graphs = [featurizer.smiles_to_graph(smiles) for smiles in smiles_list]

    # Prepare the graphs for the model
    graph_dicts = [{
        "node_features": g.node_features,
        "edge_index": g.edge_index,
        "edge_features": g.edge_features
    } for g in graphs]

    # Test the classical model predictions
    for graph, smiles in zip(graph_dicts, smiles_list):
        prediction = model.predict(params, graph)
        print(f"Prediction for {smiles}: {prediction}")


if __name__ == "__main__":
    main()