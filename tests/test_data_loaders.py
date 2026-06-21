#tests/test_data_loaders.py
"""
Tests for data loading and preprocessing pipeline.
"""

import pytest
import numpy as np
import jax.numpy as jnp


def test_molecular_featurizer_init():
    """Test MolecularFeaturizer initialization."""
    from src.data.molecular_features import MolecularFeaturizer
    feat = MolecularFeaturizer()
    assert feat.node_feat_dim == 33
    assert feat.edge_feat_dim == 6


def test_smiles_to_graph():
    """Test SMILES to graph conversion."""
    from src.data.molecular_features import MolecularFeaturizer
    feat = MolecularFeaturizer()

    # Valid SMILES
    graph = feat.smiles_to_graph("CCO")
    assert graph is not None
    assert graph.num_nodes == 3  # C, C, O (without H)
    assert graph.node_features.shape[1] == feat.node_feat_dim
    assert graph.edge_features.shape[1] == feat.edge_feat_dim

    # Benzene
    graph = feat.smiles_to_graph("c1ccccc1")
    assert graph is not None
    assert graph.num_nodes == 6


def test_invalid_smiles():
    """Test invalid SMILES returns None."""
    from src.data.molecular_features import MolecularFeaturizer
    feat = MolecularFeaturizer()
    assert feat.smiles_to_graph("INVALID") is None


def test_batch_smiles_to_graphs():
    """Test batch featurization."""
    from src.data.molecular_features import MolecularFeaturizer
    feat = MolecularFeaturizer()
    smiles = ["CCO", "c1ccccc1", "CC(=O)O"]
    graphs = feat.batch_smiles_to_graphs(smiles, show_progress=False)
    assert len(graphs) == 3
    assert all(g is not None for g in graphs)


def test_molecular_fingerprints():
    """Test Morgan fingerprint generation."""
    from src.data.molecular_features import MolecularFingerprints
    fp = MolecularFingerprints.morgan_fingerprint("CCO")
    assert fp is not None
    assert fp.shape == (2048,)
    assert fp.dtype == np.float32


def test_molecular_descriptors():
    """Test RDKit descriptor extraction."""
    from src.data.molecular_features import MolecularFingerprints
    desc = MolecularFingerprints.molecular_descriptors("CCO")
    assert desc is not None
    assert desc.shape == (10,)


def test_quantum_encoder():
    """Test quantum encoding pipeline."""
    from src.data.quantum_encoding import QuantumEncoder
    encoder = QuantumEncoder(n_qubits=4)

    features = np.random.randn(100, 20)
    encoder.fit(features)
    assert encoder.is_fitted

    encoded = encoder.transform(features)
    assert encoded.shape == (100, 4)

    # Check range [0, pi]
    assert float(jnp.min(encoded)) >= 0.0
    assert float(jnp.max(encoded)) <= np.pi + 0.01


def test_quantum_encoder_single_sample():
    """Test quantum encoding on single sample."""
    from src.data.quantum_encoding import QuantumEncoder
    encoder = QuantumEncoder(n_qubits=4)
    features = np.random.randn(50, 20)
    encoder.fit(features)

    single = features[0]
    encoded = encoder.transform(single)
    assert encoded.shape == (4,)


def test_graph_to_vector():
    """Test graph-to-vector aggregation."""
    from src.data.quantum_encoding import GraphToVectorFeatures
    graph = {
        'node_features': np.random.randn(5, 33),
        'edge_index': np.array([[0, 1], [1, 0]]),
        'edge_features': np.random.randn(2, 6)
    }
    vec = GraphToVectorFeatures.aggregate_node_features(graph)
    # mean + max + min + std = 4 * 33 = 132
    assert vec.shape == (4 * 33,)


def test_collate_graphs():
    """Test graph batching/padding."""
    from src.data.preprocessors import collate_graphs

    graphs = [
        {
            'node_features': jnp.ones((3, 10)),
            'edge_index': jnp.array([[0, 1], [1, 0]]),
            'edge_features': jnp.ones((2, 4)),
        },
        {
            'node_features': jnp.ones((5, 10)),
            'edge_index': jnp.array([[0, 1, 2, 3], [1, 2, 3, 4]]),
            'edge_features': jnp.ones((4, 4)),
        },
    ]
    targets = jnp.array([1.0, 2.0])

    batched, tgt = collate_graphs(graphs, targets)
    assert batched.batch_size == 2
    assert batched.max_nodes == 5
    assert batched.node_features.shape == (2, 5, 10)
    assert batched.node_mask.shape == (2, 5)
    assert float(batched.node_mask[0, 2]) == 1.0  # 3rd node of 1st graph
    assert float(batched.node_mask[0, 3]) == 0.0  # Padding


def test_smiles_augmentation():
    """Test SMILES augmentation."""
    from src.data.preprocessors import augment_smiles
    augmented = augment_smiles("CCO", n_augmentations=5)
    assert len(augmented) >= 1
    assert "CCO" in augmented or len(augmented) > 0
