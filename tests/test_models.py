#tests/test_models.py
"""
Tests for model architectures: GNN, GAT, MPNN, Quantum, Hybrid.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np


def _dummy_graph(node_dim=33, edge_dim=6, n_nodes=10, n_edges=3):
    """Create a dummy graph dict for testing."""
    return {
        "node_features": jnp.ones((n_nodes, node_dim)),
        "edge_index": jnp.array([[0, 1, 2], [1, 2, 0]])[:, :n_edges],
        "edge_features": jnp.ones((n_edges, edge_dim)),
    }


# =========================================================
# GNN Baseline
# =========================================================

def test_gnn_predictor_init():
    """Test GNN predictor initialization."""
    from src.models.classical.gnn_baseline import GNNPredictor
    model = GNNPredictor(node_feat_dim=33, edge_feat_dim=6, hidden_dim=64, num_layers=2)
    dummy = _dummy_graph()
    params = model.init_params(dummy)
    assert params is not None
    assert model.count_parameters() > 0


def test_gnn_forward_pass():
    """Test GNN forward pass produces scalar output."""
    from src.models.classical.gnn_baseline import GNNPredictor
    model = GNNPredictor(node_feat_dim=33, edge_feat_dim=6, hidden_dim=64, num_layers=2)
    dummy = _dummy_graph()
    params = model.init_params(dummy)
    pred = model.predict(params, dummy)
    assert pred.shape == ()  # scalar
    assert jnp.isfinite(pred)


def test_gnn_gradient_flow():
    """Test that gradients flow through the GNN."""
    from src.models.classical.gnn_baseline import GNNPredictor
    model = GNNPredictor(node_feat_dim=33, edge_feat_dim=6, hidden_dim=64, num_layers=2)
    dummy = _dummy_graph()
    params = model.init_params(dummy)

    def loss_fn(p):
        return model.forward(p, dummy, training=False, rngs={}) ** 2

    grads = jax.grad(loss_fn)(params)
    flat_grads = jax.tree_util.tree_leaves(grads)
    total_grad_norm = sum(float(jnp.linalg.norm(g)) for g in flat_grads)
    assert total_grad_norm > 0  # Gradients are non-zero


# =========================================================
# GAT
# =========================================================

def test_gat_predictor_init():
    """Test GAT predictor initialization."""
    from src.models.classical.attention_gnn import GATPredictor
    model = GATPredictor(
        node_feat_dim=33, edge_feat_dim=6,
        hidden_dim=64, num_layers=2, num_heads=4, head_dim=16
    )
    dummy = _dummy_graph()
    params = model.init_params(dummy)
    assert params is not None
    assert model.count_parameters() > 0


def test_gat_forward_pass():
    """Test GAT forward pass."""
    from src.models.classical.attention_gnn import GATPredictor
    model = GATPredictor(
        node_feat_dim=33, edge_feat_dim=6,
        hidden_dim=64, num_layers=2, num_heads=4, head_dim=16
    )
    dummy = _dummy_graph()
    params = model.init_params(dummy)
    pred = model.predict(params, dummy)
    assert pred.shape == ()
    assert jnp.isfinite(pred)


# =========================================================
# MPNN
# =========================================================

def test_mpnn_predictor_init():
    """Test MPNN predictor initialization."""
    from src.models.classical.mpnn import MPNNPredictor
    model = MPNNPredictor(
        node_feat_dim=33, edge_feat_dim=6,
        hidden_dim=64, num_steps=2, set2set_iterations=2
    )
    dummy = _dummy_graph()
    params = model.init_params(dummy)
    assert params is not None
    assert model.count_parameters() > 0


def test_mpnn_forward_pass():
    """Test MPNN forward pass."""
    from src.models.classical.mpnn import MPNNPredictor
    model = MPNNPredictor(
        node_feat_dim=33, edge_feat_dim=6,
        hidden_dim=64, num_steps=2, set2set_iterations=2
    )
    dummy = _dummy_graph()
    params = model.init_params(dummy)
    pred = model.predict(params, dummy)
    assert pred.shape == ()
    assert jnp.isfinite(pred)


# =========================================================
# Quantum
# =========================================================

def test_quantum_circuit_init():
    """Test VQC initialization."""
    from src.models.quantum.quantum_circuits import VariationalQuantumCircuit
    vqc = VariationalQuantumCircuit(n_qubits=4, n_layers=2)
    assert vqc.n_params == 2 * 2 * 4  # layers * 2 * qubits


def test_quantum_circuit_forward():
    """Test VQC forward pass."""
    from src.models.quantum.quantum_circuits import QuantumNeuralNetwork
    qnn = QuantumNeuralNetwork(n_qubits=4, n_layers=2, output_dim=1)
    params = qnn.initialize_parameters()
    x = jnp.array([0.5, 1.0, 1.5, 2.0])
    pred = qnn.forward(params, x)
    assert pred.shape == ()
    assert jnp.isfinite(pred)


def test_quantum_parameter_count():
    """Test quantum parameter counting."""
    from src.models.quantum.quantum_circuits import QuantumNeuralNetwork
    qnn = QuantumNeuralNetwork(n_qubits=4, n_layers=2, output_dim=1)
    # Circuit: 2 layers * 2*4 = 16 + readout: 1*4 + 1 = 5 = 21
    assert qnn.count_parameters() == 21


# =========================================================
# Hybrid
# =========================================================

def test_hybrid_predictor_init():
    """Test hybrid model initialization."""
    from src.models.hybrid.hybrid_model import HybridRegressor
    model = HybridRegressor(
        node_feat_dim=33, edge_feat_dim=6,
        gnn_hidden_dim=64, gnn_layers=2,
        n_qubits=4, quantum_layers=2
    )
    dummy = _dummy_graph()
    params = model.init_params(dummy)
    assert params is not None
    assert model.count_parameters() > 0


def test_hybrid_forward_pass():
    """Test hybrid model forward pass."""
    from src.models.hybrid.hybrid_model import HybridRegressor
    model = HybridRegressor(
        node_feat_dim=33, edge_feat_dim=6,
        gnn_hidden_dim=64, gnn_layers=2,
        n_qubits=4, quantum_layers=2
    )
    dummy = _dummy_graph()
    params = model.init_params(dummy)
    pred = model.predict(params, dummy)
    assert pred.shape == ()
    assert jnp.isfinite(pred)


# =========================================================
# Hybrid V2
# =========================================================

def test_hybrid_v2_init():
    """Test Hybrid V2 model initialization."""
    from src.models.hybrid.hybrid_model import HybridV2Regressor
    model = HybridV2Regressor(
        node_feat_dim=33, edge_feat_dim=6,
        gnn_hidden_dim=64, gnn_layers=2,
        n_qubits=4, quantum_layers=2, fusion_dim=32
    )
    dummy = _dummy_graph()
    params = model.init_params(dummy)
    assert params is not None
    assert model.count_parameters() > 0


def test_hybrid_v2_forward():
    """Test Hybrid V2 forward pass."""
    from src.models.hybrid.hybrid_model import HybridV2Regressor
    model = HybridV2Regressor(
        node_feat_dim=33, edge_feat_dim=6,
        gnn_hidden_dim=64, gnn_layers=2,
        n_qubits=4, quantum_layers=2, fusion_dim=32
    )
    dummy = _dummy_graph()
    params = model.init_params(dummy)
    pred = model.predict(params, dummy)
    assert pred.shape == ()
    assert jnp.isfinite(pred)
