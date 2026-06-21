#tests/test_training.py
"""
Tests for training infrastructure: losses, metrics, callbacks, trainers.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np


# =========================================================
# Loss Functions
# =========================================================

def test_mse_loss():
    from src.training.losses import mse_loss
    preds = jnp.array([1.0, 2.0, 3.0])
    targets = jnp.array([1.0, 2.0, 3.0])
    assert float(mse_loss(preds, targets)) == pytest.approx(0.0)


def test_mae_loss():
    from src.training.losses import mae_loss
    preds = jnp.array([1.0, 2.0, 3.0])
    targets = jnp.array([1.5, 2.5, 3.5])
    assert float(mae_loss(preds, targets)) == pytest.approx(0.5)


def test_huber_loss():
    from src.training.losses import huber_loss
    preds = jnp.array([0.0])
    targets = jnp.array([0.5])
    loss = float(huber_loss(preds, targets, delta=1.0))
    assert loss > 0.0


def test_smooth_l1_loss():
    from src.training.losses import smooth_l1_loss
    preds = jnp.array([0.0])
    targets = jnp.array([0.5])
    loss = float(smooth_l1_loss(preds, targets))
    assert loss > 0.0


def test_multi_task_loss():
    from src.training.losses import multi_task_loss
    preds = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    targets = jnp.array([[1.1, 2.1], [3.1, 4.1]])
    log_sigmas = jnp.array([0.0, 0.0])
    loss = float(multi_task_loss(preds, targets, log_sigmas))
    assert loss > 0.0


def test_get_loss_fn():
    from src.training.losses import get_loss_fn
    fn = get_loss_fn("mse")
    assert callable(fn)

    with pytest.raises(ValueError):
        get_loss_fn("nonexistent_loss")


# =========================================================
# Metrics
# =========================================================

def test_metrics_mae():
    from src.training.metrics import mae
    assert mae(jnp.array([1.0, 2.0]), jnp.array([1.5, 2.5])) == pytest.approx(0.5)


def test_metrics_rmse():
    from src.training.metrics import rmse
    result = rmse(jnp.array([1.0, 2.0]), jnp.array([1.0, 2.0]))
    assert result == pytest.approx(0.0, abs=1e-6)


def test_metrics_r2():
    from src.training.metrics import r2_score
    # Perfect predictions
    r2 = r2_score(jnp.array([1.0, 2.0, 3.0]), jnp.array([1.0, 2.0, 3.0]))
    assert r2 == pytest.approx(1.0, abs=1e-5)


def test_metrics_pearson():
    from src.training.metrics import pearson_correlation
    corr = pearson_correlation(
        jnp.array([1.0, 2.0, 3.0]),
        jnp.array([1.0, 2.0, 3.0])
    )
    assert corr == pytest.approx(1.0, abs=1e-5)


def test_compute_all_metrics():
    from src.training.metrics import compute_all_metrics
    preds = jnp.array([1.0, 2.0, 3.0])
    targets = jnp.array([1.1, 2.1, 3.1])
    metrics = compute_all_metrics(preds, targets)
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics
    assert "pearson" in metrics


# =========================================================
# Callbacks
# =========================================================

def test_early_stopping():
    from src.training.callbacks import EarlyStopping
    es = EarlyStopping(patience=3, mode="min")

    assert es(1.0, 0) is False  # Improvement
    assert es(0.9, 1) is False  # Improvement
    assert es(1.0, 2) is False  # No improvement (1/3)
    assert es(1.0, 3) is False  # No improvement (2/3)
    assert es(1.0, 4) is True   # No improvement (3/3) -> stop


def test_gradient_monitor():
    from src.training.callbacks import GradientMonitor
    monitor = GradientMonitor()

    dummy_grads = {"w": jnp.array([1.0, 2.0, 3.0])}
    stats = monitor(dummy_grads)
    assert "grad_norm" in stats
    assert stats["grad_norm"] > 0


def test_lr_scheduler():
    from src.training.callbacks import LRScheduler
    scheduler = LRScheduler(schedule_type="cosine", init_lr=0.001, decay_steps=100)
    lr0 = scheduler.get_lr()
    assert lr0 == pytest.approx(0.001)

    for _ in range(50):
        scheduler.step()
    lr50 = scheduler.get_lr()
    assert lr50 < lr0  # Should have decayed


# =========================================================
# Config
# =========================================================

def test_config_loader():
    from src.utils.config_loader import ExperimentConfig, DataConfig, ModelConfig, TrainingConfig
    config = ExperimentConfig()
    assert config.data.dataset == "qm9"
    assert config.model.model_type == "gnn"
    assert config.training.learning_rate == 1e-3


def test_config_to_dict():
    from src.utils.config_loader import ExperimentConfig
    config = ExperimentConfig(name="test")
    d = config.to_dict()
    assert d["name"] == "test"
    assert "data" in d
    assert "model" in d
    assert "training" in d


# =========================================================
# Reproducibility
# =========================================================

def test_set_global_seed():
    from src.utils.reproducibility import set_global_seed
    key = set_global_seed(42)
    assert key is not None


def test_deterministic_splits():
    from src.utils.reproducibility import get_deterministic_split_indices
    train1, val1, test1 = get_deterministic_split_indices(100, seed=42)
    train2, val2, test2 = get_deterministic_split_indices(100, seed=42)
    np.testing.assert_array_equal(train1, train2)
    np.testing.assert_array_equal(val1, val2)


# =========================================================
# Ensemble
# =========================================================

def test_ensemble_basic():
    from src.models.hybrid.ensemble import ModelEnsemble
    ensemble = ModelEnsemble()
    ensemble.add_model("m1", lambda g: 1.0, weight=0.5)
    ensemble.add_model("m2", lambda g: 2.0, weight=0.5)
    pred = ensemble.predict({})
    assert pred == pytest.approx(1.5, abs=0.01)


def test_ensemble_uncertainty():
    from src.models.hybrid.ensemble import ModelEnsemble
    ensemble = ModelEnsemble()
    ensemble.add_model("m1", lambda g: 1.0, weight=0.5)
    ensemble.add_model("m2", lambda g: 3.0, weight=0.5)
    mean, unc = ensemble.predict_with_uncertainty({})
    assert mean == pytest.approx(2.0, abs=0.01)
    assert unc > 0  # There is disagreement
