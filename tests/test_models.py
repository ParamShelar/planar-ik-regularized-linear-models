import numpy as np
import pytest

from planar_ik.models import TorchRidge, fit_linear, fit_ridge


def test_torch_numpy_ridge_parity():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(123)
    Phi = rng.normal(size=(200, 30))
    Y = rng.normal(size=(200, 2))
    lam = 0.37

    W_np = fit_ridge(Phi, Y, lam)
    W_torch = TorchRidge.fit(
        torch.as_tensor(Phi, dtype=torch.float64),
        torch.as_tensor(Y, dtype=torch.float64),
        lam,
    )

    assert np.allclose(W_np, W_torch.detach().cpu().numpy(), atol=1e-6)

    Phi_b = np.column_stack([np.ones(len(Phi)), Phi])
    I = np.eye(Phi_b.shape[1])
    I[0, 0] = 0.0
    W_manual = np.linalg.solve(Phi_b.T @ Phi_b + lam * I, Phi_b.T @ Y)
    assert np.allclose(W_np, W_manual, atol=1e-12)


def test_ridge_lambda_zero_matches_ols():
    rng = np.random.default_rng(456)
    Phi = rng.normal(size=(300, 5))
    true_W = rng.normal(size=(6, 2))
    Phi_b = np.column_stack([np.ones(len(Phi)), Phi])
    Y = Phi_b @ true_W

    W_ols = fit_linear(Phi, Y)
    W_ridge_zero = fit_ridge(Phi, Y, lam=0.0)

    assert np.allclose(W_ridge_zero, W_ols, atol=1e-6)


def test_ridge_bias_column_not_penalized_constant_target():
    rng = np.random.default_rng(789)
    Phi = rng.normal(size=(200, 8))
    constant = np.array([1.75, -0.5])
    Y = np.tile(constant, (len(Phi), 1))

    W = fit_ridge(Phi, Y, lam=1e6)

    assert np.allclose(W[0], constant, atol=1e-10)
    assert np.linalg.norm(W[1:]) < 1e-8
