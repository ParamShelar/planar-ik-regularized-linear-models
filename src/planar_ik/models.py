"""Linear and ridge models used in the planar IK study."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _with_bias(X: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.column_stack([np.ones(len(X)), X])


def fit_linear(
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Fit ordinary least squares with a prepended bias column.

    Parameters
    ----------
    X:
        Design matrix without a bias column.
    Y:
        Regression targets.

    Returns
    -------
    ndarray
        Weight matrix including the intercept in row zero.
    """
    Xb = _with_bias(X)
    W, *_ = np.linalg.lstsq(Xb, Y, rcond=None)
    return W


def predict_linear(
    W: NDArray[np.float64],
    X: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Predict with an OLS weight matrix that includes a bias row.

    Parameters
    ----------
    W:
        Weight matrix returned by :func:`fit_linear`.
    X:
        Design matrix without a bias column.

    Returns
    -------
    ndarray
        Predicted targets.
    """
    Xb = _with_bias(X)
    return Xb @ W


def fit_ridge(
    Phi: NDArray[np.float64],
    Y: NDArray[np.float64],
    lam: float,
) -> NDArray[np.float64]:
    """Fit closed-form ridge regression with an unpenalized intercept.

    Parameters
    ----------
    Phi:
        Design matrix without a bias column.
    Y:
        Regression targets.
    lam:
        Nonnegative ridge penalty.

    Returns
    -------
    ndarray
        Weight matrix including the intercept in row zero.
    """
    if lam < 0:
        raise ValueError("lam must be nonnegative")

    Phi_b = _with_bias(Phi)
    n_features = Phi_b.shape[1]
    I = np.eye(n_features)
    I[0, 0] = 0.0

    A = Phi_b.T @ Phi_b + lam * I
    b = Phi_b.T @ Y
    W = np.linalg.solve(A, b)
    return W


def predict_ridge(
    W: NDArray[np.float64],
    Phi: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Predict with a ridge weight matrix that includes a bias row.

    Parameters
    ----------
    W:
        Weight matrix returned by :func:`fit_ridge`.
    Phi:
        Design matrix without a bias column.

    Returns
    -------
    ndarray
        Predicted targets.
    """
    Phi_b = _with_bias(Phi)
    return Phi_b @ W


class TorchRidge:
    """Closed-form ridge regression implemented with PyTorch tensors."""

    @staticmethod
    def fit(Phi: object, Y: object, lam: float) -> object:
        """Fit ridge regression on tensors with an unpenalized intercept.

        Parameters
        ----------
        Phi:
            PyTorch tensor design matrix without a bias column.
        Y:
            PyTorch tensor targets.
        lam:
            Nonnegative ridge penalty.

        Returns
        -------
        torch.Tensor
            Weight tensor including the intercept in row zero.
        """
        if lam < 0:
            raise ValueError("lam must be nonnegative")
        try:
            import torch
        except ImportError as exc:
            raise ImportError("TorchRidge requires PyTorch to be installed") from exc

        ones = torch.ones((Phi.shape[0], 1), dtype=Phi.dtype, device=Phi.device)
        Phi_b = torch.cat([ones, Phi], dim=1)
        I = torch.eye(Phi_b.shape[1], dtype=Phi.dtype, device=Phi.device)
        I[0, 0] = 0.0
        A = Phi_b.T @ Phi_b + lam * I
        b = Phi_b.T @ Y
        return torch.linalg.solve(A, b)

    @staticmethod
    def predict(W: object, Phi: object) -> object:
        """Predict from a closed-form PyTorch ridge fit.

        Parameters
        ----------
        W:
            Weight tensor including the intercept in row zero.
        Phi:
            PyTorch tensor design matrix without a bias column.

        Returns
        -------
        torch.Tensor
            Predicted targets.
        """
        try:
            import torch
        except ImportError as exc:
            raise ImportError("TorchRidge requires PyTorch to be installed") from exc

        ones = torch.ones((Phi.shape[0], 1), dtype=Phi.dtype, device=Phi.device)
        Phi_b = torch.cat([ones, Phi], dim=1)
        return Phi_b @ W


class TorchRidgeGD:
    """Gradient-descent ridge regression for parity experiments."""

    def __init__(
        self,
        lam: float,
        lr: float = 1e-2,
        steps: int = 1000,
    ) -> None:
        """Create a gradient-descent ridge trainer.

        Parameters
        ----------
        lam:
            Nonnegative ridge penalty.
        lr:
            Learning rate.
        steps:
            Number of optimization steps.
        """
        if lam < 0:
            raise ValueError("lam must be nonnegative")
        self.lam = lam
        self.lr = lr
        self.steps = steps
        self.W: object | None = None

    def fit(self, Phi: object, Y: object) -> object:
        """Fit ridge regression with gradient descent.

        Parameters
        ----------
        Phi:
            PyTorch tensor design matrix without a bias column.
        Y:
            PyTorch tensor targets.

        Returns
        -------
        torch.Tensor
            Learned weight tensor including the intercept in row zero.
        """
        try:
            import torch
        except ImportError as exc:
            raise ImportError("TorchRidgeGD requires PyTorch to be installed") from exc

        ones = torch.ones((Phi.shape[0], 1), dtype=Phi.dtype, device=Phi.device)
        Phi_b = torch.cat([ones, Phi], dim=1)
        W = torch.zeros(
            (Phi_b.shape[1], Y.shape[1]),
            dtype=Phi.dtype,
            device=Phi.device,
            requires_grad=True,
        )
        optimizer = torch.optim.SGD([W], lr=self.lr)

        for _ in range(self.steps):
            optimizer.zero_grad()
            residual = Phi_b @ W - Y
            penalty = torch.sum(W[1:] ** 2)
            loss = torch.mean(residual**2) + self.lam * penalty / len(Phi_b)
            loss.backward()
            optimizer.step()

        self.W = W.detach()
        return self.W

    def predict(self, Phi: object) -> object:
        """Predict with the fitted gradient-descent model.

        Parameters
        ----------
        Phi:
            PyTorch tensor design matrix without a bias column.

        Returns
        -------
        torch.Tensor
            Predicted targets.
        """
        if self.W is None:
            raise RuntimeError("TorchRidgeGD must be fit before predict")
        return TorchRidge.predict(self.W, Phi)


def assert_torch_numpy_ridge_parity(
    seed: int = 123,
    n_samples: int = 200,
    n_features: int = 30,
    lam: float = 0.37,
    atol: float = 1e-6,
) -> bool:
    """Assert that PyTorch closed-form ridge matches NumPy ridge.

    Parameters
    ----------
    seed:
        Random seed for the parity problem.
    n_samples:
        Number of random rows.
    n_features:
        Number of random features.
    lam:
        Ridge penalty.
    atol:
        Absolute tolerance for the comparison.

    Returns
    -------
    bool
        ``True`` when the assertion passes.
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required for the parity check") from exc

    rng = np.random.default_rng(seed)
    Phi = rng.normal(size=(n_samples, n_features))
    Y = rng.normal(size=(n_samples, 2))
    W_np = fit_ridge(Phi, Y, lam)

    Phi_t = torch.as_tensor(Phi, dtype=torch.float64)
    Y_t = torch.as_tensor(Y, dtype=torch.float64)
    W_t = TorchRidge.fit(Phi_t, Y_t, lam).detach().cpu().numpy()

    if not np.allclose(W_np, W_t, atol=atol):
        raise AssertionError("PyTorch ridge does not match NumPy ridge")
    return True
