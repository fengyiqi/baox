import jax.numpy as jnp
from jax.scipy.linalg import cho_solve, cholesky
from jax import grad
from typing import Callable, Optional, Tuple
from baox.surrogate.kernel import KernelBase

class GaussianProcess:
    def __init__(self, kernel: KernelBase, mean_function: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None, noise: float = 1e-3):
        self.kernel = kernel
        self.mean_function = mean_function if mean_function else lambda x: jnp.zeros(
            x.shape[0])
        self.noise = noise
        self.X_train: Optional[jnp.ndarray] = None
        self.y_train: Optional[jnp.ndarray] = None
        self.K_inv: Optional[jnp.ndarray] = None

    def fit(self, X: jnp.ndarray, y: jnp.ndarray, lr: float = 0.01, steps: int = 50) -> None:
        self.X_train = X
        self.y_train = y
        if isinstance(self.noise, float):
            noise = self.noise * jnp.eye(X.shape[0])
        else:
            noise = jnp.diag(self.noise)

        def loss(params: jnp.ndarray, X: jnp.ndarray, y: jnp.ndarray, kernel: KernelBase) -> float:
            # Assign params to kernel explicitly
            lengthscale, variance = jnp.exp(params)
            kernel.lengthscale, kernel.variance = lengthscale, variance

            # Compute kernel matrix
            K = kernel(X, X) + noise
            L = cholesky(K, lower=True)
            
            # Solve for K_inv * y efficiently
            alpha = cho_solve((L, True), y)

            # Log determinant using Cholesky for better stability
            log_det_K = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))

            # Compute negative log marginal likelihood
            return 0.5 * y.T @ alpha + 0.5 * log_det_K

        grad_loss = grad(loss)
        params = jnp.array([jnp.log(self.kernel.lengthscale), jnp.log(self.kernel.variance)])
        for _ in range(steps):
            params -= lr * grad_loss(params, self.X_train, self.y_train, self.kernel)
        self.kernel.lengthscale, self.kernel.variance = jnp.exp(params)

        K = self.kernel(X, X) + self.noise * jnp.eye(X.shape[0])
        L = cholesky(K, lower=True)
        self.K_inv = cho_solve((L, True), jnp.eye(X.shape[0]))

    def predict(self, X_test: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        K_s = self.kernel(self.X_train, X_test)
        K_ss = self.kernel(X_test, X_test) + self.noise * \
            jnp.eye(X_test.shape[0])

        mean_train = self.mean_function(self.X_train)
        mean_test = self.mean_function(X_test)

        y_train_centered = self.y_train - mean_train
        mu_s = mean_test + K_s.T @ self.K_inv @ y_train_centered

        v = cho_solve((cholesky(K_ss, lower=True), True), K_s.T)
        cov_s = K_ss - K_s.T @ self.K_inv @ K_s

        return mu_s, jnp.diag(cov_s)