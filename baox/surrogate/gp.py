import jax.numpy as jnp
import jax
import optax
from jax.scipy.linalg import cho_solve, cholesky
from jax import grad, lax
from typing import Callable, Optional, Tuple
from baox.surrogate.kernel import KernelBase

class GaussianProcess:
    def __init__(self, kernel: KernelBase, mean_function: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None, noise: float = 1e-3):
        """
        Gaussian Process regression with hyperparameter optimization using Adam + lax.scan.

        :param kernel: Kernel function.
        :param mean_function: Mean function, default is zero.
        :param noise: Initial noise level.
        """
        self.kernel = kernel
        self.mean_function = mean_function if mean_function else lambda x: jnp.zeros(x.shape[0])
        self.noise = noise
        self.X_train: Optional[jnp.ndarray] = None
        self.y_train: Optional[jnp.ndarray] = None
        self.K_inv: Optional[jnp.ndarray] = None

    def fit(self, X: jnp.ndarray, y: jnp.ndarray, lr: float = 0.02, steps: int = 500) -> None:
        """
        Train GP and optimize hyperparameters using Adam and `lax.scan` for speed.

        :param X: Training inputs.
        :param y: Training outputs.
        :param lr: Learning rate for Adam optimizer.
        :param steps: Number of optimization steps.
        """
        self.X_train = X
        self.y_train = y

        # Initialize parameters in log-space for positivity constraints
        params = jnp.log(jnp.array([self.kernel.lengthscale, self.kernel.variance, self.noise]))

        # Define Adam optimizer
        optimizer = optax.adam(lr)
        opt_state = optimizer.init(params)

        # Define negative log marginal likelihood loss
        def loss(log_params: jnp.ndarray) -> float:
            lengthscale, variance, noise = jnp.exp(log_params)  # Convert to positive values
            self.kernel.lengthscale, self.kernel.variance = lengthscale, variance

            K = self.kernel(X, X) + noise * jnp.eye(X.shape[0]) + 1e-6 * jnp.eye(X.shape[0])  # Add jitter
            L = cholesky(K, lower=True)

            alpha = cho_solve((L, True), y)
            log_det_K = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))

            return 0.5 * y.T @ alpha + 0.5 * log_det_K

        # Compute gradients of loss
        loss_grad = grad(loss)

        # Function for one step of optimization
        def step(state, _):
            params, opt_state = state
            grads = loss_grad(params)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), None

        # Run optimization using lax.scan
        (params, _), _ = lax.scan(step, (params, opt_state), None, length=steps)

        # Update final hyperparameters
        self.kernel.lengthscale, self.kernel.variance, self.noise = jnp.exp(params)

        # Compute final kernel inversion
        K = self.kernel(X, X) + self.noise * jnp.eye(X.shape[0]) + 1e-6 * jnp.eye(X.shape[0])  # Ensure stability
        L = cholesky(K, lower=True)
        self.K_inv = cho_solve((L, True), jnp.eye(X.shape[0]))

    def predict(self, X_test: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Predict using the trained Gaussian Process.

        :param X_test: Test inputs.
        :return: Mean and variance predictions.
        """
        X_test = X_test.reshape(-1, 1)
        K_s = self.kernel(self.X_train, X_test)
        K_ss = self.kernel(X_test, X_test) + self.noise * jnp.eye(X_test.shape[0]) + 1e-6 * jnp.eye(X_test.shape[0])

        mean_train = self.mean_function(self.X_train)
        mean_test = self.mean_function(X_test)

        y_train_centered = self.y_train - mean_train
        mu_s = mean_test + K_s.T @ self.K_inv @ y_train_centered

        cov_s = K_ss - K_s.T @ self.K_inv @ K_s
        var_s = jnp.clip(jnp.diag(cov_s), a_min=1e-9, a_max=None)  # Avoid negative variance

        return mu_s, var_s
