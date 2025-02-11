import jax.numpy as jnp
import jax
import optax
from jax.scipy.linalg import cho_solve, cholesky
from jax import grad, lax
from typing import Callable, Optional, Tuple
from baox.surrogate.kernel import KernelBase

class AutoRegressiveMFGP:
    def __init__(self, kernel_low: KernelBase, kernel_delta: KernelBase):
        """
        Auto-Regressive Multi-Fidelity Gaussian Process.

        :param kernel_low: Kernel for low-fidelity function.
        :param kernel_delta: Kernel for discrepancy function.
        """
        self.kernel_low = kernel_low  # k_l(x, x')
        self.kernel_delta = kernel_delta  # k_delta(x, x')
        self.X_low, self.y_low = None, None
        self.X_high, self.y_high = None, None
        self.K_inv_low, self.K_inv_delta = None, None
        self.rho = 1.0  # Correlation parameter

    def fit(self, X_low: jnp.ndarray, y_low: jnp.ndarray, X_high: jnp.ndarray, y_high: jnp.ndarray, lr: float = 0.05, steps: int = 500):
        """
        Train the AR-MFGP model.

        :param X_low: Low-fidelity inputs.
        :param y_low: Low-fidelity outputs.
        :param X_high: High-fidelity inputs.
        :param y_high: High-fidelity outputs.
        :param lr: Learning rate for Adam optimizer.
        :param steps: Number of optimization steps.
        """
        self.X_low, self.y_low = X_low, y_low
        self.X_high, self.y_high = X_high, y_high

        # Initialize parameters (log-space for positivity)
        params = jnp.log(jnp.array([
            self.kernel_low.lengthscale, self.kernel_low.variance, 1e-3,  # Low-fidelity kernel
            self.kernel_delta.lengthscale, self.kernel_delta.variance, 1e-3,  # Discrepancy kernel
            1.0  # Correlation parameter rho
        ]))

        # Adam optimizer
        optimizer = optax.adam(lr)
        opt_state = optimizer.init(params)

        def loss(log_params: jnp.ndarray) -> float:
            """
            Compute the negative log marginal likelihood.
            """
            # Extract parameters
            low_lengthscale, low_variance, low_noise, delta_lengthscale, delta_variance, delta_noise, rho = jnp.exp(log_params)

            # Update kernel parameters
            self.kernel_low.lengthscale, self.kernel_low.variance = low_lengthscale, low_variance
            self.kernel_delta.lengthscale, self.kernel_delta.variance = delta_lengthscale, delta_variance
            self.rho = rho

            # Compute covariance matrices
            K_low = self.kernel_low(X_low, X_low) + low_noise * jnp.eye(X_low.shape[0]) + 1e-6 * jnp.eye(X_low.shape[0])
            K_delta = self.kernel_delta(X_high, X_high) + delta_noise * jnp.eye(X_high.shape[0]) + 1e-6 * jnp.eye(X_high.shape[0])

            # Cholesky decomposition
            L_low = cholesky(K_low, lower=True)

            # Compute alpha for low-fidelity
            alpha_low = cho_solve((L_low, True), y_low)

            # Compute residuals for high-fidelity
            y_residual = y_high - rho * self.kernel_low(X_high, X_low) @ alpha_low

            # Solve for discrepancy process
            L_delta = cholesky(K_delta, lower=True)
            alpha_delta = cho_solve((L_delta, True), y_residual)

            # Compute negative log marginal likelihood
            log_det_K_low = 2.0 * jnp.sum(jnp.log(jnp.diag(L_low)))
            log_det_K_delta = 2.0 * jnp.sum(jnp.log(jnp.diag(L_delta)))

            return jnp.squeeze(
                0.5 * y_low.T @ alpha_low + 0.5 * log_det_K_low +
                0.5 * y_residual.T @ alpha_delta + 0.5 * log_det_K_delta
            )

        # Compute gradients
        loss_grad = grad(loss)

        # One step optimization
        def step(state, _):
            params, opt_state = state
            grads = loss_grad(params)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), None

        # Optimize
        (params, _), _ = lax.scan(step, (params, opt_state), None, length=steps)

        # Update final parameters
        (
            self.kernel_low.lengthscale, self.kernel_low.variance, low_noise,
            self.kernel_delta.lengthscale, self.kernel_delta.variance, delta_noise,
            self.rho
        ) = jnp.exp(params)

        # Compute inverses
        K_low = self.kernel_low(X_low, X_low) + low_noise * jnp.eye(X_low.shape[0]) + 1e-6 * jnp.eye(X_low.shape[0])
        L_low = cholesky(K_low, lower=True)
        self.K_inv_low = cho_solve((L_low, True), jnp.eye(X_low.shape[0]))

        K_delta = self.kernel_delta(X_high, X_high) + delta_noise * jnp.eye(X_high.shape[0]) + 1e-6 * jnp.eye(X_high.shape[0])
        L_delta = cholesky(K_delta, lower=True)
        self.K_inv_delta = cho_solve((L_delta, True), jnp.eye(X_high.shape[0]))


    def predict(self, X_test: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Predict using AR-MFGP.

        :param X_test: Test inputs.
        :return: Mean and variance.
        """
        # Compute covariance between training and test points
        K_s_low = self.kernel_low(self.X_low, X_test)  # Covariance between low-fidelity training and test points
        K_s_high = self.kernel_delta(self.X_high, X_test)  # Covariance between high-fidelity training and test points

        # Compute mean predictions for low-fidelity and discrepancy GPs
        mu_low = K_s_low.T @ self.K_inv_low @ self.y_low  # Low-fidelity prediction at test points

        # Compute discrepancy mean at high-fidelity training points
        mu_delta = self.kernel_delta(self.X_high, X_test).T @ self.K_inv_delta @ self.y_high
        mu_delta_train = self.kernel_delta(self.X_high, self.X_high).T @ self.K_inv_delta @ self.y_high

        # Compute discrepancy mean at test points
        # mu_delta = K_s_high.T @ self.K_inv_delta @ self.y_high


        # Compute the correction term using high-fidelity training data
        K_ss_high = self.kernel_delta(self.X_high, self.X_high) + 1e-6 * jnp.eye(self.X_high.shape[0])
        K_inv_high = jnp.linalg.inv(K_ss_high)  # Inverse of high-fidelity covariance matrix

        correction_term = K_s_high.T @ K_inv_high @ (self.y_high - self.rho * self.kernel_low(self.X_low, self.X_high).T @ self.K_inv_low @ self.y_low - mu_delta_train)

        # print(mu_low.shape, correction_term.shape, mu_delta_train.shape)
        mu_mf = self.rho * mu_low + mu_delta + correction_term

        # Compute prior variances at test points
        K_ss_low = self.kernel_low(X_test, X_test) + 1e-6 * jnp.eye(X_test.shape[0])  # Prior variance of low-fidelity GP
        K_ss_delta = self.kernel_delta(X_test, X_test)  # Corrected: full self-covariance, not just diagonal

        # Compute low-fidelity and discrepancy variances
        var_low = K_ss_low - K_s_low.T @ self.K_inv_low @ K_s_low  # Low-fidelity variance
        var_delta = K_ss_delta - K_s_high.T @ self.K_inv_delta @ K_s_high  # Discrepancy variance

        # Compute variance correction term
        var_correction = K_s_high.T @ K_inv_high @ K_s_high  # This term reduces uncertainty

        var_mf = self.rho**2 * var_low + K_ss_delta - var_correction  # Use full matrices, not just diag

        # Ensure positive variance
        var_mf = jnp.clip(jnp.diag(var_mf), a_min=1e-9, a_max=None)

        
        return mu_mf.squeeze(), var_mf



