import jax.numpy as jnp
import jax
import optax
from jax.scipy.linalg import cho_solve, cholesky
from jax import grad, lax
from typing import Callable, Optional, Tuple, Union, List
from baox.surrogate.kernel import KernelBase
from baox.data_types import MultiFidelityDataset, MFGPHyperparameters
import copy
import warnings

# prepare for MFDGP


class AutoRegressiveMFGP:
    """
    Auto-Regressive Multi-Fidelity Gaussian Process (MFGP) model.

    This model implements an auto-regressive framework for multi-fidelity Gaussian Process modeling.
    It leverages training data from multiple fidelity levels to improve prediction accuracy, where the
    high-fidelity data is modeled using corrections based on lower fidelities. The model requires a
    MultiFidelityDataset with at least two fidelity levels, a kernel or list of kernels for the fidelity
    levels, noise parameters, and rho parameters that quantify the correlation between fidelities.

    Attributes:
        dim (int): The input dimension of the training data.
        dataset (MultiFidelityDataset): The multi-fidelity dataset containing training data for each fidelity level.
        fidelities (int): The number of fidelity levels available in the dataset.
        kernels (List[KernelBase]): List of kernel functions, one for each fidelity level. If a single kernel
            is provided, the same kernel is used for all fidelity levels (with a warning).
        noises (jnp.ndarray): Array of noise parameters, one for each fidelity level. If a single noise parameter
            is provided, the same noise is used for all fidelity levels (with a warning).
        rhos (jnp.ndarray): Array of correlation parameters between fidelity levels. The length of this array
            must be equal to (fidelities - 1). If a single rho is provided, it is replicated for all transitions
            (with a warning).
        K_invs (dict): A dictionary to store the inverse kernel matrices for each fidelity level.
        trainable_params (MFGPHyperparameters): Hyperparameters for the MFGP model, including lengthscales,
            variances, noise parameters, and rho parameters.
    """

    def __init__(
        self,
        mf_dataset: MultiFidelityDataset,
        kernels: Optional[Union[KernelBase, List[KernelBase]]],
        noise: jnp.ndarray = 1e-2,
        rho: jnp.ndarray = 1.0
    ):
        """
        Initialize the Auto-Regressive Multi-Fidelity Gaussian Process model.

        Args:
            mf_dataset (MultiFidelityDataset): A multi-fidelity dataset containing training data for different fidelity levels.
            kernels (KernelBase or List[KernelBase]): A single kernel or a list of kernels to be used for each fidelity level.
                If a single kernel is provided, the same kernel is applied to all fidelity levels.
            noise (jnp.ndarray or float): Noise parameter(s) for the model. If a single float is provided, the same noise is used
                for all fidelity levels.
            rho (jnp.ndarray or float): Correlation parameter(s) between successive fidelity levels. For a multi-fidelity model
                with N fidelity levels, there should be N-1 rho parameters. If a single float is provided, the same rho is used
                for all transitions.

        Raises:
            ValueError: If the dataset contains less than two fidelity levels.
            ValueError: If the number of provided kernels does not match the number of fidelity levels.
            ValueError: If the number of provided noise parameters does not match the number of fidelity levels.
            ValueError: If the number of provided rho parameters is not equal to the number of fidelity levels minus one.

        Notes:
            - If a single kernel or noise parameter is provided, a warning is issued indicating that the same value is used for all fidelity levels.
            - The model stores the inverse kernel matrices in the attribute `K_invs` and the initial trainable hyperparameters
              in `trainable_params`.
        """

        self.dim = mf_dataset.get_data(0).x_train.shape[1]
        self.dataset = mf_dataset
        self.fidelities = len(mf_dataset.list_fidelities())
        if self.fidelities == 1:
            raise ValueError(
                "MultiFidelityDataset must contain at least two fidelity levels.")
        if isinstance(kernels, KernelBase):
            self.kernels = [kernels] * self.fidelities
            warnings.warn(
                "Single kernel provided. Using the same kernel for all fidelity levels.")
        else:
            if len(kernels) != self.fidelities:
                raise ValueError(
                    "Number of kernels must match number of fidelity levels.")
            self.kernels = kernels

        if isinstance(noise, float):
            self.noises = jnp.array([noise] * (self.fidelities))
            warnings.warn(
                "Single noise parameter provided. Using the same noise for all fidelity levels.")
        else:
            if len(noise) != self.fidelities:
                raise ValueError(
                    "Number of noise parameters must be equal to number of fidelity levels.")
            self.noises = noise

        if isinstance(rho, float):
            self.rhos = jnp.array([rho] * (self.fidelities - 1))
            warnings.warn(
                "Single rho parameter provided. Using the same rho for all fidelity levels.")
        else:
            if len(rho) != self.fidelities - 1:
                raise ValueError(
                    "Number of rho parameters must be equal to number of fidelity levels minus one.")
            self.rhos = rho

        self.K_invs = {}
        self.trainable_params: MFGPHyperparameters = MFGPHyperparameters(
            lengthscale=jnp.array([k.lengthscale for k in self.kernels]),
            variance=jnp.array([k.variance for k in self.kernels]),
            noise=self.noises,
            rho=self.rhos
        )

    def _loss(self, log_params: MFGPHyperparameters) -> float:
        """
        Compute the overall negative log marginal likelihood (NLL) for an auto-regressive
        multi-fidelity Gaussian Process with an arbitrary number of fidelity levels.

        The model assumes the following relationships:
            y^(0)(x) = f^(0)(x)                             (base GP)
            y^(l)(x) = rho^(l-1) * f^(l-1)(x) + δ^(l)(x),  for l >= 1

        where δ^(l)(x) ~ GP(0, K^(l)(x,x')). The total loss is computed as the sum of
        the negative log marginal likelihoods for the base GP and for each of the discrepancy GPs.

        The parameters in log_params are assumed to be ordered as:
        - For fidelity 0: [low_lengthscale (d,), low_variance, low_noise]
        - For each fidelity l >= 1:
                [delta_lengthscale (d,), delta_variance, delta_noise, rho_l]

        Args:
            log_params (MFGPHyperparameters): Hyperparameters in log-space, including
                lengthscale, variance, noise, and rho parameters for each fidelity level.

        Returns:
            float: A scalar representing the overall negative log marginal likelihood.
        """
        fidelities = self.fidelities
        params: MFGPHyperparameters = jax.tree_map(jnp.exp, log_params)
        noise = params.noise[0]
        # Update kernel parameters for fidelity 0
        self.kernels[0].lengthscale = params.lengthscale[0]
        self.kernels[0].variance = params.variance[0]

        data_0 = self.dataset.get_data(0)
        x_0, y_0 = data_0.x_train, data_0.y_train

        K0 = self.kernels[0](x_0, x_0) + noise * jnp.eye(x_0.shape[0]) + 1e-6 * jnp.eye(x_0.shape[0])
        L0 = cholesky(K0, lower=True)
        alpha0 = cho_solve((L0, True), y_0)
        log_det_K0 = 2.0 * jnp.sum(jnp.log(jnp.diag(L0)))
        # Base NLL loss (omitting constant terms)
        loss_total = 0.5 * (y_0.T @ alpha0 + log_det_K0)
        
        def pred_0(x):
            return self.kernels[0](x, x_0) @ alpha0
        pred = pred_0  # initial composite predictor

        for l in range(1, fidelities):
            noise = params.noise[l]
            rho = params.rho[l-1]
            self.kernels[l].lengthscale = params.lengthscale[l]
            self.kernels[l].variance = params.variance[l]

            data_l = self.dataset.get_data(l)
            X_l, y_l = data_l.x_train, data_l.y_train

            pred_lower = pred(X_l)
            # Compute the residual (discrepancy) for fidelity level l:
            y_residual = y_l - rho * pred_lower
            # Build the covariance matrix for the discrepancy process
            K_l = self.kernels[l](X_l, X_l) + noise * jnp.eye(X_l.shape[0]) + 1e-6 * jnp.eye(X_l.shape[0])
            L_l = cholesky(K_l, lower=True)
            alpha_l = cho_solve((L_l, True), y_residual)
            log_det_K_l = 2.0 * jnp.sum(jnp.log(jnp.diag(L_l)))

            loss_total += 0.5 * (y_residual.T @ alpha_l + log_det_K_l)

            # Capture the previous predictor so we don't call the updated one recursively.
            prev_pred = pred
            def _pred(x, prev_pred=prev_pred, rho=rho, X_l=X_l, alpha_l=alpha_l, kernel=self.kernels[l]):
                return rho * prev_pred(x) + kernel(x, X_l) @ alpha_l
            pred = _pred

        return jnp.squeeze(loss_total)
  

    def fit(self, lr: float = 0.001, steps: int = 5000):
        """
        Train the auto-regressive multi-fidelity Gaussian Process (AR-MFGP) model.

        This method optimizes the model's hyperparameters (stored in log-space) using the Adam optimizer
        combined with a gradient descent scheme implemented via `lax.scan`. After optimization, the model's
        hyperparameters are updated, and the inverse kernel matrices for both the low- and high-fidelity
        processes are computed.

        The procedure is as follows:
        1. Initialize the hyperparameters (in log-space) and set up the Adam optimizer.
        2. Compute the gradient of the negative log marginal likelihood (NLL) using the `_loss` function.
        3. Perform iterative optimization over a specified number of steps.
        4. Update the trainable parameters by converting them back from log-space.
        5. Update the kernel parameters for both fidelity levels and compute the corresponding kernel
            matrix inverses (stored in `self.K_invs`).

        Args:
            lr (float): Learning rate for the Adam optimizer.
            steps (int): Number of optimization steps to perform.

        Returns:
            None
        """

        # transform parameters to log space to ensure positivity
        params = jax.tree_map(jnp.log, self.trainable_params)

        optimizer = optax.adam(lr)
        opt_state = optimizer.init(params)
        loss_grad = grad(self._loss)

        # One step optimization
        def step(state, _):
            params, opt_state = state
            grads = loss_grad(params)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), None

        (params, _), _ = lax.scan(step, (params, opt_state), None, length=steps)

        # Update final parameters
        self.trainable_params = jax.tree_map(jnp.exp, params)
        print(self.trainable_params)
        self._update_K_invs()
        
    def _update_K_invs(self) -> None:
        """
        Update the inverse kernel matrices for all fidelity levels.

        For each fidelity level, this method:
        1. Retrieves the training inputs.
        2. Updates the kernel parameters (lengthscale and variance) using the current trainable parameters.
        3. Constructs the kernel matrix by evaluating the kernel on the training inputs and adding a noise term
            (with a small jitter for numerical stability).
        4. Computes the Cholesky decomposition of the kernel matrix.
        5. Computes and stores the inverse of the kernel matrix in self.K_invs.

        Returns:
            None
        """

        for i in range(0, self.fidelities):
            x_l = self.dataset.get_data(i).x_train
            # update kernel parameters by the way
            # TODO use jit instead of updating kernel parameters
            self.kernels[i].lengthscale = self.trainable_params.lengthscale[i]
            self.kernels[i].variance = self.trainable_params.variance[i]
            noise = self.trainable_params.noise[i]
            K_l = self.kernels[i](x_l, x_l) + noise * \
                jnp.eye(x_l.shape[0]) + 1e-6 * jnp.eye(x_l.shape[0])
            L = cholesky(K_l, lower=True)
            self.K_invs[i] = cho_solve((L, True), jnp.eye(x_l.shape[0]))
            


    def _predict(self, X_new: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Predict the composite mean and covariance at new input locations using the trained
        auto-regressive multi-fidelity GP model. This version recursively updates the composite
        predictor using the stored inverses self.K_invs for each fidelity level.
        
        For fidelity level 0:
            m^(0)(x) = k^(0)(x, X_0) @ K_invs[0] @ y_0
            S^(0)(x, x') = k^(0)(x, x') - k^(0)(x, X_0) @ K_invs[0] @ k^(0)(x, X_0)^T
        
        For fidelity level l >= 1:
            m^(l)(x) = rho^(l-1) * m^(l-1)(x) + k^(l)(x, X_l) @ K_invs[l] @ (y_l - rho^(l-1)*m^(l-1)(X_l))
            S^(l)(x, x') = rho^(l-1)^2 * S^(l-1)(x, x') + [ k^(l)(x, x') - k^(l)(x, X_l) @ K_invs[l] @ k^(l)(x, X_l)^T ]
        
        Args:
            X_new (jnp.ndarray): New input locations of shape [n_new, d].
        
        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: The composite predicted mean and covariance at X_new.
        """
        data_0 = self.dataset.get_data(0)
        X0, y0 = data_0.x_train, data_0.y_train
        K0_inv = self.K_invs[0]
        
        # Base predictive mean and covariance at new inputs.
        mu0 = self.kernels[0](X_new, X0) @ (K0_inv @ y0)
        cov0 = self.kernels[0](X_new, X_new) - self.kernels[0](X_new, X0) @ K0_inv @ self.kernels[0](X_new, X0).T

        composite_mu = mu0
        composite_cov = cov0
        
        # Define the base predictor function for any inputs.
        def m0(X):
            return self.kernels[0](X, X0) @ (K0_inv @ y0)
        
        pred = m0  # composite predictor (mean function) starting from fidelity 0
        
        # --- Loop over higher fidelities (l = 1, 2, ..., self.fidelities - 1) ---
        for l in range(1, self.fidelities):
            data_l = self.dataset.get_data(l)
            X_l, y_l = data_l.x_train, data_l.y_train
            rho_l = self.rhos[l-1]
            K_l_inv = self.K_invs[l]
            
            # Compute the composite prediction at the training inputs for fidelity l.
            # This ensures we have the correct shape (matching y_l).
            pred_lower = pred(X_l)  # shape: (n_l,)
            # Compute the discrepancy residual:
            residual = y_l - rho_l * pred_lower  # both should have shape (n_l,)
            
            # Compute discrepancy prediction at new inputs:
            m_delta_new = self.kernels[l](X_new, X_l) @ (K_l_inv @ residual)
            composite_mu = rho_l * composite_mu + m_delta_new

            # Compute discrepancy covariance at new inputs:
            cov_delta = self.kernels[l](X_new, X_new) - self.kernels[l](X_new, X_l) @ K_l_inv @ self.kernels[l](X_new, X_l).T
            composite_cov = (rho_l ** 2) * composite_cov + cov_delta

            # --- Update composite predictor function for training inputs ---
            # Capture the current predictor to avoid recursive self-reference.
            prev_pred = pred
            def new_pred(x, prev_pred=prev_pred, rho=rho_l, X_l=X_l, K_l_inv=K_l_inv, residual=residual, kernel=self.kernels[l]):
                return rho * prev_pred(x) + kernel(x, X_l) @ (K_l_inv @ residual)
            pred = new_pred

        return composite_mu, composite_cov


    def predict(self, X_test: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Predict the mean and variance at new test input locations using the AR-MFGP model.

        This method calls the internal _predict function to obtain the composite predictive
        mean and covariance, and then extracts the variance (diagonal of the covariance matrix),
        ensuring that the variance is non-negative by applying a minimum clip value.

        Args:
            X_test (jnp.ndarray): Test input locations of shape [n_test, d].

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: A tuple containing:
                - mu: Predicted mean (vector) at X_test.
                - var_s: Predicted variance (vector) at X_test.
        """
        mu, cov = self._predict(X_test)
        var_s = jnp.clip(jnp.diag(cov), min=1e-9, max=None)
        return mu, var_s

    def joint_predict(self, X_test: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute the full joint predictive distribution at new test input locations using the AR-MFGP model.

        This method returns the complete predictive mean and covariance matrix without
        reducing the covariance to a variance vector, allowing for joint uncertainty analysis.

        Args:
            X_test (jnp.ndarray): Test input locations of shape [n_test, d].

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: A tuple (mu, cov) where:
                - mu: The predictive mean vector at X_test.
                - cov: The full predictive covariance matrix at X_test.
        """
        return self._predict(X_test)
    
    def _test_2f_loss(self, log_params: jnp.ndarray) -> float:
        """
        Compute the negative log marginal likelihood.
        """
        # Extract parameters
        # Dimensionality of input space
        d = self.kernel_low.lengthscale.shape[0]
        low_lengthscale = jnp.exp(log_params[:d])
        low_variance = jnp.exp(log_params[d])
        low_noise = jnp.exp(log_params[d+1])

        delta_lengthscale = jnp.exp(log_params[d+2:2*d+2])
        delta_variance = jnp.exp(log_params[2*d+2])
        delta_noise = jnp.exp(log_params[2*d+3])

        rho = jnp.exp(log_params[-1])

        # Update kernel parameters
        self.kernel_low.lengthscale, self.kernel_low.variance = low_lengthscale, low_variance
        self.kernel_delta.lengthscale, self.kernel_delta.variance = delta_lengthscale, delta_variance
        self.rho = rho

        # Compute covariance matrices
        K_low = self.kernel_low(self.X_low, self.X_low) + low_noise * \
            jnp.eye(self.X_low.shape[0]) + 1e-6 * jnp.eye(self.X_low.shape[0])
        K_delta = self.kernel_delta(
            self.X_high, self.X_high) + delta_noise * jnp.eye(self.X_high.shape[0]) + 1e-6 * jnp.eye(self.X_high.shape[0])

        # Cholesky decomposition
        L_low = cholesky(K_low, lower=True)
        L_delta = cholesky(K_delta, lower=True)

        # Compute alpha for low-fidelity
        alpha_low = cho_solve((L_low, True), self.y_low)

        # Compute residuals for high-fidelity
        y_residual = self.y_high - rho * \
            self.kernel_low(self.X_high, self.X_low) @ alpha_low

        # Solve for discrepancy process
        alpha_delta = cho_solve((L_delta, True), y_residual)

        # Compute negative log marginal likelihood
        log_det_K_low = 2.0 * jnp.sum(jnp.log(jnp.diag(L_low)))
        log_det_K_delta = 2.0 * jnp.sum(jnp.log(jnp.diag(L_delta)))

        return jnp.squeeze(
            0.5 * self.y_low.T @ alpha_low + 0.5 * log_det_K_low +
            0.5 * y_residual.T @ alpha_delta + 0.5 * log_det_K_delta
        )

    def _test_2f_predict(self, X_test: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Predict using AR-MFGP.

        :param X_test: Test inputs.
        :return: Mean and variance.
        """
        # Compute covariance between training and test points
        # Covariance between low-fidelity training and test points
        K_s_low = self.kernel_low(self.X_low, X_test)
        # Covariance between high-fidelity training and test points
        K_s_high = self.kernel_delta(self.X_high, X_test)

        # Compute mean predictions for low-fidelity and discrepancy GPs
        # Low-fidelity prediction at test points
        mu_low = K_s_low.T @ self.K_inv_low @ self.y_low

        # Compute discrepancy mean at high-fidelity training points
        mu_delta = self.kernel_delta(
            self.X_high, X_test).T @ self.K_inv_delta @ self.y_high
        mu_delta_train = self.kernel_delta(
            self.X_high, self.X_high).T @ self.K_inv_delta @ self.y_high

        # Compute discrepancy mean at test points
        # mu_delta = K_s_high.T @ self.K_inv_delta @ self.y_high

        # Compute the correction term using high-fidelity training data
        K_ss_high = self.kernel_delta(
            self.X_high, self.X_high) + 1e-6 * jnp.eye(self.X_high.shape[0])
        # Inverse of high-fidelity covariance matrix
        K_inv_high = jnp.linalg.inv(K_ss_high)

        correction_term = K_s_high.T @ K_inv_high @ (self.y_high - self.rho * self.kernel_low(
            self.X_low, self.X_high).T @ self.K_inv_low @ self.y_low - mu_delta_train)

        # print(mu_low.shape, correction_term.shape, mu_delta_train.shape)
        mu_mf = self.rho * mu_low + mu_delta + correction_term

        # Compute prior variances at test points
        # + 1e-6 * jnp.eye(X_test.shape[0])  # Prior variance of low-fidelity GP
        K_ss_low = self.kernel_low(X_test, X_test)
        # Corrected: full self-covariance, not just diagonal
        K_ss_delta = self.kernel_delta(X_test, X_test)

        # Compute low-fidelity and discrepancy variances
        var_low = K_ss_low - K_s_low.T @ self.K_inv_low @ K_s_low  # Low-fidelity variances
        # Compute variance correction term
        var_correction = K_s_high.T @ K_inv_high @ K_s_high  # This term reduces uncertainty

        var_mf = self.rho**2 * var_low + K_ss_delta - \
            var_correction  # Use full matrices, not just diag

        # Ensure positive variance
        # var_mf = jnp.clip(jnp.diag(var_mf), a_min=1e-9, a_max=None)

        return mu_mf.squeeze(), var_mf