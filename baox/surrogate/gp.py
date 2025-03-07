import jax.numpy as jnp
import jax
import optax
from jax.scipy.linalg import cho_solve, cholesky
from jax import grad, lax
from typing import Callable, Optional, Tuple, Self
from baox.surrogate.kernel import KernelBase, MaternKernel
import copy
from baox.surrogate.base import BaseGaussianProcess
from baox.data_types import GPHyperparameters
from baox.surrogate.mean import ConstantMean


class SingleOuputGaussianProcess(BaseGaussianProcess):
    """
    Gaussian Process regression model for single-output problems with hyperparameter
    optimization using the Adam optimizer and `lax.scan` for efficient iteration.

    This class extends BaseGaussianProcess and handles training data,
    kernel functions, and mean functions. It supports hyperparameter optimization
    in log-space to enforce positivity constraints.

    Attributes:
        x_train (jnp.ndarray): Training inputs with shape [n, d] where d is the input dimension.
        y_train (jnp.ndarray): Training outputs with shape [n].
        kernel (KernelBase): Kernel function used to compute the covariance matrix.
        mean_function (Callable[[jnp.ndarray], jnp.ndarray]): Mean function of the GP.
        noise (float): Noise level in the model.
        K_inv (Optional[jnp.ndarray]): Inverse of the kernel matrix computed during training.
    """

    def __init__(
        self,
        x_train: jnp.ndarray,
        y_train: jnp.ndarray,
        kernel: KernelBase = None,
        mean_function: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
        noise: float = 1e-2,
    ):
        """
        Initialize the SingleOutputGaussianProcess with training data, kernel, and mean function.

        Args:
            x_train (jnp.ndarray): Training inputs with shape [n, d].
            y_train (jnp.ndarray): Training outputs with shape [n].
            kernel (KernelBase, optional): Kernel function for computing the covariance matrix.
                                           If None, a default MaternKernel is used.
            mean_function (Callable[[jnp.ndarray], jnp.ndarray], optional): Mean function. Defaults to ConstantMean with zero constant.
            noise (float): Initial noise level. Defaults to 1e-2.

        Raises:
            ValueError: If x_train is not a 2D array.
            ValueError: If y_train is not a 1D array.
            ValueError: If the number of training samples in x_train and y_train do not match.
        """
        if x_train.ndim != 2:
            raise ValueError("Training inputs must have shape [n, d]")
        if y_train.ndim != 1:
            raise ValueError("Training outputs must have shape [n]")
        if x_train.shape[0] != y_train.shape[0]:
            raise ValueError("Training inputs and outputs must have the same number of samples")
        
        self.d_input = x_train.shape[1]
        self.x_train = x_train
        self.y_train = y_train
        if kernel is None:
            kernel = MaternKernel(lengthscale=jnp.ones(self.d_input), variance=1.0)
        if mean_function is None:
            mean_function = ConstantMean(constant=0.0)
        
            
        super().__init__(kernel, mean_function, noise)

        self.K_inv: Optional[jnp.ndarray] = None

    def fit(
        self, 
        lr: float = 0.01, 
        steps: int = 200
    ) -> None:
        """
        Train the Gaussian Process by optimizing hyperparameters using Adam and lax.scan.

        The method optimizes the hyperparameters (lengthscale, variance, noise) in log-space.
        It computes the negative log marginal likelihood and its gradient, and updates the parameters
        over a number of optimization steps.

        Args:
            lr (float): Learning rate for the Adam optimizer.
            steps (int): Number of optimization steps.

        Returns:
            None
        """
        # Initialize parameters in log-space for positivity constraints
        log_params = jax.tree_map(jnp.log, self.trainable_params)
        optimizer = optax.adam(lr)
        opt_state = optimizer.init(log_params)
        
        def loss(_log_params: GPHyperparameters) -> float:
            """
            Compute the negative log marginal likelihood loss for the current hyperparameters.

            Args:
                _log_params (GPHyperparameters): Hyperparameters in log-space.

            Returns:
                float: The negative log marginal likelihood.
            """
            lengthscale = jnp.exp(_log_params.lengthscale)
            variance = jnp.exp(_log_params.variance)
            noise = jnp.exp(_log_params.noise)

            K = self.kernel(
                self.x_train, 
                self.x_train, 
                lengthscale, 
                variance
            ) + noise * jnp.eye(self.x_train.shape[0])
            L = cholesky(K + 1e-6 * jnp.eye(self.x_train.shape[0]), lower=True)

            alpha = cho_solve((L, True), self.y_train)
            log_det_K = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))

            # Adding constant term following the original formulation
            return 0.5 * self.y_train.T @ alpha + 0.5 * log_det_K + \
                0.5 * self.x_train.shape[0] * jnp.log(2.0 * jnp.pi)

        # Compute gradients of loss
        loss_grad = grad(loss)

        def step(state, _):
            """
            Perform a single optimization step.

            Args:
                state (Tuple): A tuple (params, opt_state) representing the current parameters and optimizer state.
                _ : Dummy variable for compatibility with lax.scan (unused).

            Returns:
                Tuple: Updated state (params, opt_state) and None.
            """
            params, opt_state = state
            grads = loss_grad(params)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), None

        # Run optimization using lax.scan for efficient iteration
        (log_params, _), _ = lax.scan(step, (log_params, opt_state), None, length=steps)

        # Update final hyperparameters in original space
        self.trainable_params = jax.tree_map(jnp.exp, log_params)
        self._update_to_kernel()
        self._update_K_inv()
        
    def _update_to_kernel(self) -> None:
        """
        Update the kernel's hyperparameters from the optimized trainable parameters.

        This method sets the kernel's lengthscale and variance based on the current
        trainable parameters.

        Returns:
            None
        """
        self.kernel.lengthscale = self.trainable_params.lengthscale
        self.kernel.variance = self.trainable_params.variance

    def _update_K_inv(self) -> None:
        """
        Compute and update the inverse of the kernel matrix based on the training data.

        The inverse is computed using Cholesky decomposition and stored in self.K_inv.

        Returns:
            None
        """
        n = self.x_train.shape[0]
        K = self.kernel(
            self.x_train, 
            self.x_train, 
        ) + self.trainable_params.noise * jnp.eye(n) + \
            1e-6 * jnp.eye(n)
        L = cholesky(K, lower=True)
        self.K_inv = cho_solve((L, True), jnp.eye(n))

    def _predict(self, X_test: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute the predictive mean and covariance for the test inputs using the trained GP.

        This method computes the predictive distribution using the standard GP regression equations.

        Args:
            X_test (jnp.ndarray): Test input data.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: A tuple containing:
                - mu_s: Predictive mean.
                - cov_s: Predictive covariance.
        """
        K_s = self.kernel(self.x_train, X_test)
        K_ss = self.kernel(X_test, X_test)

        mean_train = self.mean_function(self.x_train)
        mean_test = self.mean_function(X_test)

        y_train_centered = self.y_train - mean_train
        mu_s = mean_test + K_s.T @ self.K_inv @ y_train_centered

        cov_s = K_ss - K_s.T @ self.K_inv @ K_s

        return mu_s, cov_s

    def predict(self, X_test: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Predict the mean and variance for the test inputs.

        This method wraps the internal _predict method and ensures that the predicted variance
        is non-negative by clipping the diagonal of the covariance matrix.

        Args:
            X_test (jnp.ndarray): Test input data.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: A tuple containing:
                - mu (jnp.ndarray): Predictive mean.
                - var_s (jnp.ndarray): Predictive variance.
        """
        mu, cov = self._predict(X_test)
        var_s = jnp.clip(jnp.diag(cov), a_min=1e-9, a_max=None)
        return mu, var_s

    def joint_predict(self, X_test: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Perform joint prediction to obtain the full predictive mean and covariance matrix.

        Args:
            X_test (jnp.ndarray): Test input data.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: A tuple containing:
                - mu (jnp.ndarray): Predictive mean.
                - cov (jnp.ndarray): Predictive covariance matrix.
        """
        return self._predict(X_test)

    def given(self, x_train: jnp.ndarray, y_train: jnp.ndarray) -> Self:
        """
        Return a new Gaussian Process with additional training data without re-fitting.

        The new GP is created by concatenating the existing training data with the new data,
        updating the kernel inversion accordingly.

        Args:
            x_train (jnp.ndarray): Additional training inputs.
            y_train (jnp.ndarray): Additional training outputs.

        Returns:
            Self: A new instance of SingleOuputGaussianProcess with the combined training data.
        """
        new_gp = copy.deepcopy(self)
        new_gp.x_train = jnp.concatenate([self.x_train, x_train], axis=0)
        new_gp.y_train = jnp.concatenate([self.y_train, y_train], axis=0)
        new_gp._update_K_inv()
        return new_gp
    
    def __str__(self):
        n_samples, n_features = self.x_train.shape
        mean_func_name = (
            self.mean_function.__class__.__name__
            if hasattr(self.mean_function, "__class__")
            else str(self.mean_function)
        )
        kernel_name = self.kernel.__class__.__name__
        return (
            f"SingleOutputGaussianProcess:\n"
            f"\tTraining Data: {n_samples} samples, {n_features} features\n"
            f"\tKernel: {kernel_name}\n"
            f"\t\tLengthscale: {self.kernel.lengthscale}\n"
            f"\t\tVariance: {self.kernel.variance}\n"
            f"\tNoise: {self.trainable_params.noise}\n"
            f"\tMean Function: {mean_func_name}\n"
        )


# Alias for convenience
SingleOuputGP = SingleOuputGaussianProcess
