import jax.numpy as jnp
from jax.scipy.stats import norm
from baox.surrogate.gp import SingleOuputGP
import jax
import optax
from baox.acquisition.base import BaseAcquisitionFunction

class ExpectedImprovement(BaseAcquisitionFunction):
    """
    Expected Improvement (EI) acquisition function for Bayesian Optimization.

    This class implements the Expected Improvement acquisition function, which quantifies
    the expected gain over the current best observation. The EI is optimized using a gradient‐
    based method with multiple random restarts to find a candidate point in the input domain
    (assumed to be normalized in [0,1]^d).

    Attributes:
        gp (SingleOuputGP): A Gaussian Process surrogate model.
        xi (float): Exploration parameter that encourages exploration by adding a margin.
        n_restarts (int): Number of random restarts for the optimization procedure.
        n_steps (int): Number of gradient descent steps per restart.
        lr (float): Learning rate for the optimizer.
    """
    def __init__(self, gp: SingleOuputGP, xi: float = 0.01, n_restarts: int = 8, n_steps: int = 50, lr: float = 1e-2):
        """
        Initialize the Expected Improvement acquisition function.

        Args:
            gp (SingleOuputGP): A surrogate Gaussian Process model.
            xi (float, optional): Exploration parameter. Default is 0.01.
            n_restarts (int, optional): Number of random restarts for optimization. Default is 8.
            n_steps (int, optional): Number of optimization steps per restart. Default is 50.
            lr (float, optional): Learning rate for the Adam optimizer. Default is 1e-2.
        """
        super().__init__(gp)
        self.xi = xi
        self.n_restarts = n_restarts
        self.n_steps = n_steps
        self.lr = lr

    def propose_candidates(self, key: jax.random.PRNGKey, *args) -> jnp.ndarray:
        """
        Optimize the Expected Improvement (EI) acquisition function to propose a candidate point.

        This method performs multiple random restarts using gradient-based optimization (Adam)
        to minimize the negative EI value. The candidate point returned is in the normalized
        domain [0, 1]^d.

        Args:
            key (jax.random.PRNGKey): A PRNG key for random number generation.
            *args: Additional arguments (unused).

        Returns:
            jnp.ndarray: A candidate point (vector) of shape (d,) that maximizes EI.
        """
        d = self.gp.x_train.shape[1]

        def EI_objective(x: jnp.ndarray) -> jnp.ndarray:
            """
            Compute the negative Expected Improvement (EI) at a given point.

            EI is computed as:
                EI(x) = (μ(x) - f_best - ξ) * Φ(Z) + σ(x) * φ(Z)
            where Z = (μ(x) - f_best - ξ) / σ(x), μ and σ are the predictive mean and standard deviation,
            f_best is the best observed function value, Φ is the CDF and φ is the PDF of the standard normal distribution.

            Args:
                x (jnp.ndarray): A point in the normalized domain of shape (d,).

            Returns:
                jnp.ndarray: Negative EI value (scalar).
            """
            mu, sigma = self.gp.predict(x[None, :])
            sigma = jnp.maximum(sigma, 1e-9)
            f_best = jnp.max(self.gp.y_train) if self.gp.y_train.size > 0 else 0.0
            Z = (mu - f_best - self.xi) / sigma
            ei = (mu - f_best - self.xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
            return -ei[0]

        optimizer = optax.adam(self.lr)

        def single_restart(subkey: jax.random.PRNGKey):
            """
            Perform one random restart for EI optimization.

            Args:
                subkey (jax.random.PRNGKey): PRNG key for initialization.

            Returns:
                Tuple[jnp.ndarray, jnp.ndarray]:
                    - The optimized candidate point (vector) of shape (d,).
                    - The final negative EI value (scalar) at that candidate.
            """
            x0 = jax.random.uniform(subkey, shape=(d,))
            opt_state = optimizer.init(x0)

            def step_fn(carry, _):
                x, opt_state = carry
                val, grads = jax.value_and_grad(EI_objective)(x)
                updates, opt_state = optimizer.update(grads, opt_state)
                x = optax.apply_updates(x, updates)
                x = jnp.clip(x, 0.0, 1.0)
                return (x, opt_state), val

            (xf, _), _ = jax.lax.scan(step_fn, (x0, opt_state), jnp.arange(self.n_steps))
            final_val = EI_objective(xf)
            return xf, final_val

        subkeys = jax.random.split(key, self.n_restarts)
        final_xs, final_vals = jax.vmap(single_restart)(subkeys)
        best_idx = jnp.argmin(final_vals)
        best_x = final_xs[best_idx]
        return best_x

    def evaluate(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate the Expected Improvement (EI) at given points.

        This method computes the EI at each provided point using the surrogate GP model.

        Args:
            X (jnp.ndarray): An array of points in the normalized domain of shape (n_points, d).

        Returns:
            jnp.ndarray: An array of EI values corresponding to each point in X.
        """
        mu, sigma = self.gp.predict(X)
        sigma = jnp.maximum(sigma, 1e-9)
        f_best = jnp.max(self.gp.y_train) if self.gp.y_train.size > 0 else 0.0
        Z = (mu - f_best - self.xi) / sigma
        return (mu - f_best - self.xi) * norm.cdf(Z) + sigma * norm.pdf(Z)