import jax.numpy as jnp
from baox.surrogate.gp import SingleOuputGP
import jax
import optax
from baox.acquisition.base import BaseAcquisitionFunction
import copy
from jax.scipy.stats import norm



class qExpectedImprovementFantasy(BaseAcquisitionFunction):
    """
    Implementation of a 'fantasy' (or 'Kriging Believer') qEI approach
    where each new point is found by gradient-based optimization of single-point EI.
    After we choose each point, we add a fantasy observation to a copy of the GP
    so that subsequent picks don't collapse to the same location.
    """

    def __init__(
        self,
        gp: SingleOuputGP,
        xi: float = 0.01,
        n_restarts: int = 8,
        n_steps: int = 50,
        lr: float = 1e-2
    ):
        """
        :param gp: The current (fully trained) GP model.
        :param xi: Exploration parameter for EI.
        :param n_restarts: Number of random restarts for each point's gradient-based search.
        :param n_steps: Number of gradient steps per random restart.
        :param lr: Learning rate for the optimizer.
        """
        super().__init__(gp)
        self.xi = xi
        self.n_restarts = n_restarts
        self.n_steps = n_steps
        self.lr = lr

    def propose_candidates(
        self,
        key: jax.random.PRNGKey,
        batch_size: int
    ) -> jnp.ndarray:
        """
        Ignore X_candidates. For each point in the batch, do:
          1) gradient-based max of EI,
          2) add to GP copy as a 'fantasy' observation,
          3) re-fit or re-update the copy.
        Returns [batch_size, d] array of chosen points.
        """
        chosen_points = []
        gp_copy = copy.deepcopy(self.gp)  # so we don't modify the real GP
        # dimension of the input space (1 if 1D)

        for _ in range(batch_size):
            # 1) Find best single point by gradient-based max of EI
            x_star = self._maximize_single_EI(gp_copy, key)

            # 2) "Fantasy" observation = predicted mean (or a random sample from posterior)
            mu, _ = gp_copy.predict(x_star[None, :])
            # or mu[0] + random.normal(...) * jnp.sqrt(sigma[0])
            y_fantasy = mu[0]

            # 3) Temporarily update gp_copy with the new "fantasy" data
            gp_copy.x_train = jnp.concatenate(
                [gp_copy.x_train, x_star[None, :]], axis=0)
            gp_copy.y_train = jnp.concatenate(
                [gp_copy.y_train, jnp.array([y_fantasy])], axis=0)
            gp_copy.fit()

            chosen_points.append(x_star)

            # Update random key
            key, _ = jax.random.split(key)

        return jnp.stack(chosen_points)

    def _maximize_single_EI(self, gp_copy: SingleOuputGP, key) -> jnp.ndarray:
        """
        Use vmap to do multiple random restarts in parallel. For each restart:
          - sample a random x0 in domain
          - do gradient-based 'descent' on negative EI 
          - return best final solution among all restarts
        """
        d = gp_copy.x_train.shape[1]

        # Negative single-point EI objective (so that we do gradient descent).
        def EI_objective(x: jnp.ndarray) -> jnp.ndarray:
            # x: shape (d,)
            # We'll do a single-point EI, returning scalar. 
            return -self._ei_single_point_mc(gp_copy, x[None, :], key)

        # The function that runs one entire "restart"
        def single_restart(subkey: jax.random.PRNGKey):
            # 1) Random init
            x0 = jax.random.uniform(subkey, shape=(d,))
            opt_state = optimizer.init(x0)

            # 2) A single gradient step function
            def step_fn(carry, _):
                x, opt_state = carry
                val, grads = jax.value_and_grad(EI_objective)(x)
                updates, opt_state = optimizer.update(grads, opt_state)
                x = optax.apply_updates(x, updates)
                # Clip to domain
                x = jnp.clip(x, 0, 1)
                return (x, opt_state), val

            # 3) Unroll n_steps using jax.lax.scan
            (xf, _), vals = jax.lax.scan(step_fn, (x0, opt_state), jnp.arange(self.n_steps))
            final_val = vals[-1]
            return xf, final_val  # shape (d,), scalar

        # Build an optimizer
        optimizer = optax.adam(self.lr)

        # Create subkeys for each random restart
        subkeys = jax.random.split(key, self.n_restarts)  # shape (n_restarts,)

        # Vectorize single_restart over subkeys => 
        #   final_xs shape: (n_restarts, d)
        #   final_vals shape: (n_restarts,)
        final_xs, final_vals = jax.vmap(single_restart)(subkeys)

        # 4) Pick best among the restarts
        best_idx = jnp.argmin(final_vals)  # since we minimized negative EI
        best_x = final_xs[best_idx]
        return best_x
    
    def _ei_single_point(self, gp_local, X: jnp.ndarray) -> jnp.ndarray:
        """
        Standard single-point EI formula:
          EI(x) = (mu - f_best - xi)*Phi(Z) + sigma*phi(Z)
          Z = (mu - f_best - xi)/sigma
        """
        mu, sigma = gp_local.predict(X)
        sigma = jnp.maximum(sigma, 1e-9)
        f_best = jnp.max(gp_local.y_train) 
        Z = (mu - f_best - self.xi) / sigma
        ei_value = (mu - f_best - self.xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
        return ei_value[0]  # return as scalar
    
    def _ei_single_point_mc(
        self,
        gp_local: SingleOuputGP,
        X: jnp.ndarray,
        key: jax.random.PRNGKey,
        n_samples: int = 256
    ) -> jnp.ndarray:
        """
        Monte Carlo approximation of single-point EI.
        
        EI(x) = E[max(0, f(x) - f_best - xi]]
        
        We sample from f(x) ~ N(mu, sigma) using the GP posterior at X.
        
        :param gp_local: a trained/copy of GaussianProcess
        :param X: shape [N, d] input points
        :param key: PRNG key for randomness
        :param n_samples: number of MC samples
        :return: EI values, shape [N]
        """
        mu, sigma = gp_local.predict(X)   # each shape [N]
        sigma = jnp.maximum(sigma, 1e-9)  # avoid zero stdev
        f_best = jnp.max(gp_local.y_train) if gp_local.y_train.size > 0 else 0.0

        # Draw samples from the posterior: f_samples ~ N(mu, sigma)
        # shape will be [N, n_samples]
        subkey, _ = jax.random.split(key)
        normal_samples = jax.random.normal(subkey, shape=(X.shape[0], n_samples))
        f_samples = mu[:, None] + jnp.sqrt(sigma)[:, None] * normal_samples

        # Improvement = max(0, f - f_best - xi)
        improvement = jnp.maximum(f_samples - f_best, 0.0)

        # MC average across the samples axis
        ei_estimate = jnp.mean(improvement, axis=1)  # shape [N]
        return ei_estimate[0]