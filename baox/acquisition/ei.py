import jax.numpy as jnp
from jax.scipy.stats import norm
from baox.surrogate.gp import SingleOuputGP
import jax
import optax
from baox.acquisition.base import BaseAcquisitionFunction

class ExpectedImprovement(BaseAcquisitionFunction):
    def __init__(self, gp: SingleOuputGP, xi: float = 0.01, n_restarts: int = 8, n_steps: int = 50, lr: float = 1e-2):
        super().__init__(gp)
        self.xi = xi
        self.n_restarts = n_restarts
        self.n_steps = n_steps
        self.lr = lr

    def propose_candidates(self, key: jax.random.PRNGKey, *args) -> jnp.ndarray:
        """
        Optimize EI using gradient-based optimization with multiple random restarts.
        Returns the point maximizing EI.
        """
        d = self.gp.x_train.shape[1]

        def EI_objective(x: jnp.ndarray) -> jnp.ndarray:
            # We assume x is of shape (d,) and reshape to (1, d)
            mu, sigma = self.gp.predict(x[None, :])
            sigma = jnp.maximum(sigma, 1e-9)
            f_best = jnp.max(self.gp.y_train) if self.gp.y_train.size > 0 else 0.0
            Z = (mu - f_best - self.xi) / sigma
            # Negative EI for minimization
            ei = (mu - f_best - self.xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
            return -ei[0]

        optimizer = optax.adam(self.lr)

        def single_restart(subkey: jax.random.PRNGKey):
            # Random initialization in the domain [0, 1]^d
            x0 = jax.random.uniform(subkey, shape=(d,))
            opt_state = optimizer.init(x0)

            def step_fn(carry, _):
                x, opt_state = carry
                val, grads = jax.value_and_grad(EI_objective)(x)
                updates, opt_state = optimizer.update(grads, opt_state)
                x = optax.apply_updates(x, updates)
                # Clip to domain [0, 1]
                x = jnp.clip(x, 0.0, 1.0)
                return (x, opt_state), val

            (xf, _), _ = jax.lax.scan(step_fn, (x0, opt_state), jnp.arange(self.n_steps))
            final_val = EI_objective(xf)
            return xf, final_val

        subkeys = jax.random.split(key, self.n_restarts)
        final_xs, final_vals = jax.vmap(single_restart)(subkeys)
        best_idx = jnp.argmin(final_vals)  # since we minimized negative EI
        best_x = final_xs[best_idx]
        return best_x

    def evaluate(self, X: jnp.ndarray) -> jnp.ndarray:
        # This remains the same for evaluation if needed.
        mu, sigma = self.gp.predict(X)
        sigma = jnp.maximum(sigma, 1e-9)
        f_best = jnp.max(self.gp.y_train) if self.gp.y_train.size > 0 else 0.0
        Z = (mu - f_best - self.xi) / sigma
        return (mu - f_best - self.xi) * norm.cdf(Z) + sigma * norm.pdf(Z)