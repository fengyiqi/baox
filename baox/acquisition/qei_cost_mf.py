import jax.numpy as jnp
from jax.scipy.stats import norm
import jax
import copy
import optax
from functools import partial
from baox.acquisition.base import BaseAcquisitionFunction

    

class qCostAwareMultiFidelityEI(BaseAcquisitionFunction):
    """
    Joint acquisition function that proposes a batch of candidates,
    each of which includes a fidelity indicator. The candidate vector is
    of shape (batch_size, d+1), where the first d components are the input
    (in normalized [0,1]^d space) and the last component is a fidelity indicator.
    
    The fidelity indicator is relaxed to a continuous value and mapped to a cost:
      - values near 0 correspond to low-fidelity evaluations (cost_low)
      - values near 1 correspond to high-fidelity evaluations (cost_high)
    """
    def __init__(self, gp, cost_low: float, cost_high: float, n_samples: int = 256,
                 n_restarts: int = 16, n_steps: int = 200, lr: float = 1e-2):
        super().__init__(gp)
        self.cost_low = cost_low
        self.cost_high = cost_high
        self.n_samples = n_samples
        self.n_restarts = n_restarts
        self.n_steps = n_steps
        self.lr = lr

    def propose_candidates(self, key: jax.random.PRNGKey, batch_size: int) -> jnp.ndarray:
        """
        Jointly optimize over a candidate batch that includes both the input location
        (first d dimensions) and a fidelity indicator (last dimension). Returns an array of
        shape (batch_size, d+1). After optimization, you can threshold the fidelity indicator
        (e.g., <0.5 → low, >=0.5 → high).
        """
        d = self.gp.X_high.shape[1]  # dimension of the input space (excluding fidelity indicator)

        def objective(x: jnp.ndarray) -> jnp.ndarray:
            # Reshape flattened vector into (batch_size, d+1)
            qEI_value = self._qEI_mc(x, key)
            qEI_value = jnp.nan_to_num(qEI_value, nan=-1e6)
            return - qEI_value

        optimizer = optax.adam(self.lr)

        def single_restart(subkey: jax.random.PRNGKey):
            # Initialize randomly in [0,1]^(batch_size x (d+1))
            x0 = jax.random.uniform(subkey, shape=(batch_size, d + 1))
            x0_flat = x0
            opt_state = optimizer.init(x0_flat)

            def step_fn(carry, _):
                x_flat, opt_state = carry
                val, grads = jax.value_and_grad(objective)(x_flat)
                updates, opt_state = optimizer.update(grads, opt_state)
                x_flat = optax.apply_updates(x_flat, updates)
                # Ensure the candidates stay within [0, 1]
                x_flat = jnp.clip(x_flat, 0.0, 1.0)
                return (x_flat, opt_state), val

            (xf, _), vals = jax.lax.scan(step_fn, (x0_flat, opt_state), jnp.arange(self.n_steps))
            final_val = vals[-1]
            return xf, final_val

        subkeys = jax.random.split(key, self.n_restarts)
        final_xs, final_vals = jax.vmap(single_restart)(subkeys)
        best_idx = jnp.argmin(final_vals)  # because we minimize the negative objective
        best_flat = final_xs[best_idx]
        return best_flat

    def _qEI_mc(self, X: jnp.ndarray, key: jax.random.PRNGKey) -> jnp.ndarray:
        """
        Monte Carlo estimate of cost-aware qEI over a batch of candidate inputs X.
        X: candidate inputs of shape (q, d)
        cost_per_candidate: a vector of shape (q,) with cost for each candidate
        """
        q = X.shape[0]
        # Get joint predictions (mean and covariance) for the batch from the mf-GP.
        x_test = X[:, :-1]  # shape (q, d)
        fid_ind = X[:, -1]  # shape (q,)
        mu, cov = self.gp.joint_predict(x_test)  # mu shape: (q,), cov shape: (q, q)
        fid_prob = jax.nn.sigmoid(10.0 * (fid_ind - 0.5))  # sharp transition
        cost_per_candidate = (1.0 - fid_prob) * self.cost_low + fid_prob * self.cost_high
        
        # Compute f_best based on high-fidelity observations only.
        f_best = jnp.max(self.gp.y_high) if self.gp.y_high.size > 0 else 0.0
        # Ensure covariance is positive definite.
        L = jnp.linalg.cholesky(cov + 1e-9 * jnp.eye(q))
        key, subkey = jax.random.split(key)
        z = jax.random.normal(subkey, (self.n_samples, q))
        f_samples = mu + z @ L.T  # shape (n_samples, q)
        
        # Compute per-candidate improvement and adjust by candidate-specific cost.
        improvement = jnp.maximum(f_samples - f_best, 0.0) # shape (n_samples, q)
        improvement = improvement / cost_per_candidate[None, :] # shape (n_samples, q)
        
        # For each Monte Carlo sample, choose the candidate with the highest cost-adjusted improvement.
        best_adjusted = jnp.max(improvement, axis=1)
        qEI_cost = jnp.mean(best_adjusted)# / jnp.mean(cost_per_candidate)
        return qEI_cost
    
