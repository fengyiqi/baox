import jax.numpy as jnp
from jax.scipy.stats import norm
from baox.surrogate.gp import SingleOuputGP
import jax
import copy
import optax
from functools import partial
from baox.acquisition.base import BaseAcquisitionFunction


class qExpectedImprovementJoint(BaseAcquisitionFunction):
    """
    Single-shot (joint) qEI: We pick all q points simultaneously by
    maximizing the qEI in one shot, i.e. qEI(X_batch) ~ E[ max(0, max(f(xi)) - f_best ) ].
    """

    def __init__(
        self,
        gp: SingleOuputGP,
        n_samples: int = 128,
        n_restarts: int = 5,
        n_steps: int = 50,
        lr: float = 1e-2
    ):
        super().__init__(gp)
        self.n_samples = n_samples   
        self.n_restarts = n_restarts
        self.n_steps = n_steps
        self.lr = lr

    def propose_candidates(
        self,
        key: jax.random.PRNGKey,
        batch_size: int
    ) -> jnp.ndarray:
        """
        Return a [batch_size, d] array by maximizing joint qEI in a single shot.
        """
        d = self.gp.x_train.shape[1]

        best_xbatch = self._maximize_qEI(key, batch_size, d)
        return best_xbatch

    def _maximize_qEI(self, key, q: int, d: int) -> jnp.ndarray:
        """
        Jointly maximize qEI over X_batch in [0,1]^(q*d). 
        We'll flatten X_batch to shape (q*d,).
        Then do multiple random restarts (vmap).
        """
        def qEI_objective(x_flat: jnp.ndarray) -> jnp.ndarray:
            X_batch = x_flat.reshape((q, d))
            return -self._qEI_mc(X_batch, key, self.n_samples)

        optimizer = optax.adam(self.lr)

        def single_restart(subkey: jax.random.PRNGKey):
            x0 = jax.random.uniform(subkey, shape=(q, d), minval=0.0, maxval=1.0)
            x0_flat = x0.ravel()
            opt_state = optimizer.init(x0_flat)

            def step_fn(carry, _):
                x_flat, opt_state = carry
                val, grads = jax.value_and_grad(qEI_objective)(x_flat)
                updates, opt_state = optimizer.update(grads, opt_state)
                x_flat = optax.apply_updates(x_flat, updates)
                x_flat = jnp.clip(x_flat, 0.0, 1.0)
                return (x_flat, opt_state), val

            (xf, _), vals = jax.lax.scan(step_fn, (x0_flat, opt_state), jnp.arange(self.n_steps))
            final_val = vals[-1]  
            return xf, final_val

        subkeys = jax.random.split(key, self.n_restarts)
        final_xs, final_vals = jax.vmap(single_restart)(subkeys)

        best_idx = jnp.argmin(final_vals) 
        best_flat = final_xs[best_idx]
        best_xbatch = best_flat.reshape((q, d))
        return best_xbatch

    def _qEI_mc(self, X_batch: jnp.ndarray, key: jax.random.PRNGKey, n_samples: int) -> jnp.ndarray:
        """
        Monte Carlo estimate of qEI(X_batch):
        
          E[max(0, max(f(x) for x in X_batch) - f_best )].
          
        We assume the GP provides a joint posterior for X_batch,
        i.e. we can sample from MVN. 
        If your GP only returns independent predictions, that's a simplification.
        
        :param X_batch: shape [q, d]
        :return: scalar qEI
        """
        q = X_batch.shape[0]
        mu, cov = self.gp.joint_predict(X_batch)

        L = jnp.linalg.cholesky(cov + 1e-9*jnp.eye(q))  
        key, subkey = jax.random.split(key)
        z = jax.random.normal(subkey, (n_samples, q))
        f_samples = mu + z @ L.T  

        f_best = jnp.max(self.gp.y_train) if self.gp.y_train.size > 0 else 0.0
        best_in_batch = jnp.max(f_samples, axis=1) 
        improvement = jnp.maximum(best_in_batch - f_best, 0.0)
        qEI = jnp.mean(improvement)
        return qEI