# File: qkg_mf.py  (you could add this to your existing baox/acquisition/ directory)

import jax
import jax.numpy as jnp
import optax
from baox.acquisition.base import BaseAcquisitionFunction
from baox.surrogate import AutoRegressiveMFGP

class qKnowledgeGradientMF(BaseAcquisitionFunction):
    """
    Simple q-Knowledge-Gradient for Multi-Fidelity GP.
    """
    def __init__(self, gp: AutoRegressiveMFGP, n_samples: int = 2,
                 n_restarts: int = 1, n_steps: int = 16, lr: float = 0.1):
        super().__init__(gp)
        self.gp = gp
        self.dataset = gp.dataset
        self.n_samples = n_samples      # Monte Carlo draws for fantasy
        self.n_restarts = n_restarts    # Number of random restarts for candidate search
        self.n_steps = n_steps          # Gradient steps per restart
        self.lr = lr

    def propose_candidates(self, key: jax.random.PRNGKey, batch_size: int) -> jnp.ndarray:
        """
        Returns (batch_size, d+1) candidate points: (x, fidelity).
        We do a naive random+gradient search to maximize q-KG.
        """
        d = self.gp.dim
        def objective(x_flat: jnp.ndarray) -> jnp.ndarray:
            # x_flat is shape (batch_size, d+1).
            # We compute the negative KG for gradient-based *maximization*.
            kg_value = self._qKG(x_flat)
            return -kg_value

        optimizer = optax.adam(self.lr)

        def single_restart(subkey):
            x0 = jax.random.uniform(subkey, shape=(batch_size, d + 1))
            opt_state = optimizer.init(x0)

            x_current = x0
            state_current = opt_state
            vals = []
            for _ in range(self.n_steps):
                val, grads = jax.value_and_grad(objective)(x_current)
                updates, state_current = optimizer.update(grads, state_current)
                x_current = optax.apply_updates(x_current, updates)
                # Keep (x, fidelity) in [0,1]
                x_current = jnp.clip(x_current, 0.0, 1.0)
                vals.append(val)
            xf = x_current
            final_val = vals[-1]
            return xf, final_val

        subkeys = jax.random.split(key, self.n_restarts)
        results_x = []
        results_val = []
        for sk in subkeys:
            x, val = single_restart(sk)
            results_x.append(x)
            results_val.append(val)
        results_x = jnp.stack(results_x)
        results_val = jnp.stack(results_val)
        best_idx = jnp.argmin(results_val)  # because we minimize the negative objective
        return results_x[best_idx]

    def _qKG(self, X_batch: jnp.ndarray) -> jnp.ndarray:
        """
        Approximate the Knowledge Gradient for a batch of shape (q, d+1).
        1) Sample outcomes from the current GP posterior at X_batch.
        2) For each outcome, build a 'fantasy' GP and estimate its best achievable value.
        3) Return avg(new_best - old_best).
        """
        # 1) current best (e.g., best y among all fidelities or just the highest fidelity)
        current_best = self._current_best()

        # 2) Draw MC samples from posterior at X_batch
        Y_samples = self._sample_from_posterior(X_batch, self.n_samples)

        new_bests = []
        for i in range(self.n_samples):
            # Build fantasy dataset
            fantasy_gp = self._build_fantasy_gp(X_batch, Y_samples[i])
            # Evaluate best under fantasy
            nb = self._approx_best(fantasy_gp)
            new_bests.append(nb)
        new_bests = jnp.stack(new_bests)  # shape (n_samples,)

        return jnp.mean(new_bests) - current_best

    def _sample_from_posterior(self, X_batch: jnp.ndarray, n_samples: int) -> jnp.ndarray:
        """
        Draw samples from gp's joint posterior at X_batch, shape (q, d+1).
        Returns shape (n_samples, q).
        """
        q = X_batch.shape[0]
        x_dim = self.gp.dim
        # separate inputs vs. fidelity
        x_only = X_batch[:, :x_dim]
        fid_cont = X_batch[:, x_dim]
        # Map continuous -> discrete fidelity
        fid_index = jnp.clip(
            jnp.floor(fid_cont * self.gp.fidelities).astype(int),
            0, self.gp.fidelities - 1
        )
        # We do a quick joint_predict for x_only, ignoring fidelity. 
        # Or handle separate blocks if the batch has mixed fidelities.
        mu, cov = self.gp.joint_predict(x_only)  # shape (q,), (q,q)
        L = jnp.linalg.cholesky(cov + 1e-9 * jnp.eye(q))

        rng_key = jax.random.PRNGKey(0)  # or pass a real key
        z = jax.random.normal(rng_key, (n_samples, q))
        samples = mu + z @ L.T
        return samples

    def _build_fantasy_gp(self, X_batch: jnp.ndarray, Y_batch: jnp.ndarray) -> AutoRegressiveMFGP:
        """
        Return a 'fantasy' GP that includes (X_batch, Y_batch) as newly observed data
        in their respective fidelities.
        """
        from copy import deepcopy
        fantasy_dataset = deepcopy(self.gp.dataset)
        x_dim = self.gp.dim
        q = X_batch.shape[0]

        for i in range(q):
            x_i = X_batch[i, :x_dim]
            fid_cont = X_batch[i, x_dim]
            fid_i = int(jnp.clip(jnp.floor(fid_cont * self.gp.fidelities), 0, self.gp.fidelities - 1))
            ds = fantasy_dataset.get_data(fid_i)
            new_x = jnp.concatenate([ds.x_train, x_i[None, :]], axis=0)
            new_y = jnp.concatenate([ds.y_train, Y_batch[i:i+1]], axis=0)
            fantasy_dataset.set_data(fid_i, ds._replace(x_train=new_x, y_train=new_y))

        # Build new MFGP with same hyperparams
        fantasy_gp = AutoRegressiveMFGP(
            mf_dataset=fantasy_dataset,
            kernels=self.gp.kernels,
            noise=self.gp.noises,
            rho=self.gp.rhos
        )
        # Just update K_invs with the new data; skip hyperparam re-optimization
        fantasy_gp._update_K_invs()
        return fantasy_gp

    def _current_best(self) -> float:
        """
        Return current best observed value among all fidelities,
        or specifically highest fidelity if preferred.
        """
        # Example: best among all fidelities
        best_vals = []
        for f in range(self.gp.fidelities):
            ds_f = self.dataset.get_data(f)
            if ds_f.y_train.size > 0:
                best_vals.append(ds_f.y_train.max())
        return jnp.max(jnp.array(best_vals)) if best_vals else 0.0

    def _approx_best(self, gp: AutoRegressiveMFGP) -> float:
        """
        Approximate the best achievable value under gp, e.g. by sampling random points
        at highest fidelity. Return a scalar float.
        """
        # Suppose we only care about best at highest fidelity
        n_random = 200
        d = gp.dim
        fid_high = gp.fidelities - 1

        key = jax.random.PRNGKey(0)
        X_rand = jax.random.uniform(key, shape=(n_random, d))
        # Evaluate at highest fidelity
        mu_rand, _ = gp.predict(X_rand)  # or partial approach
        return jnp.max(mu_rand)