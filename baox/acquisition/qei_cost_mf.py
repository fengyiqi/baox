import jax.numpy as jnp
import jax
import optax
from baox.acquisition.base import BaseAcquisitionFunction
from baox.surrogate import AutoRegressiveMFGP
from typing import Tuple

def kmeans_clustering(points: jnp.ndarray, num_clusters: int, num_iters: int = 10) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Perform simple k-means clustering on the given points using JAX.

    This function initializes cluster centers by randomly selecting points, then iteratively
    refines the centers by assigning each point to its closest center and recomputing the centers
    as the mean of the assigned points.

    Args:
        points (jnp.ndarray): Array of shape (n_points, d) containing the points to cluster.
        num_clusters (int): The desired number of clusters.
        num_iters (int): The number of iterations for the k-means algorithm. Default is 10.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]:
            - centers: Array of shape (num_clusters, d) representing the final cluster centers.
            - assignments: Array of shape (n_points,) containing the cluster index for each point.
    """
    key = jax.random.PRNGKey(0)
    init_indices = jax.random.choice(key, points.shape[0], shape=(num_clusters,), replace=False)
    centers = points[init_indices]

    for _ in range(num_iters):
        dists = jnp.linalg.norm(points[:, None, :] - centers[None, :, :], axis=-1)
        assignments = jnp.argmin(dists, axis=1)
        def compute_center(j):
            mask = assignments == j
            return jnp.where(jnp.sum(mask) > 0, jnp.mean(points[mask], axis=0), centers[j])
        centers = jnp.stack([compute_center(j) for j in range(num_clusters)], axis=0)
    return centers, assignments

    

class qCostAwareMultiFidelityEI(BaseAcquisitionFunction):
    """
    Joint acquisition function for cost-aware multi-fidelity optimization using Expected Improvement (EI).

    This acquisition function proposes a batch of candidate points, each augmented with a fidelity
    indicator. The candidate vector is of shape (batch_size, d+1), where the first d components are
    the input location (in normalized [0,1]^d space) and the last component is a continuous fidelity indicator.
    The fidelity indicator is later thresholded (e.g., <0.5 → low fidelity, ≥0.5 → high fidelity) and mapped
    to an associated evaluation cost.

    Attributes:
        gp (AutoRegressiveMFGP): A trained auto-regressive multi-fidelity Gaussian Process model.
        dataset: The dataset associated with the GP.
        n_samples (int): Number of Monte Carlo samples for the qEI estimate.
        n_restarts (int): Number of random restarts for candidate optimization.
        n_steps (int): Number of optimization steps per restart.
        lr (float): Learning rate for the optimizer.
        costs: A list of cost values corresponding to each fidelity level.
        num_fidelities (int): The number of fidelity levels.
    """

    def __init__(self, gp: AutoRegressiveMFGP, n_samples: int = 512,
                 n_restarts: int = 32, n_steps: int = 256, lr: float = 1e-2):
        super().__init__(gp)
        self.dataset = gp.dataset
        self.gp = gp
        self.n_samples = n_samples
        self.n_restarts = n_restarts
        self.n_steps = n_steps
        self.lr = lr
        self.costs = self.dataset.list_costs()
        self.num_fidelities = len(self.costs)
        

    def propose_candidates(self, key: jax.random.PRNGKey, batch_size: int) -> jnp.ndarray:
        """
        Initialize the cost-aware multi-fidelity Expected Improvement (EI) acquisition function.

        Args:
            gp (AutoRegressiveMFGP): The multi-fidelity GP model.
            n_samples (int): Number of Monte Carlo samples to use for qEI estimation. Default is 512.
            n_restarts (int): Number of random restarts for the optimization process. Default is 32.
            n_steps (int): Number of optimization steps per restart. Default is 256.
            lr (float): Learning rate for the optimizer. Default is 1e-2.
        """
        d = self.gp.dim 

        def objective(x: jnp.ndarray) -> jnp.ndarray:
            qEI_value = self._qEI_mc(x, key)
            qEI_value = jnp.nan_to_num(qEI_value, nan=-1e6)
            return -qEI_value

        optimizer = optax.adam(self.lr)

        def single_restart(subkey: jax.random.PRNGKey):
            x0 = jax.random.uniform(subkey, shape=(batch_size, d + 1))
            x0_flat = x0
            opt_state = optimizer.init(x0_flat)

            def step_fn(carry, _):
                x_flat, opt_state = carry
                val, grads = jax.value_and_grad(objective)(x_flat)
                updates, opt_state = optimizer.update(grads, opt_state)
                x_flat = optax.apply_updates(x_flat, updates)
                x_flat = jnp.clip(x_flat, 0.0, 1.0)
                return (x_flat, opt_state), val

            (xf, _), vals = jax.lax.scan(step_fn, (x0_flat, opt_state), jnp.arange(self.n_steps))
            final_val = vals[-1]
            return xf, final_val

        subkeys = jax.random.split(key, self.n_restarts)
        final_xs, final_vals = jax.vmap(single_restart)(subkeys)
        best_idx = jnp.argmin(final_vals)  # because we minimize the negative objective
        best_flat = final_xs[best_idx]
        
        
        # Flatten all candidate batches from restarts: shape (n_restarts * batch_size, d+1)
        # all_candidates = final_xs.reshape(-1, d + 1)

        # # Apply k-means clustering on the input space (first d dimensions)
        # # TODO consider fidelity and all training data
        # centers, assignments = kmeans_clustering(all_candidates[:, :d], batch_size, num_iters=10)

        # # For each cluster, pick the candidate closest to the cluster center (including fidelity indicator)
        # def select_candidate(j):
        #     mask = assignments == j
        #     candidates_in_cluster = all_candidates[mask]
        #     if candidates_in_cluster.shape[0] == 0:
        #         # Fallback: return the j-th candidate from all_candidates
        #         return all_candidates[j]
        #     dists = jnp.linalg.norm(candidates_in_cluster[:, :d] - centers[j][None, :], axis=1)
        #     best_idx = jnp.argmin(dists)
        #     return candidates_in_cluster[best_idx]

        # selected = jnp.stack([select_candidate(j) for j in range(batch_size)], axis=0)
        return best_flat

    def _qEI_mc(self, X: jnp.ndarray, key: jax.random.PRNGKey) -> jnp.ndarray:
        """
        Estimate the cost-aware qEI (expected improvement) using Monte Carlo sampling.

        For each candidate in the batch, the method computes the improvement over the best observed
        function value for its corresponding fidelity level, adjusted by the evaluation cost.

        Args:
            X (jnp.ndarray): Candidate points of shape (q, d+1), where q is the batch size.
            key (jax.random.PRNGKey): A PRNG key for random sampling.

        Returns:
            jnp.ndarray: The Monte Carlo estimate of the cost-aware qEI value.
        """
        q = X.shape[0]
        x_test = X[:, :-1]  # shape (q, d)
        fid_ind = X[:, -1]  # shape (q,)
        
        mu, var_s = self.gp.predict(x_test)
        
        fidelity = jnp.floor(fid_ind * self.num_fidelities).astype(int)
        fidelity = jnp.clip(fidelity, 0, self.num_fidelities - 1)
        costs_array = jnp.array(self.costs)
        cost_per_candidate = costs_array[fidelity]
        
        all_y_train = jnp.array([jnp.max(self.dataset.get_data(i).y_train) for i in range(self.num_fidelities)])
        f_best_per_candidate = all_y_train[fidelity]  # shape (q,)
        
        # Since covariance matrix is not positive definite, 
        # we use the variance to sample from the posterior.
        key, subkey = jax.random.split(key)
        z = jax.random.normal(subkey, (self.n_samples, q))
        f_samples = mu[None, :] + z * jnp.sqrt(var_s)[None, :]
        
        improvement = jnp.maximum(f_samples - f_best_per_candidate, 0.0)
        improvement = improvement
        
        qEI_cost = jnp.mean(improvement) / jnp.mean(cost_per_candidate)
        
        return qEI_cost
    
