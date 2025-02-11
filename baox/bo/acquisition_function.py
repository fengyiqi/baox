import jax.numpy as jnp
from jax.scipy.stats import norm
from baox.surrogate.gp import GaussianProcess
import jax
import jax.random as random
from jax.scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


class BaseAcquisitionFunction:
    def __init__(self, gp: GaussianProcess):
        self.gp = gp
    
    def evaluate(self, X: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError("Subclasses must implement evaluate method")

class ExpectedImprovement(BaseAcquisitionFunction):

    def __init__(self, gp: GaussianProcess, xi: float = 0.01):
        super().__init__(gp)
        self.xi = xi

    def propose_candidates(self, X_candidates: jnp.ndarray, step_i: int, *args) -> jnp.ndarray:
        """Selects the best candidate maximizing EI."""
        acq_values = self.evaluate(X_candidates)
        return X_candidates[jnp.argmax(acq_values)]
      
    def evaluate(self, X: jnp.ndarray) -> jnp.ndarray:
        mu, sigma = self.gp.predict(X)
        sigma = jnp.maximum(sigma, 1e-9)
        f_best = jnp.max(self.gp.y_train) if self.gp.y_train.size > 0 else 0.0
        Z = (mu - f_best - self.xi) / sigma
        return (mu - f_best - self.xi) * norm.cdf(Z) + sigma * norm.pdf(Z)

class qExpectedImprovement(BaseAcquisitionFunction):

    def __init__(self, gp: GaussianProcess, xi: float = 0.01, n_samples: int = 100):
        super().__init__(gp)
        self.xi = xi
        self.n_samples = n_samples

    def propose_candidates(self, X_candidates: jnp.ndarray, progress: float, key: jax.random.PRNGKey, batch_size: int) -> jnp.ndarray:
        """Selects a batch of candidates maximizing qEI, ensuring they haven't been evaluated before."""
        mask = jnp.all(jnp.abs(X_candidates[:, None] - self.gp.X_train[None, :]) > 1e-6, axis=1)
        X_filtered = X_candidates[mask].reshape(-1, X_candidates.shape[-1])
        
        if X_filtered.shape[0] == 0:
            return X_candidates[jnp.argsort(self.evaluate(X_candidates, key))[-batch_size:]]
        
        acq_values = self.evaluate(X_filtered, progress, key)

        if batch_size > 1:
            # select maximum EI and cluster the rest
            top_indices = jnp.argsort(acq_values)[-len(acq_values) // 5:-1] 
            selected_candidates = X_candidates[top_indices]
            kmeans = KMeans(n_clusters=batch_size - 1, n_init=1).fit(selected_candidates)
            cluster_centers = jnp.array(kmeans.cluster_centers_)
            cluster_centers = jnp.concatenate([X_filtered[jnp.argsort(acq_values)][-1:], cluster_centers], axis=0)
            return cluster_centers
        else:
            top_indices = jnp.argsort(acq_values)[-batch_size:]
            return X_filtered[top_indices]
    
    def evaluate(self, X: jnp.ndarray, progress: float, key: jax.random.PRNGKey) -> jnp.ndarray:
        samples, sigma = self.sample_gp_prediction(X, key, self.n_samples)
        f_best = jnp.max(self.gp.y_train) if self.gp.y_train.size > 0 else 0.0
        improvement = jnp.maximum(samples - f_best, 0)

        exploration_bonus = sigma / jnp.max(sigma)  # Normalize exploration term
        alpha_max = 0.1
        beta = 5.0
        alpha = alpha_max * jnp.exp(-beta * progress)
        return jnp.mean(improvement, axis=1) + alpha * exploration_bonus
    
    def sample_gp_prediction(self, X: jnp.ndarray, key: jax.random.PRNGKey, n_samples: int) -> jnp.ndarray:
        mu, sigma = self.gp.predict(X)
        sigma = jnp.maximum(sigma, 1e-9)
        return mu[:, None] + jnp.sqrt(sigma[:, None]) * random.normal(key, shape=(X.shape[0], n_samples)), sigma