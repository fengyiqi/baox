import jax.numpy as jnp
from jax.scipy.stats import norm
from typing import Callable, Tuple
from baox.surrogate.gp import GaussianProcess
from baox.surrogate.kernel import RBFKernel, MaternKernel
import jax
import jax.random as random
import matplotlib.pyplot as plt
from baox.base_optimization import BaseOptimization
from baox.bo.acquisition_function import ExpectedImprovement, qExpectedImprovement


class BayesianOptimization(BaseOptimization):
    def __init__(self, objective: Callable[[jnp.ndarray], float], bounds: Tuple[float, float], kernel=None, noise: float = 1e-1, batch_size: int = None, n_iter: int = 20):
        super().__init__(objective, bounds, n_iter)
        self.noise = noise
        self.kernel = kernel if kernel else MaternKernel(lengthscale=1.0, variance=1.0, nu=1.5)
        self.gp = GaussianProcess(self.kernel, noise=noise)
        self.acquisition = ExpectedImprovement(self.gp) if batch_size is None else qExpectedImprovement(self.gp)
        self.batch_size = 1 if batch_size is None else batch_size
        self.n_iter = n_iter
    
    def optimize_acquisition(self, key: jax.random.PRNGKey, step_i: int, num_candidates: int = 256) -> jnp.ndarray:
        """Finds the next candidate points by maximizing the acquisition function."""
        assert self.batch_size < num_candidates // 5, "Batch size must be less than 1/5 of the number of candidates due to KMeans clustering"
        X_candidates = jnp.linspace(self.bounds[0], self.bounds[1], num_candidates).reshape(-1, 1)
        # X_candidates = jnp.random.uniform(key, (num_candidates, 1), minval=self.bounds[0], maxval=self.bounds[1])
        return self.acquisition.propose_candidates(X_candidates, step_i / self.n_iter, key, self.batch_size)

    
    def step(self, key: jax.random.PRNGKey, step_i: int):
        print(f"Running step {step_i}")
        if self.X_train.size == 0:
            self.X_train = jax.random.uniform(key, (5,), minval=self.bounds[0], maxval=self.bounds[1]).reshape(-1, 1)
            self.y_train = self.objective(self.X_train).flatten()
        else:
            X_next = self.optimize_acquisition(key, step_i).reshape(-1, 1)
            self.X_train = jnp.concatenate([self.X_train, X_next], axis=0)
            self.y_train = jnp.concatenate([self.y_train, self.objective(X_next).flatten()], axis=0)
        
        self.gp.fit(self.X_train, self.y_train)
        self.plot_iteration(step_i)
    
    def plot_iteration(self, step_i: int):
        X_test = jnp.linspace(self.bounds[0], self.bounds[1], 500).reshape(-1, 1)
        mu, sigma = self.gp.predict(X_test)
        plt.figure(figsize=(10, 5))
        plt.fill_between(X_test.flatten(), mu - 1.96 * jnp.sqrt(sigma), mu + 1.96 * jnp.sqrt(sigma), alpha=0.2, label="Confidence Interval")
        plt.plot(X_test, mu, label="GP Mean Prediction", color="blue")
        plt.scatter(self.X_train[:-self.batch_size], self.y_train[:-self.batch_size], color="black", label="Training Points")
        plt.scatter(self.X_train[-self.batch_size:], self.y_train[-self.batch_size:], color="red", label="Observed Points")
        plt.plot(X_test, self.objective(X_test), color="black", linestyle="--", label="True Function")
        plt.legend(loc="upper left")
        # plt.ylim(-11, 5)
        plt.grid()
        plt.title(f"BO with {self.acquisition.__class__.__name__} (Step {step_i})")
        plt.xlabel("X")
        plt.ylabel("Objective Function")
        plt.tight_layout()
        plt.savefig(f'{self.acquisition.__class__.__name__}_cand_{self.batch_size}_{step_i}.png')
        plt.close()
        

