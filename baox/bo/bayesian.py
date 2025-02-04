import jax.numpy as jnp
from jax.scipy.stats import norm
from typing import Callable, Tuple
from baox.surrogate.gp import GaussianProcess
from baox.surrogate.kernel import RBFKernel, MaternKernel
import jax
import jax.random as random
import matplotlib.pyplot as plt

class BayesianOptimization:
    def __init__(self, objective: Callable[[jnp.ndarray], float], bounds: Tuple[float, float], 
                 kernel=None, noise: float = 1e-1):
        self.objective = objective
        self.bounds = bounds
        self.noise = noise
        self.kernel = kernel if kernel else MaternKernel(lengthscale=1.0, variance=1.0, nu=1.5)
        self.gp = GaussianProcess(self.kernel, noise=noise)
        self.X_train = jnp.array([]).reshape(-1, 1)
        self.y_train = jnp.array([])
    
    def acquisition_function(self, X: jnp.ndarray, xi: float = 0.01) -> jnp.ndarray:
        """Expected Improvement (EI) acquisition function."""
        mu, sigma = self.gp.predict(X)
        sigma = jnp.maximum(sigma, 1e-9)  # Avoid division by zero
        f_best = jnp.max(self.y_train) if self.y_train.size > 0 else 0.0
        Z = (mu - f_best - xi) / sigma
        return (mu - f_best - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
    
    def sample_gp_prediction(self, X: jnp.ndarray, key: jax.random.PRNGKey, n_samples: int = 10) -> jnp.ndarray:
        """Sample from the GP posterior distribution using Monte Carlo sampling."""
        mu, sigma = self.gp.predict(X)
        sigma = jnp.maximum(sigma, 1e-9)  # Avoid zero variance
        return mu[:, None] + jnp.sqrt(sigma[:, None]) * random.normal(key, shape=(X.shape[0], n_samples))
    
    def acquisition_mc(self, X: jnp.ndarray, key: jax.random.PRNGKey, n_samples: int = 10) -> jnp.ndarray:
        """Monte Carlo-based Expected Improvement (MC-EI)."""
        samples = self.sample_gp_prediction(X, key, n_samples)
        f_best = jnp.max(self.y_train) if self.y_train.size > 0 else 0.0
        improvement = jnp.maximum(samples - f_best, 0)
        return jnp.mean(improvement, axis=1)  # MC estimate of EI
    
    def optimize_acquisition(self, key: jax.random.PRNGKey, num_candidates: int = 100, use_mc: bool = False, n_samples: int = 10) -> jnp.ndarray:
        """Find the next point by maximizing the acquisition function using either analytical EI or MC-based EI."""
        X_candidates = jnp.linspace(self.bounds[0], self.bounds[1], num_candidates).reshape(-1, 1)
        if use_mc:
            acq_values = self.acquisition_mc(X_candidates, key, n_samples)
        else:
            acq_values = self.acquisition_function(X_candidates)
        return X_candidates[jnp.argmax(acq_values)]
    
    def step(self, key: jax.random.PRNGKey, use_mc: bool = False, n_samples: int = 10, step: int = 0) -> None:
        """Perform one iteration of Bayesian Optimization."""
        if self.X_train.size == 0:
            self.X_train = jax.random.uniform(key, (5, ), minval=self.bounds[0], maxval=self.bounds[1]).reshape(-1, 1)
            self.y_train = self.objective(self.X_train).flatten()
        else:
            X_next = self.optimize_acquisition(key, use_mc=use_mc, n_samples=n_samples).reshape(-1, 1)
            self.X_train = jnp.concatenate([self.X_train, X_next], axis=0)
            self.y_train = jnp.concatenate([self.y_train, self.objective(X_next).flatten()], axis=0)
        
        
        self.gp.fit(self.X_train, self.y_train)
        self.plot_iteration(step)
    
    def plot_iteration(self, step: int):
        """Visualize the GP model and acquisition function after each iteration."""
        X_test = jnp.linspace(self.bounds[0], self.bounds[1], 500).reshape(-1, 1)
        mu, sigma = self.gp.predict(X_test)
        print(f"Step {step}: Best point found: {self.X_train[jnp.argmax(self.y_train)]}")
        plt.figure(figsize=(10, 5))
        plt.fill_between(X_test.flatten(), mu - 1.96 * jnp.sqrt(sigma), mu + 1.96 * jnp.sqrt(sigma), alpha=0.2, label="Confidence Interval")
        plt.plot(X_test, mu, label="GP Mean Prediction", color="blue")
        plt.scatter(self.X_train[:-1], self.y_train[:-1], color="black", label="Training Points")
        plt.scatter(self.X_train[-1], self.y_train[-1], color="red", label="Observed Points")
        # plot true function
        plt.plot(X_test, self.objective(X_test), color="black", linestyle="--", label="True Function")
        
        plt.legend()
        plt.grid()
        plt.title(f"Bayesian Optimization Progress (Step {step})")
        plt.xlabel("X")
        plt.ylabel("Objective Function")
        plt.savefig(f'iteration_{step}.png')
        plt.close()
    
    def run(self, key: jax.random.PRNGKey, n_iter: int = 10, use_mc: bool = False, n_samples: int = 10):
        """Run Bayesian Optimization for n_iter iterations."""
        keys = jax.random.split(key, n_iter)
        for n in range(n_iter):
            self.step(keys[n], use_mc=use_mc, n_samples=n_samples, step=n)
        return self.X_train, self.y_train