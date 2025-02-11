import jax.numpy as jnp
from typing import Callable, Tuple
import jax


class BaseOptimization:
    def __init__(self, objective: Callable[[jnp.ndarray], float], bounds: Tuple[float, float], n_iter: int = 20):
        self.objective = objective
        self.bounds = bounds
        self.n_iter = n_iter
        self.X_train = jnp.array([]).reshape(-1, 1)
        self.y_train = jnp.array([])
    
    def step(self, key: jax.random.PRNGKey):
        raise NotImplementedError("Subclasses must implement step method")
    
    def run(self, key: jax.random.PRNGKey):
        keys = jax.random.split(key, self.n_iter)
        for n in range(self.n_iter):
            self.step(keys[n], n)
        return self.X_train, self.y_train