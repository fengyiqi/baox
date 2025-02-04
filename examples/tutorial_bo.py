import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"

import jax
jax.config.update("jax_enable_x64", True)

from baox.bo.bayesian import BayesianOptimization
import jax.numpy as jnp

# Example usage
def objective_function(x):
    return -jnp.sin(3 * x) - x**2 + 0.7 * x

bounds = (-2, 2)
key = jax.random.key(0)
bo = BayesianOptimization(objective_function, bounds)
X_samples, y_samples = bo.run(key, n_iter=15, use_mc=True, n_samples=10)

print(f"Best point found: {X_samples[jnp.argmax(y_samples)]}")
