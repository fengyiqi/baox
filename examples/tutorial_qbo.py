import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"

import jax
jax.config.update("jax_enable_x64", True)

from baox.bo.bayesian import BayesianOptimization
from baox.surrogate.kernel import MaternKernel, RBFKernel
import jax.numpy as jnp

if __name__ == "__main__":
    def objective_function(x):
        # return -jnp.sin(3 * x) - x**2 + 0.7 * x
        # return jnp.sin(3 * x) * jnp.cos(2 * x) + jnp.sin(5 * x) + 0.2 * x
        # return jnp.sin(5 * x) * (1 - jnp.tanh(x**2)) + 0.2 * jnp.cos(3 * x)
        # return jnp.exp(-x**2) * jnp.sin(8 * x) + 0.1 * jnp.cos(3 * x) + 0.2 * jnp.sin(5 * x)
        return jnp.sin(4 * x) * jnp.exp(-0.2 * x**2) + 0.3 * jnp.cos(2.5 * x)
    
    bounds = (-10, 10)
    key = jax.random.PRNGKey(42)
    
    # analytical EI
    # bo = BayesianOptimization(objective_function, bounds, n_iter=50)
    # X_eval, y_eval = bo.run(key)
    # X_opt, y_opt = X_eval[jnp.argmax(y_eval)], jnp.max(y_eval)
    # print("Optimal solution:", X_opt, y_opt)
    
    # MC-EI (qEI with batch size 1)
    # bo_batch = BayesianOptimization(objective_function, bounds, batch_size=1)
    # X_eval, y_eval = bo_batch.run(key, n_iter=30)
    # X_opt_batch, y_opt_batch = X_eval[jnp.argmax(y_eval)], jnp.max(y_eval)
    # print("Optimal solution:", X_opt_batch, y_opt_batch)
    
    # qEI with batch size 3
    bo_batch = BayesianOptimization(objective_function, bounds, batch_size=5, n_iter=20)
    X_eval, y_eval = bo_batch.run(key)
    X_opt_batch, y_opt_batch = X_eval[jnp.argmax(y_eval)], jnp.max(y_eval)
    print("Optimal solution:", X_opt_batch, y_opt_batch)
