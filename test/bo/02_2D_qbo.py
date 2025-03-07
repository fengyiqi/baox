import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"

import jax
jax.config.update("jax_enable_x64", True)

from baox.bo import BayesianOptimization
from baox.acquisition import qExpectedImprovementFantasy, qExpectedImprovementJoint
import jax.numpy as jnp
from baox.test_functions import currin



bounds = [[0, 1], [0, 1]]
key = jax.random.key(42)

# analytical EI
bo = BayesianOptimization(currin, bounds, n_iter=60)
X_eval, y_eval = bo.run(key)
X_opt, y_opt = X_eval[jnp.argmax(y_eval)], jnp.max(y_eval)
print("Optimal solution:", X_opt, y_opt)

# MC-EI (qEI with batch size 1)
bo_batch = BayesianOptimization(currin, bounds, n_iter=20, batch_size=3, acquisition=qExpectedImprovementFantasy)
X_eval, y_eval = bo_batch.run(key)
X_opt_batch, y_opt_batch = X_eval[jnp.argmax(y_eval)], jnp.max(y_eval)
print("Optimal solution:", X_opt_batch, y_opt_batch)

# qEI with batch size 3
bo_batch = BayesianOptimization(currin, bounds, n_iter=20, batch_size=3, acquisition=qExpectedImprovementJoint)
X_eval, y_eval = bo_batch.run(key)
X_opt_batch, y_opt_batch = X_eval[jnp.argmax(y_eval)], jnp.max(y_eval)
print("Optimal solution:", X_opt_batch, y_opt_batch)
