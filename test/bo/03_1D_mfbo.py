import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from baox.surrogate import AutoRegressiveMFGP, MaternKernel
from baox.data_types import MultiFidelityDataset
from baox.utils import generate_dataset
from baox.test_functions import MFForrester
from baox.bo import MultiFidelityBayesianOptimization

bounds = (-0.8, 1)

# Generate initial data for four fidelities:
data_0 = generate_dataset(MFForrester.f_0, jnp.array([bounds]), 12, cost=1.0)
data_1 = generate_dataset(MFForrester.f_1, jnp.array([bounds]), 6, cost=2.0)
data_2 = generate_dataset(MFForrester.f_2, jnp.array([bounds]), 4, cost=4.0)
data_3 = generate_dataset(MFForrester.f_3, jnp.array([bounds]), 2, cost=4.0)

# Create a Multi-Fidelity Dataset with four fidelities.
mf_dataset = MultiFidelityDataset([data_0, data_1, data_2, data_3])

kernel = MaternKernel(lengthscale=jnp.array([1.0]), variance=1.0)
mfgp = AutoRegressiveMFGP(mf_dataset, kernel, noise=0.01)
mfgp.fit() 

objectives = {
    0: MFForrester.f_0, 
    1: MFForrester.f_1, 
    2: MFForrester.f_2, 
    3: MFForrester.f_3
}

mfbo = MultiFidelityBayesianOptimization(
    objectives=objectives,
    bounds=jnp.array([bounds]),
    gp=mfgp,
    batch_size=5,  # number of batch candidate at a time
    n_iter=20   
)

key = jax.random.key(0)
mfbo.run(key)