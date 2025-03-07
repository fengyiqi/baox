import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
from baox.surrogate.kernel import MaternKernel
from baox.surrogate import AutoRegressiveMFGP, SingleOuputGP
from baox.utils import generate_dataset, denormalize
import copy
from baox.data_types import MultiFidelityDataset
from baox.test_functions import MFForrester

bounds = (-1, 1)
kernel = MaternKernel(lengthscale=jnp.array([1.0]), variance=1.0)
    
data_0 = generate_dataset(MFForrester.f_0, jnp.array([bounds]), 48, seed=1)
data_1 = generate_dataset(MFForrester.f_1, jnp.array([bounds]), 24, seed=1)
data_2 = generate_dataset(MFForrester.f_2, jnp.array([bounds]), 18, seed=1)
data_3 = generate_dataset(MFForrester.f_3, jnp.array([bounds]), 8, seed=1)

gp = SingleOuputGP(data_3.x_train, data_3.y_train, copy.deepcopy(kernel), noise=1e-2)
gp.fit(steps=1000)

# Train Multi-Fidelity GP
mf_dataset = MultiFidelityDataset([data_0, data_1, data_2, data_3])
mfgp = AutoRegressiveMFGP(mf_dataset, copy.deepcopy(kernel), noise=1e-2)
mfgp.fit(steps=1000)

# Generate test points for prediction
X_test = jnp.linspace(0, 1, 100).reshape(-1, 1)
x = denormalize(X_test, bounds[0], bounds[1])
mu, var = gp.predict(X_test)
mu_mf, var_mf = mfgp.predict(X_test)
error = jnp.abs(mu - MFForrester.f_3(x).flatten())
error_mf = jnp.abs(mu_mf - MFForrester.f_3(x).flatten())

# Plot results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].plot(x, MFForrester.f_3(x), label="Fidelity 3", color="black", linestyle="--", linewidth=2)
axes[0].scatter(data_3.x, data_3.y_train, color="black", marker="x")
axes[0].plot(x, mu, color="orange", label="GP", linewidth=2)
axes[0].fill_between(x.flatten(), mu - 2 * jnp.sqrt(var), mu + 2 * jnp.sqrt(var), color="orange", alpha=0.2)
axes[0].set_title("Single-Fidelity GP")
axes[0].legend()

axes[1].plot(x, MFForrester.f_3(x), label="Fidelity 3", color="black", linestyle="--", linewidth=2)
axes[1].scatter(data_3.x, data_3.y_train, color="black", marker="x")
axes[1].plot(x, MFForrester.f_2(x), label="Fidelity 2", color="red", linestyle="--")
axes[1].scatter(data_2.x, data_2.y_train, color="red", marker="x")
axes[1].plot(x, MFForrester.f_1(x), label="Fidelity 1", color="blue", linestyle="--")
axes[1].scatter(data_1.x, data_1.y_train, color="blue", marker="x")
axes[1].plot(x, MFForrester.f_0(x), label="Fidelity 0", color="green", linestyle="--")
axes[1].scatter(data_0.x, data_0.y_train, color="green", marker="x")

axes[1].plot(x, mu_mf, color="purple", label="MF-GP",  linewidth=2)
axes[1].fill_between(x.flatten(), mu_mf - 2 * jnp.sqrt(var_mf), mu_mf + 2 * jnp.sqrt(var_mf), color="purple", alpha=0.2)
axes[1].set_title("Multi-Fidelity GP")
axes[1].legend()

axes[2].plot(x, error, color="orange", label="GP - Fidelity 3")
axes[2].plot(x, error_mf, color="purple", label="MFGP - Fidelity 3")
axes[2].set_title("Absolute Error")
axes[2].legend()

plt.tight_layout()
plt.savefig(f"05_1D_mfgp_4.png")
