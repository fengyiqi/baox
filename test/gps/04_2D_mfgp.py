import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
from baox.surrogate.kernel import MaternKernel
from baox.surrogate.gp import SingleOuputGP
from baox.surrogate.mf_gp import AutoRegressiveMFGP
from baox.utils import normalize, generate_dataset
from baox.test_functions import MFBranin
from baox.data_types import MultiFidelityDataset
    


# Generate training data
num_low = 64  
num_high = 8
seed = 2

low_f = generate_dataset(MFBranin.low_f, [(-5, 10), (0, 15)], num_low, seed=seed)
high_f = generate_dataset(MFBranin.high_f, [(-5, 10), (0, 15)], num_high, seed=seed)

# Train single-fidelity GPs
gp_low = SingleOuputGP(low_f.x_train, low_f.y_train, MaternKernel(lengthscale=jnp.array([1.0, 1.0]), variance=1.0), noise=1e-2)
gp_low.fit()

gp_high = SingleOuputGP(high_f.x_train, high_f.y_train, MaternKernel(lengthscale=jnp.array([1.0, 1.0]), variance=1.0), noise=1e-2)
gp_high.fit()

# Define kernels for low- and high-fidelity functions
kernel_low = MaternKernel(lengthscale=jnp.array([1.0, 1.0]), variance=1.0)
kernel_high = MaternKernel(lengthscale=jnp.array([1.0, 1.0]), variance=1.0)
mfgp = AutoRegressiveMFGP(
    MultiFidelityDataset([low_f, high_f]),
    [kernel_low, kernel_high])
mfgp.fit()

# Generate test points
x1_range = jnp.linspace(-5, 10, 50)
x2_range = jnp.linspace(0, 15, 50)
X1, X2 = jnp.meshgrid(x1_range, x2_range)

X_test = jnp.column_stack((X1.ravel(), X2.ravel()))

# Compute true function values
y_true = MFBranin.high_f(X_test).reshape(X1.shape)
y_low_fidelity = MFBranin.low_f(X_test).reshape(X1.shape)

X_test = normalize(X_test, jnp.array([-5, 0]), jnp.array([10, 15]))
# Predictions
mu_low, _ = gp_low.predict(X_test)
mu_high, _ = gp_high.predict(X_test)
mu_mf, _ = mfgp.predict(X_test)

# Compute absolute error
error_low = jnp.abs(mu_low - y_true.ravel()).reshape(X1.shape)
error_high = jnp.abs(mu_high - y_true.ravel()).reshape(X1.shape)
error_mf = jnp.abs(mu_mf - y_true.ravel()).reshape(X1.shape)

# Reshape for plotting
mu_low = mu_low.reshape(X1.shape)
mu_high = mu_high.reshape(X1.shape)
mu_mf = mu_mf.reshape(X1.shape)

fig, axes = plt.subplots(3, 3, figsize=(18, 15))
ax = axes[0, 0]
c0 = ax.contourf(X1, X2, y_low_fidelity, levels=jnp.linspace(0, 250, 51), cmap="viridis", extend="max")
fig.colorbar(c0, ax=ax)
ax.set_title("Low-Fidelity Function")

ax = axes[0, 1]
c1 = ax.contourf(X1, X2, y_true, levels=jnp.linspace(0, 250, 51), cmap="viridis", extend="max")
fig.colorbar(c1, ax=ax)
ax.set_title("High-Fidelity Function")

axes[0, 2].axis("off")

# First row: Low-Fidelity GP
ax = axes[1, 0]
c2 = ax.contourf(X1, X2, mu_low, levels=jnp.linspace(0, 250, 51), cmap="viridis", extend="both")
fig.colorbar(c2, ax=ax)
ax.scatter(low_f.x[:, 0], low_f.x[:, 1], c="red", label="Low-Fidelity Data", s=20)
ax.set_title("Low-Fidelity GP Prediction")
ax.legend(loc="upper left")

ax = axes[2, 0]
c3 = ax.contourf(X1, X2, error_low, levels=jnp.linspace(0, 80, 41), cmap="Greys", extend="max")
fig.colorbar(c3, ax=ax)
ax.set_title("Low-Fidelity GP Error")

# Second row: High-Fidelity GP
ax = axes[1, 1]
c4 = ax.contourf(X1, X2, mu_high, levels=jnp.linspace(0, 250, 51), cmap="viridis", extend="both")
fig.colorbar(c4, ax=ax)
ax.scatter(high_f.x[:, 0], high_f.x[:, 1], c="white", label="High-Fidelity Data", s=20)
ax.set_title("High-Fidelity GP Prediction")
ax.legend(loc="upper left")

ax = axes[2, 1]
c5 = ax.contourf(X1, X2, error_high, levels=jnp.linspace(0, 80, 41), cmap="Greys", extend="max")
fig.colorbar(c5, ax=ax)
ax.set_title("High-Fidelity GP Error")

# Third row: Multi-Fidelity GP
ax = axes[1, 2]
c6 = ax.contourf(X1, X2, mu_mf, levels=jnp.linspace(0, 250, 51), cmap="viridis", extend="both")
fig.colorbar(c6, ax=ax)
ax.scatter(low_f.x[:, 0], low_f.x[:, 1], c="red", label="Low-Fidelity Data", s=20, alpha=0.6)
ax.scatter(high_f.x[:, 0], high_f.x[:, 1], c="white", label="High-Fidelity Data", s=20)
ax.set_title("Multi-Fidelity GP Prediction")
ax.legend(loc="upper left")

ax = axes[2, 2]
c7 = ax.contourf(X1, X2, error_mf, levels=jnp.linspace(0, 80, 41), cmap="Greys", extend="max")
fig.colorbar(c7, ax=ax)
ax.set_title("Multi-Fidelity GP Error")

plt.tight_layout()
plt.savefig("04_2D_mfgp.png")
