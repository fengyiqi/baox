import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
from baox.surrogate.kernel import MaternKernel
from baox.surrogate.gp import GaussianProcess
from baox.surrogate.mf_gp import AutoRegressiveMFGP

# Generate synthetic low-fidelity and high-fidelity data
key = random.PRNGKey(0)

def high_fidelity_function(X):
    """Simulated high-fidelity function."""
    return (6.0 * X - 2.0)**2 * jnp.sin(12.0 * X - 4.0)
    # return 5.0 * X**2 * jnp.sin(12.0 * X)

def low_fidelity_function(X):
    """Simulated low-fidelity function."""
    return 0.5 * high_fidelity_function(X) + 10.0 * (X - 0.5) + 5.0
    # return 2.0 * high_fidelity_function(X) + (X**3 - 0.5) * jnp.sin(3.0 * X - 0.5) + 4.0 * jnp.cos(2.0 * X)

bounds = (0, 1)

X_low = jnp.linspace(bounds[0], bounds[1], 12).reshape(-1, 1)
y_low = low_fidelity_function(X_low)

X_high = random.uniform(key, (4,), minval=bounds[0], maxval=bounds[1]).reshape(-1, 1)
X_high = jnp.linspace(bounds[0], bounds[1], 4).reshape(-1, 1)
y_high = high_fidelity_function(X_high)

# Define kernels
kernel_low = MaternKernel(lengthscale=1.0, variance=1.0)
kernel_high = MaternKernel(lengthscale=1.0, variance=1.0)

# Train GP on Low-Fidelity Data Only
gp_low = GaussianProcess(kernel_low, noise=1e-2)
gp_low.fit(X_low, y_low.flatten())

# Train GP on High-Fidelity Data Only
gp_high = GaussianProcess(kernel_high, noise=1e-2)
gp_high.fit(X_high, y_high.flatten())

# Train Multi-Fidelity GP
kernel_low = MaternKernel(lengthscale=1.0, variance=1.0)
kernel_high = MaternKernel(lengthscale=1.0, variance=1.0)
mfgp = AutoRegressiveMFGP(kernel_low, kernel_high)
mfgp.fit(X_low, y_low, X_high, y_high, lr=0.01, steps=200)

# Generate test points for prediction
X_test = jnp.linspace(bounds[0], bounds[1], 100).reshape(-1, 1)
mu_low, var_low = gp_low.predict(X_test)
mu_high, var_high = gp_high.predict(X_test)
mu_mf, var_mf = mfgp.predict(X_test)

# Compute absolute errors for comparison
error_low = jnp.abs(mu_low - high_fidelity_function(X_test.flatten()))
error_high = jnp.abs(mu_high - high_fidelity_function(X_test.flatten()))
error_mf = jnp.abs(mu_mf - high_fidelity_function(X_test.flatten()))

# Plot results
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# First subplot: Low-Fidelity GP
ax = axes[0, 0]
ax.scatter(X_low, y_low, color="black", marker="o", label="Low-Fidelity Data")
ax.scatter(X_high, y_high, color="black", marker="x", label="High-Fidelity Data")
ax.plot(X_test.flatten(), low_fidelity_function(X_test.flatten()), color="black", linestyle="--", label="Low-Fidelity Function")
ax.plot(X_test.flatten(), high_fidelity_function(X_test.flatten()), color="black", linestyle="-", label="High-Fidelity Function")
ax.plot(X_test.flatten(), mu_low, color="blue", linestyle="-", label="GP (Low-Fidelity)")
ax.fill_between(X_test.flatten(), mu_low - 1.96 * jnp.sqrt(var_low), 
                mu_low + 1.96 * jnp.sqrt(var_low), color="blue", alpha=0.2)
ax.set_title("GP on Low-Fidelity Data")
ax.legend()
ax.grid()

# Second subplot: High-Fidelity GP
ax = axes[0, 1]
ax.scatter(X_low, y_low, color="black", marker="o", label="Low-Fidelity Data")
ax.scatter(X_high, y_high, color="black", marker="x", label="High-Fidelity Data")
ax.plot(X_test.flatten(), low_fidelity_function(X_test.flatten()), color="black", linestyle="--", label="Low-Fidelity Function")
ax.plot(X_test.flatten(), high_fidelity_function(X_test.flatten()), color="black", linestyle="-", label="High-Fidelity Function")
ax.plot(X_test.flatten(), mu_high, color="red", linestyle="-", label="GP (High-Fidelity)")
ax.fill_between(X_test.flatten(), mu_high - 1.96 * jnp.sqrt(var_high), 
                mu_high + 1.96 * jnp.sqrt(var_high), color="red", alpha=0.2)
ax.set_title("GP on High-Fidelity Data")
ax.legend()
ax.grid()

# Third subplot: Multi-Fidelity GP
ax = axes[1, 0]
ax.scatter(X_low, y_low, color="black", marker="o", label="Low-Fidelity Data")
ax.scatter(X_high, y_high, color="black", marker="x", label="High-Fidelity Data")
ax.plot(X_test.flatten(), low_fidelity_function(X_test.flatten()), color="black", linestyle="--", label="Low-Fidelity Function")
ax.plot(X_test.flatten(), high_fidelity_function(X_test.flatten()), color="black", linestyle="-", label="High-Fidelity Function")
ax.plot(X_test.flatten(), mu_mf, color="green", label="MF-GP Prediction")
ax.fill_between(X_test.flatten(), mu_mf - 1.96 * jnp.sqrt(var_mf), 
                mu_mf + 1.96 * jnp.sqrt(var_mf), color="green", alpha=0.2)
ax.set_title("Multi-Fidelity GP")
ax.legend()
ax.grid()

# Fourth subplot: Error Comparison
ax = axes[1, 1]
ax.plot(X_test.flatten(), error_low, color="blue", linestyle="-", label="GP (Low-Fidelity) Error")
ax.plot(X_test.flatten(), error_high, color="red", linestyle="-", label="GP (High-Fidelity) Error")
ax.plot(X_test.flatten(), error_mf, color="green", linestyle="-", label="MF-GP Error")
ax.set_title("Error Comparison (w.r.t High-Fidelity)")
ax.set_xlabel("X")
ax.set_ylabel("Absolute Error")
ax.legend()
ax.grid()

plt.tight_layout()
plt.savefig("mfgp_comparison_A.png")
