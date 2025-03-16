import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
from baox.surrogate.kernel import MaternKernel, RBFKernel
from baox.surrogate.gp import SingleOuputGP
from baox.surrogate.mf_gp import AutoRegressiveMFGP
from baox.test_functions import MFLinearA, MFLinearB
from baox.utils import generate_dataset, denormalize
import copy
from baox.data_types import MultiFidelityDataset

bounds = (0, 1)
kernel = MaternKernel(lengthscale=jnp.array([1.0]), variance=1.0)

for f in [MFLinearA, MFLinearB]:
    
    low_f = generate_dataset(f.low_f, jnp.array([bounds]), 16, seed=2)
    high_f = generate_dataset(f.high_f, jnp.array([bounds]), 5, seed=2)

    # Train GP on Low-Fidelity Data Only
    gp_low = SingleOuputGP(
        low_f.x_train, 
        low_f.y_train, 
        copy.deepcopy(kernel), 
        noise=1e-2
    )
    gp_low.fit()

    # Train GP on High-Fidelity Data Only
    gp_high = SingleOuputGP(
        high_f.x_train, 
        high_f.y_train,
        copy.deepcopy(kernel), 
        noise=1e-2
    )
    gp_high.fit()

    # Train Multi-Fidelity GP
    mf_dataset = MultiFidelityDataset([low_f, high_f])
    mfgp = AutoRegressiveMFGP(mf_dataset, kernel, noise=1e-2)
    mfgp.fit()

    # Generate test points for prediction
    X_test = jnp.linspace(0, 1, 100).reshape(-1, 1)
    x = denormalize(X_test, bounds[0], bounds[1])
    mu_low, var_low = gp_low.predict(X_test)
    mu_high, var_high = gp_high.predict(X_test)
    mu_mf, var_mf = mfgp.predict(X_test)

    # Compute absolute errors for comparison
    error_low = jnp.abs(mu_low - f.high_f(x))
    error_high = jnp.abs(mu_high - f.high_f(x))
    error_mf = jnp.abs(mu_mf - f.high_f(x))

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # First subplot: Low-Fidelity GP
    ax: plt.Axes = axes[0, 0]
    ax.scatter(low_f.x, low_f.y_train, color="black", marker="o", label="Low-Fidelity Data")
    ax.scatter(high_f.x, high_f.y_train, color="black", marker="x", label="High-Fidelity Data")
    ax.plot(x.flatten(), f.low_f(x), color="black", linestyle="--", label="Low-Fidelity Function")
    ax.plot(x.flatten(), f.high_f(x), color="black", linestyle="-", label="High-Fidelity Function")
    ax.plot(x.flatten(), mu_low, color="blue", linestyle="-", label="GP (Low-Fidelity)")
    ax.fill_between(x.flatten(), mu_low - 1.96 * jnp.sqrt(var_low), 
                    mu_low + 1.96 * jnp.sqrt(var_low), color="blue", alpha=0.2)
    ax.set_title("GP on Low-Fidelity Data")
    ax.legend()
    ax.grid()

    # Second subplot: High-Fidelity GP
    ax: plt.Axes = axes[0, 1]
    ax.scatter(low_f.x, low_f.y_train, color="black", marker="o", label="Low-Fidelity Data")
    ax.scatter(high_f.x, high_f.y_train, color="black", marker="x", label="High-Fidelity Data")
    ax.plot(x.flatten(), f.low_f(x), color="black", linestyle="--", label="Low-Fidelity Function")
    ax.plot(x.flatten(), f.high_f(x), color="black", linestyle="-", label="High-Fidelity Function")
    ax.plot(x.flatten(), mu_high, color="red", linestyle="-", label="GP (High-Fidelity)")
    ax.fill_between(x.flatten(), mu_high - 1.96 * jnp.sqrt(var_high), 
                    mu_high + 1.96 * jnp.sqrt(var_high), color="red", alpha=0.2)
    ax.set_title("GP on High-Fidelity Data")
    ax.legend()
    ax.grid()

    # Third subplot: Multi-Fidelity GP
    ax: plt.Axes = axes[1, 0]
    ax.scatter(low_f.x, low_f.y_train, color="black", marker="o", label="Low-Fidelity Data")
    ax.scatter(high_f.x, high_f.y_train, color="black", marker="x", label="High-Fidelity Data")
    ax.plot(x.flatten(), f.low_f(x), color="black", linestyle="--", label="Low-Fidelity Function")
    ax.plot(x.flatten(), f.high_f(x), color="black", linestyle="-", label="High-Fidelity Function")
    ax.plot(x.flatten(), mu_mf, color="green", label="MF-GP Prediction")
    ax.fill_between(x.flatten(), mu_mf - 1.96 * jnp.sqrt(var_mf), 
                    mu_mf + 1.96 * jnp.sqrt(var_mf), color="green", alpha=0.2)
    ax.set_title("Multi-Fidelity GP")
    ax.legend()
    ax.grid()

    # Fourth subplot: Error Comparison
    ax: plt.Axes = axes[1, 1]
    ax.plot(x.flatten(), error_low, color="blue", linestyle="-", label="GP (Low-Fidelity) Error")
    ax.plot(x.flatten(), error_high, color="red", linestyle="-", label="GP (High-Fidelity) Error")
    ax.plot(x.flatten(), error_mf, color="green", linestyle="-", label="MF-GP Error")
    ax.set_title("Error Comparison (w.r.t High-Fidelity)")
    ax.set_xlabel("X")
    ax.set_ylabel("Absolute Error")
    ax.legend()
    ax.grid()

    plt.tight_layout()
    class_name = f.__name__
    plt.savefig(f"03_1D_mfgp_{class_name}.png")
