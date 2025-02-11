import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from baox.surrogate.kernel import RBFKernel, MaternKernel
import matplotlib.pyplot as plt
from baox.surrogate.gp import GaussianProcess

if __name__ == "__main__":
    key = jax.random.PRNGKey(0)

    # Original Training Data
    X_train = jnp.array([[-2.2], [-2.0], [-1.0], [-0.9], [0.6], [1.1], [2.0], [3.0]])
    y_train = jnp.sin(X_train).flatten()

    # Standardize X and y
    X_mean, X_std = X_train.mean(), X_train.std()
    y_mean, y_std = y_train.mean(), y_train.std()

    X_train_norm = (X_train - X_mean) / X_std
    y_train_norm = (y_train - y_mean) / y_std

    # Define Test Points
    X_test = jnp.linspace(-4, 4, 200).reshape(-1, 1)
    X_test_norm = (X_test - X_mean) / X_std
    Y_test = jnp.sin(X_test).flatten()

    kernel_configs = [
        ("blue", RBFKernel(lengthscale=1.0, variance=1.0)),
        ("green", MaternKernel(lengthscale=1.0, variance=1.0, nu=1.5)),
        ("red", MaternKernel(lengthscale=1.0, variance=1.0, nu=2.5))
    ]

    plt.figure(figsize=(10, 5))
    plt.scatter(X_train, y_train, color="black", label="Training Data")
    plt.plot(X_test, Y_test, color="black", linestyle="--", label="True Function")

    for color, kernel in kernel_configs:
        gp = GaussianProcess(kernel=kernel, noise=1e-3)
        gp.fit(X_train_norm, y_train_norm)

        mu_s_norm, var_s_norm = gp.predict(X_test_norm)

        # De-normalize predictions
        mu_s = mu_s_norm * y_std + y_mean
        var_s = var_s_norm * (y_std**2)

        label = f"{kernel.__class__.__name__}"
        if isinstance(kernel, MaternKernel):
            label += f" (nu={kernel.nu})"
        print(f"{kernel.__class__.__name__}\tLengthscale: {kernel.lengthscale}\tVariance: {kernel.variance}\tNoise: {gp.noise}")

        plt.plot(X_test, mu_s, label=label, color=color)
        plt.fill_between(X_test.flatten(), mu_s - 1.96 * jnp.sqrt(var_s), mu_s + 1.96 * jnp.sqrt(var_s), alpha=0.2, color=color)
    
    plt.legend()
    plt.title("Gaussian Process Regression")
    plt.grid()
    plt.tight_layout()
    plt.savefig("gp_regression.png")
    plt.show()