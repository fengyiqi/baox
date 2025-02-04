import jax.numpy as jnp
from baox.surrogate.kernel import RBFKernel, MaternKernel
import matplotlib.pyplot as plt
from baox.surrogate.gp import GaussianProcess
import jax

if __name__ == "__main__":
    key = jax.random.key(0)
    X_train = jnp.array([[-2.2], [-2.0], [-0.9], [0.6], [1.1], [2.0], [3.0]])
    y_train = jnp.sin(X_train).flatten()

    kernel_configs = [
        ("blue", RBFKernel(lengthscale=1.0, variance=1.0)),
        ("green", MaternKernel(lengthscale=1.0, variance=1.0, nu=1.5)),
        ("red", MaternKernel(lengthscale=1.0, variance=1.0, nu=2.5))
    ]

    plt.figure(figsize=(8, 5))
    plt.scatter(X_train, y_train, color="black", label="Training Data")
    X_test = jnp.linspace(-4, 4, 100).reshape(-1, 1)
    Y_test = jnp.sin(X_test).flatten()
    plt.plot(X_test, Y_test, color="black", linestyle="--", label="True Function")

    for color, kernel in kernel_configs:
        gp = GaussianProcess(kernel=kernel, noise=1e-3)
        gp.fit(X_train, y_train)

        mu_s, var_s = gp.predict(X_test)

        label = f"{kernel.__class__.__name__}"
        if isinstance(kernel, MaternKernel):
            label += f" (nu={kernel.nu})"
        print(f"{kernel.__class__.__name__}\tLengthscale: {kernel.lengthscale}\tVariance: {kernel.variance}")

        plt.plot(X_test, mu_s, label=label, color=color)
        plt.fill_between(X_test.flatten(), mu_s - 1.96 * jnp.sqrt(var_s), mu_s + 1.96 * jnp.sqrt(var_s), alpha=0.2, color=color)
    
    plt.legend()
    plt.title("Gaussian Process Regression with Different Kernels")
    plt.savefig("gp_regression.png")
    plt.show()