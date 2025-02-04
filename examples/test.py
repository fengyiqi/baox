import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.scipy.linalg import cho_solve, cholesky
from jax import grad
from typing import Callable, Optional


class KernelBase:
    def __init__(self, lengthscale: float = 1.0, variance: float = 1.0):
        self.lengthscale = lengthscale
        self.variance = variance

    def __call__(self, X1: jnp.ndarray, X2: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError(
            "Kernel function must be implemented in child classes")


class RBFKernel(KernelBase):
    def __call__(self, X1: jnp.ndarray, X2: jnp.ndarray) -> jnp.ndarray:
        sqdist = jnp.sum(X1**2, axis=1).reshape(-1, 1) + \
            jnp.sum(X2**2, axis=1) - 2 * jnp.dot(X1, X2.T)
        return self.variance * jnp.exp(-0.5 / self.lengthscale**2 * sqdist)


class MaternKernel(KernelBase):
    def __init__(self, lengthscale = 1, variance = 1, nu = 1.5):
        super().__init__(lengthscale, variance)
        self.nu = nu
    
    def __call__(self, X1: jnp.ndarray, X2: jnp.ndarray) -> jnp.ndarray:
        sqdist = jnp.sqrt(jnp.sum(X1**2, axis=1).reshape(-1, 1) +
                          jnp.sum(X2**2, axis=1) - 2 * jnp.dot(X1, X2.T))
        if self.nu == 1.5:
            scale = (1.0 + jnp.sqrt(3) * sqdist / self.lengthscale)
            return self.variance * scale * jnp.exp(-jnp.sqrt(3) * sqdist / self.lengthscale)
        elif self.nu == 2.5:
            scale = (1.0 + jnp.sqrt(5) * sqdist / self.lengthscale +
                     (5.0/3.0) * (sqdist**2) / (self.lengthscale**2))
            return self.variance * scale * jnp.exp(-jnp.sqrt(5) * sqdist / self.lengthscale)
        else:
            raise ValueError(
                "Currently, only Matern 1.5 and 2.5 are implemented")


class GaussianProcess:
    def __init__(self, kernel: KernelBase, mean_function: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None, noise: float = 1e-6):
        self.kernel = kernel
        self.mean_function = mean_function if mean_function else lambda x: jnp.zeros(
            x.shape[0])
        self.noise = noise
        self.X_train: Optional[jnp.ndarray] = None
        self.y_train: Optional[jnp.ndarray] = None
        self.K_inv: Optional[jnp.ndarray] = None

    def fit(self, X: jnp.ndarray, y: jnp.ndarray, lr: float = 0.01, steps: int = 50) -> None:
        self.X_train = X
        self.y_train = y

        def loss(params: jnp.ndarray) -> float:
            self.kernel.lengthscale, self.kernel.variance = params
            K = self.kernel(X, X) + self.noise * jnp.eye(X.shape[0])
            L = cholesky(K, lower=True)
            K_inv = cho_solve((L, True), jnp.eye(X.shape[0]))
            return 0.5 * y.T @ K_inv @ y + 0.5 * jnp.log(jnp.linalg.det(K))

        grad_loss = grad(loss)
        params = jnp.array([self.kernel.lengthscale, self.kernel.variance])
        for _ in range(steps):
            params -= lr * grad_loss(params)
            print(params)
        self.kernel.lengthscale, self.kernel.variance = params

        K = self.kernel(X, X) + self.noise * jnp.eye(X.shape[0])
        L = cholesky(K, lower=True)
        self.K_inv = cho_solve((L, True), jnp.eye(X.shape[0]))

    def predict(self, X_test: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        K_s = self.kernel(self.X_train, X_test)
        K_ss = self.kernel(X_test, X_test) + self.noise * \
            jnp.eye(X_test.shape[0])

        mean_train = self.mean_function(self.X_train)
        mean_test = self.mean_function(X_test)

        y_train_centered = self.y_train - mean_train
        mu_s = mean_test + K_s.T @ self.K_inv @ y_train_centered

        v = cho_solve((cholesky(K_ss, lower=True), True), K_s.T)
        cov_s = K_ss - K_s.T @ self.K_inv @ K_s

        return mu_s, jnp.diag(cov_s)


class ConstantMean:
    def __init__(self, constant: float = 0.0):
        self.constant = constant

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        return jnp.full(X.shape[0], self.constant)


if __name__ == "__main__":
    X_train = jnp.array([[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]])
    y_train = jnp.sin(X_train).flatten()

    kernel_configs = [
        ("blue", RBFKernel(lengthscale=1.0, variance=1.0)),
        ("green", MaternKernel(lengthscale=1.0, variance=1.0, nu=1.5)),
        ("red", MaternKernel(lengthscale=1.0, variance=1.0, nu=2.5))
    ]

    plt.figure(figsize=(8, 5))
    plt.scatter(X_train, y_train, color="black", label="Training Data")

    for color, kernel in kernel_configs:
        gp = GaussianProcess(kernel=kernel)
        gp.fit(X_train, y_train)

        X_test = jnp.linspace(-4, 4, 100).reshape(-1, 1)
        mu_s, var_s = gp.predict(X_test)

        label = f"{kernel.__class__.__name__}"
        if isinstance(kernel, MaternKernel):
            label += f" (nu={kernel.nu})"

        plt.plot(X_test, mu_s, label=label, color=color)
        plt.fill_between(X_test.flatten(), mu_s - 1.96 * jnp.sqrt(var_s), mu_s + 1.96 * jnp.sqrt(var_s), alpha=0.2, color=color)
    
    plt.legend()
    plt.title("Gaussian Process Regression with Different Kernels")
    plt.show()
