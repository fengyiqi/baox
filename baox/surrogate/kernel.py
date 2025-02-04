import jax.numpy as jnp
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