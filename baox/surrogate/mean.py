import jax.numpy as jnp
from typing import Callable, Optional


class ConstantMean:
    def __init__(self, constant: float = 0.0):
        self.constant = constant

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        return jnp.full(X.shape[0], self.constant)