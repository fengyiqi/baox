import jax.numpy as jnp
from baox.surrogate.gp import SingleOuputGP
import jax


class BaseAcquisitionFunction:
    def __init__(self, gp: SingleOuputGP):
        self.gp = gp

    def evaluate(self, X: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError("Subclasses must implement evaluate method")

    def propose_candidates(self, key: jax.random.PRNGKey, *args) -> jnp.ndarray:
        raise NotImplementedError(
            "Subclasses must implement propose_candidates method")
        