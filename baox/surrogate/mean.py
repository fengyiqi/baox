import jax.numpy as jnp
from typing import Callable, Optional

class ConstantMean:
    """
    Constant mean function for Gaussian Process models.

    This mean function returns a constant value for every input. It is typically used as a baseline
    mean function when no specific prior information about the data is available.

    Attributes:
        constant (float): The constant value to return.
    """
    
    def __init__(self, constant: float = 0.0):
        """
        Initialize the ConstantMean function.

        Args:
            constant (float): The constant value that the mean function returns. Defaults to 0.0.
        """
        self.constant = constant

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the constant mean for the given input data.

        Args:
            X (jnp.ndarray): Input data of shape [n, d] or [n], where n is the number of samples.
        
        Returns:
            jnp.ndarray: A 1D array of length n, where every element is equal to the constant.
        """
        return jnp.full(X.shape[0], self.constant)
