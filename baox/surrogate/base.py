from baox.surrogate.kernel import KernelBase
from typing import Callable, Optional, Tuple, Union, List
import jax.numpy as jnp
from baox.surrogate.mean import ConstantMean
from baox.data_types import GPHyperparameters


class BaseGaussianProcess:
    def __init__(
        self, 
        kernel: KernelBase, 
        mean_function: Optional[Callable[[jnp.ndarray], jnp.ndarray]], 
        noise: float
    ):
        self.kernel = kernel
        self.mean_function = mean_function 
        self.noise = noise
        self.trainable_params: GPHyperparameters = GPHyperparameters(
            lengthscale=kernel.lengthscale,
            variance=kernel.variance,
            noise=noise
        )
