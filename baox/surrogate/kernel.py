import jax.numpy as jnp
from typing import Callable, Optional
from functools import partial
import jax

class KernelBase:
    """
    Base class for kernel functions used in Gaussian Process models.

    This class provides a basic structure for defining kernel functions,
    including a method to compute a scaled Mahalanobis distance. Child classes
    should implement the __call__ method to compute the kernel matrix.

    Attributes:
        lengthscale (jnp.ndarray): The lengthscale parameter(s) used for scaling distances.
        variance (jnp.ndarray): The variance parameter for the kernel.
    """
    
    def __init__(self, lengthscale: jnp.ndarray, variance: jnp.ndarray):
        """
        Initialize the KernelBase with given lengthscale and variance.

        Args:
            lengthscale (jnp.ndarray): Lengthscale parameter(s) for the kernel.
            variance (jnp.ndarray): Variance parameter for the kernel.
        """
        self.lengthscale = jnp.asarray(lengthscale)
        self.variance = jnp.asarray(variance)

    def compute_scaled_distance(
        self, 
        X1: jnp.ndarray, 
        X2: jnp.ndarray,
        lengthscale: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute the scaled distance between two sets of inputs using the squared Mahalanobis distance.

        The distance is computed by scaling the Euclidean differences by the provided lengthscale,
        squaring, summing over dimensions, and taking the square root. A minimum value is enforced
        for numerical stability.

        Args:
            X1 (jnp.ndarray): First set of inputs, shape [n1, d].
            X2 (jnp.ndarray): Second set of inputs, shape [n2, d].
            lengthscale (jnp.ndarray): The lengthscale parameter(s) for scaling the distances.

        Returns:
            jnp.ndarray: A distance matrix of shape [n1, n2] where each entry represents the scaled distance.
        """
        lengthscale = jnp.atleast_1d(lengthscale)  # Ensure broadcastability
        lengthscale_sq = lengthscale ** 2  # Square for proper scaling
        sqdist = jnp.sum(((X1[:, None, :] - X2[None, :, :]) ** 2) / lengthscale_sq, axis=-1)
        return jnp.sqrt(jnp.maximum(sqdist, 1e-9))  # Ensure numerical stability

    def __call__(self, X1: jnp.ndarray, X2: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the kernel matrix between two sets of inputs.

        This method should be implemented by child classes.

        Args:
            X1 (jnp.ndarray): First set of inputs.
            X2 (jnp.ndarray): Second set of inputs.

        Raises:
            NotImplementedError: Always, since this method must be implemented in subclasses.
        """
        raise NotImplementedError("Kernel function must be implemented in child classes")


class RBFKernel(KernelBase):
    """
    Radial Basis Function (RBF) kernel, also known as the Gaussian kernel.

    This kernel computes the similarity between inputs as an exponentially decaying function
    of the squared Euclidean distance.
    """

    def __init__(self, lengthscale=1, variance=1):
        """
        Initialize the RBFKernel with default or provided lengthscale and variance.

        Args:
            lengthscale (scalar or array-like): The lengthscale parameter. Default is 1.
            variance (scalar or array-like): The variance parameter. Default is 1.
        """
        super().__init__(lengthscale, variance)

    def __call__(
        self, 
        X1: jnp.ndarray, 
        X2: jnp.ndarray,
        lengthscale: jnp.ndarray = None,
        variance: jnp.ndarray = None,
    ) -> jnp.ndarray:
        """
        Compute the RBF kernel matrix between two sets of inputs.

        The RBF kernel is computed as:
            K(X1, X2) = variance * exp(-0.5 * (scaled_distance)**2)

        Args:
            X1 (jnp.ndarray): First set of inputs, shape [n1, d].
            X2 (jnp.ndarray): Second set of inputs, shape [n2, d].
            lengthscale (jnp.ndarray, optional): Lengthscale to use. If None, the instance's lengthscale is used.
            variance (jnp.ndarray, optional): Variance to use. If None, the instance's variance is used.

        Returns:
            jnp.ndarray: The computed RBF kernel matrix of shape [n1, n2].
        """
        if lengthscale is None:
            lengthscale = self.lengthscale
        if variance is None:
            variance = self.variance
        dists = self.compute_scaled_distance(X1, X2, lengthscale)
        return variance * jnp.exp(-0.5 * dists**2)


class MaternKernel(KernelBase):
    """
    Matern kernel with tunable smoothness parameter (nu).

    This kernel computes the covariance between inputs using the Matern function.
    Currently, only nu values of 1.5 and 2.5 are implemented.
    """

    def __init__(
        self, 
        lengthscale: jnp.ndarray = None, 
        variance: jnp.ndarray = None, 
        nu=1.5
    ):
        """
        Initialize the MaternKernel with specified lengthscale, variance, and nu.

        Args:
            lengthscale (jnp.ndarray, optional): Lengthscale parameter. Defaults to None.
            variance (jnp.ndarray, optional): Variance parameter. Defaults to None.
            nu (float): Smoothness parameter of the Matern kernel. Supported values are 1.5 and 2.5.
        """
        super().__init__(lengthscale, variance)
        self.nu = nu
    
    def __call__(
        self, 
        X1: jnp.ndarray, 
        X2: jnp.ndarray,
        lengthscale: jnp.ndarray = None,
        variance: jnp.ndarray = None,
    ) -> jnp.ndarray:
        """
        Compute the Matern kernel matrix between two sets of inputs.

        For nu=1.5, the kernel is given by:
            K(X1, X2) = variance * (1 + sqrt(3)*d) * exp(-sqrt(3)*d)
        For nu=2.5, the kernel is given by:
            K(X1, X2) = variance * (1 + sqrt(5)*d + (5/3)*d^2) * exp(-sqrt(5)*d)
        where d is the scaled distance computed using the base class method.

        Args:
            X1 (jnp.ndarray): First set of inputs, shape [n1, d].
            X2 (jnp.ndarray): Second set of inputs, shape [n2, d].
            lengthscale (jnp.ndarray, optional): Lengthscale to use. If None, the instance's lengthscale is used.
            variance (jnp.ndarray, optional): Variance to use. If None, the instance's variance is used.

        Returns:
            jnp.ndarray: The computed Matern kernel matrix of shape [n1, n2].

        Raises:
            ValueError: If nu is not 1.5 or 2.5.
        """
        if lengthscale is None:
            lengthscale = self.lengthscale
        if variance is None:
            variance = self.variance
        dists = self.compute_scaled_distance(X1, X2, lengthscale)

        if self.nu == 1.5:
            scale = (1.0 + jnp.sqrt(3) * dists)
            return variance * scale * jnp.exp(-jnp.sqrt(3) * dists)
        elif self.nu == 2.5:
            scale = (1.0 + jnp.sqrt(5) * dists + (5.0 / 3.0) * (dists**2))
            return variance * scale * jnp.exp(-jnp.sqrt(5) * dists)
        else:
            raise ValueError("Currently, only Matern 1.5 and 2.5 are implemented")
