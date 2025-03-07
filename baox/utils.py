from baox.data_types import Dataset
from typing import Callable, Tuple, List, NamedTuple
import jax.numpy as jnp
from scipy.stats.qmc import LatinHypercube
import warnings
import jax


def normalize(X: jnp.ndarray, min_val: jnp.ndarray, max_val: jnp.ndarray) -> jnp.ndarray:
    """
    Normalize the input data X based on provided minimum and maximum values.

    The normalization is done using the formula:
        X_norm = (X - min_val) / (max_val - min_val)

    Args:
        X (jnp.ndarray): Input array to be normalized.
        min_val (jnp.ndarray): Minimum values for each dimension.
        max_val (jnp.ndarray): Maximum values for each dimension.

    Returns:
        jnp.ndarray: The normalized array with values scaled to [0, 1].
    """
    return (X - min_val) / (max_val - min_val)


def denormalize(X_norm: jnp.ndarray, min_val: jnp.ndarray, max_val: jnp.ndarray) -> jnp.ndarray:
    """
    Denormalize the normalized data back to the original scale.

    The denormalization is done using the formula:
        X = X_norm * (max_val - min_val) + min_val

    Args:
        X_norm (jnp.ndarray): Normalized data array.
        min_val (jnp.ndarray): Original minimum values for each dimension.
        max_val (jnp.ndarray): Original maximum values for each dimension.

    Returns:
        jnp.ndarray: The denormalized data array.
    """
    return X_norm * (max_val - min_val) + min_val


def generate_dataset(
    objective: Callable[[jnp.ndarray], jnp.ndarray],
    x_bounds: List[Tuple[float, float]],
    num_samples: int = 5,
    random: bool = True,
    seed: int = 0,
    noise_level: float = None
) -> Dataset:
    """
    Generate a dataset using Latin Hypercube Sampling for a given objective function.

    This function generates sample points in the input space based on the provided bounds
    using Latin Hypercube Sampling. The sampled points are then denormalized according to
    the bounds and evaluated using the objective function to produce corresponding outputs.
    Optionally, Gaussian noise can be added to the outputs.

    Args:
        objective (Callable[[jnp.ndarray], jnp.ndarray]): 
            A function mapping input points (x) to output values (y).
        x_bounds (List[Tuple[float, float]]): 
            A list of tuples specifying the lower and upper bounds for each input dimension.
        num_samples (int, optional): 
            Number of samples to generate. Defaults to 5.
        random (bool, optional): 
            If True, use Latin Hypercube Sampling for generating inputs. 
            If False, use evenly spaced samples (only supported for 1D). Defaults to True.
        seed (int, optional): 
            Random seed for reproducibility. Defaults to 0.
        noise_level (float, optional): 
            Standard deviation of Gaussian noise to add to the outputs. 
            If None, no noise is added.

    Returns:
        Dataset: A Dataset namedtuple containing:
            - x: The denormalized input points with shape [num_samples, d].
            - x_train: The normalized input points with shape [num_samples, d].
            - y_train: The corresponding outputs computed by the objective function, 
              with shape [num_samples].

    Raises:
        AssertionError: If non-random sampling is requested for multi-dimensional inputs.
    """
    # Extract lower and upper bounds as arrays
    x_min = jnp.array([b[0] for b in x_bounds])
    x_max = jnp.array([b[1] for b in x_bounds])
    dim = len(x_bounds)
    
    if random:
        if num_samples > 30:
            warnings.warn(f"Note that {num_samples} samples are generated randomly")
        sampler = LatinHypercube(d=dim, seed=seed)
        x_norm = jnp.array(sampler.random(n=num_samples))
    else:
        assert dim == 1, "Only 1D is supported for non-random sampling."
        x_norm = jnp.linspace(0, 1, num_samples).reshape(-1, 1)
        
    # Denormalize sampled points based on provided bounds
    x = denormalize(x_norm, x_min, x_max)
    # Compute outputs by applying the objective function
    y = objective(x).flatten()
    # Optionally add Gaussian noise
    if noise_level is not None:
        y += noise_level * jax.random.normal(jax.random.PRNGKey(seed), y.shape)
    
    return Dataset(x, x_norm, y)
