import jax.numpy as jnp
from typing import NamedTuple, Dict, Optional, Union, List
from collections import namedtuple
import warnings

class Dataset(NamedTuple):
    """
    Container for dataset arrays.

    Attributes:
        x (jnp.ndarray): Input data array (could be used for test or auxiliary data).
        x_train (jnp.ndarray): Training input data array.
        y_train (jnp.ndarray): Training output data array.
    """
    x: jnp.ndarray
    x_train: jnp.ndarray
    y_train: jnp.ndarray

class GPHyperparameters(NamedTuple):
    """
    Container for Gaussian Process hyperparameters.

    Attributes:
        lengthscale (jnp.ndarray): Lengthscale parameter(s) of the kernel.
        variance (jnp.ndarray): Variance parameter of the kernel.
        noise (jnp.ndarray): Noise parameter of the GP.
    """
    lengthscale: jnp.ndarray
    variance: jnp.ndarray
    noise: jnp.ndarray
    
class MFGPHyperparameters(NamedTuple):
    """
    Container for Gaussian Process hyperparameters.

    Attributes:
        lengthscale (jnp.ndarray): Lengthscale parameter(s) of the kernel.
        variance (jnp.ndarray): Variance parameter of the kernel.
        noise (jnp.ndarray): Noise parameter of the GP.
    """
    lengthscale: jnp.ndarray
    variance: jnp.ndarray
    noise: jnp.ndarray
    rho: jnp.ndarray
    
    def __str__(self):
        return (
            f"{self.__class__.__name__}:\n"
            f"\tLengthscale: {[*self.lengthscale.tolist()]}\n"
            f"\tVariance: {self.variance}\n"
            f"\tNoise: {self.noise}\n"
            f"\tRho: {self.rho}"
        )

class MultiFidelityDataset:
    """
    Container class to store training data for multiple fidelity levels.

    Each fidelity level is represented by an integer key and maps to a FidelityData
    namedtuple containing the data arrays for that fidelity.
    
    Attributes:
        _data_dict (Dict[int, FidelityData]): Internal dictionary mapping a fidelity level
            to its corresponding FidelityData.
    """
    
    def __init__(self, initial_data: Optional[Union[Dataset, List[Dataset]]] = None):
        """
        Initialize a MultiFidelityDataset.

        Args:
            initial_data (Dataset or List[Dataset], optional): Either a single Dataset instance
                (stored at fidelity level 0) or a list of Dataset instances. In the list case,
                each Dataset is stored with its index in the list as the fidelity level.
        """
        self._data_dict: Dict[int, Dataset] = {}

        if initial_data is not None:
            # Check if initial_data is a list of Dataset instances.
            if isinstance(initial_data, list) and all(isinstance(ds, Dataset) for ds in initial_data):
                for fidelity, ds in enumerate(initial_data):
                    self._data_dict[fidelity] = Dataset(ds.x, ds.x_train, ds.y_train)
            # Else if it's a single Dataset instance.
            elif isinstance(initial_data, Dataset):
                self._data_dict[0] = Dataset(initial_data.x, initial_data.x_train, initial_data.y_train)
                warnings.warn("Single Dataset provided. Assuming fidelity level 0.")
            else:
                raise ValueError("initial_data must be either a Dataset or a list of Dataset instances.")

    def add_data(self, fidelity: int, dataset: Dataset) -> None:
        """
        Add new data points to the specified fidelity level using a Dataset instance.

        If the fidelity level already exists, the new data is concatenated with the existing data.

        Args:
            fidelity (int): Integer index of the fidelity level.
            dataset (Dataset): A Dataset instance containing the data arrays to add.
        """
        if not isinstance(dataset, Dataset):
            raise ValueError("The provided dataset must be a Dataset instance.")
            
        if fidelity not in self._data_dict:
            self._data_dict[fidelity] = Dataset(dataset.x, dataset.x_train, dataset.y_train)
        else:
            current_data = self._data_dict[fidelity]
            x = jnp.concatenate([current_data.x, dataset.x], axis=0)
            x_train = jnp.concatenate([current_data.x_train, dataset.x_train], axis=0)
            y_train = jnp.concatenate([current_data.y_train, dataset.y_train], axis=0)
            self._data_dict[fidelity] = Dataset(x, x_train, y_train)

    def get_data(self, fidelity: int) -> Dataset:
        """
        Retrieve the FidelityData for the specified fidelity level.

        Args:
            fidelity (int): The fidelity level to retrieve.

        Returns:
            FidelityData: A namedtuple containing the x, x_train, and y_train arrays 
            for the given fidelity level.

        Raises:
            ValueError: If the specified fidelity level does not exist.
        """
        if fidelity not in self._data_dict:
            raise ValueError(f"Fidelity {fidelity} not found in the dataset.")
        return self._data_dict[fidelity]

    def list_fidelities(self) -> list:
        """
        List all fidelity levels that currently have data.

        Returns:
            list: A sorted list of fidelity levels present in the dataset.
        """
        return sorted(self._data_dict.keys())
    
    def __str__(self) -> None:
        """
        Print a summary of the dataset, showing the number of samples for each fidelity level.
        """
        for fidelity, data in self._data_dict.items():
            print(f"Fidelity {fidelity}: {data.x_train.shape[0]} samples.")

