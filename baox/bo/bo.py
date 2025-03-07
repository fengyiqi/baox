import jax.numpy as jnp
from jax.scipy.stats import norm
from typing import Callable, Tuple
from baox.surrogate.gp import SingleOuputGP
from baox.surrogate.kernel import RBFKernel, MaternKernel
import jax
import jax.random as random
import matplotlib.pyplot as plt
from baox.bo.base import BaseOptimization
from baox.acquisition import *
from baox.utils import generate_dataset, denormalize, normalize
from baox.data_types import Dataset
import warnings
from baox.surrogate.mf_gp import AutoRegressiveMFGP

PLOT = True

class BayesianOptimization(BaseOptimization):
    def __init__(
        self, 
        objective: Callable[[jnp.ndarray], float], 
        bounds: Tuple[float, float], 
        acquisition: BaseAcquisitionFunction = None,
        kernel=None, 
        noise: float = 1e-2, 
        batch_size: int = None, 
        n_iter: int = 20
    ):
        super().__init__(objective, bounds, n_iter)
        self.noise = noise
        self.kernel = kernel if kernel else MaternKernel(
            lengthscale=jnp.array([1.0]), variance=1.0, nu=1.5)
        self.dataset = generate_dataset(self.objective, self.bounds, 5)
        self.gp = SingleOuputGP(
            self.dataset.x_train, self.dataset.y_train, self.kernel, noise=self.noise)
        self.gp.fit()
        if acquisition is None or batch_size is None:
            warnings.warn("No acquisition function and batch size provided simultaneously. Using ExpectedImprovement with batch size 1.")
            self.batch_size = 1
            self.acquisition = ExpectedImprovement(self.gp)
        else:
            if isinstance(acquisition, ExpectedImprovement) and batch_size != 1:
                warnings.warn("ExpectedImprovement only supports batch size 1. Setting batch size to 1.")
                self.batch_size = 1
            else:
                self.batch_size = batch_size
            self.acquisition = acquisition(self.gp)
        self.n_iter = n_iter        

    def optimize_acquisition(self, key: jax.random.PRNGKey, step_i: int) -> jnp.ndarray:
        """Finds the next candidate points by maximizing the acquisition function."""

        return self.acquisition.propose_candidates(key, self.batch_size)

    def step(self, key: jax.random.PRNGKey, step_i: int):
        print(f"Running step {step_i}")
        
        X_next = self.optimize_acquisition(key, step_i).reshape(-1, len(self.bounds))
        x_original = denormalize(X_next, jnp.array(
            [x[0] for x in self.bounds]), jnp.array([x[1] for x in self.bounds]))
        y_next = self.objective(x_original).flatten()
        self.dataset = Dataset(
            x=jnp.concatenate([self.dataset.x, x_original], axis=0),
            x_train=jnp.concatenate(
                [self.dataset.x_train, X_next], axis=0),
            y_train=jnp.concatenate([self.dataset.y_train, y_next], axis=0)
        )
        self.gp.x_train = self.dataset.x_train
        self.gp.y_train = self.dataset.y_train 
        self.gp.fit()
        if PLOT:
            self.plot_iteration(step_i)

    def plot_iteration(self, step_i: int):
        if len(self.bounds) == 1:
            test_dataset = generate_dataset(
                self.objective, self.bounds, 200, False)
            x = test_dataset.x.flatten()
            mu, sigma = self.gp.predict(test_dataset.x_train)
            plt.figure(figsize=(10, 5))
            plt.fill_between(x, mu - 1.96 * jnp.sqrt(sigma), mu + 1.96 *
                            jnp.sqrt(sigma), alpha=0.2, label="Confidence Interval")
            plt.plot(x, mu, label="GP Mean Prediction", color="blue")
            plt.scatter(self.dataset.x[:-self.batch_size], self.dataset.y_train[:-
                        self.batch_size], color="black", label="Training Points")
            plt.scatter(self.dataset.x[-self.batch_size:],
                        self.dataset.y_train[-self.batch_size:], color="red", label="Observed Points")
            plt.plot(x, test_dataset.y_train, color="black",
                    linestyle="--", label="True Function")
            plt.legend(loc="upper left")
            # plt.ylim(-11, 5)
            plt.grid()
            plt.title(
                f"BO with {self.acquisition.__class__.__name__} (Step {step_i})")
            plt.xlabel("X")
            plt.ylabel("Objective Function")
            plt.tight_layout()
            plt.savefig(
                f'{self.acquisition.__class__.__name__}_cand_{self.batch_size}_{step_i}.png')
            plt.close()
        elif len(self.bounds) == 2:
            
            x1_range = jnp.linspace(self.bounds[0][0], self.bounds[0][1], 50)
            x2_range = jnp.linspace(self.bounds[1][0], self.bounds[1][1], 50)
            X1, X2 = jnp.meshgrid(x1_range, x2_range)
            X_test = jnp.column_stack((X1.ravel(), X2.ravel()))
            y_true = self.objective(X_test).reshape(X1.shape)
            X_test = normalize(X_test, jnp.array([self.bounds[0][0], self.bounds[1][0]]), jnp.array([self.bounds[0][1], self.bounds[1][1]]))
            mu, sigma = self.gp.predict(X_test)
            mu = mu.reshape(X1.shape)
            error = jnp.abs(mu - y_true).reshape(X1.shape)
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            ax = axes[0]
            c0 = ax.contourf(X1, X2, y_true, levels=50, cmap="viridis", extend="both")
            fig.colorbar(c0, ax=ax)
            ax.set_title("True Function")
            
            ax = axes[1]
            c1 = ax.contourf(X1, X2, mu, levels=50, cmap="viridis", extend="both")
            fig.colorbar(c1, ax=ax)
            ax.scatter(self.dataset.x[:-self.batch_size, 0], self.dataset.x[:-self.batch_size, 1], c="black", label="Training Points", s=20)
            ax.scatter(self.dataset.x[-self.batch_size:, 0], self.dataset.x[-self.batch_size:, 1], c="red", label="Observed Points", s=20)
            ax.set_title("GP Prediction")
            ax.legend(loc="upper left")
            
            ax = axes[2]
            c2 = ax.contourf(X1, X2, error, levels=50, cmap="Greys", extend="both")
            ax.scatter(self.dataset.x[:-self.batch_size, 0], self.dataset.x[:-self.batch_size, 1], c="black", label="Training Points", s=20)
            ax.scatter(self.dataset.x[-self.batch_size:, 0], self.dataset.x[-self.batch_size:, 1], c="red", label="Observed Points", s=20)
            fig.colorbar(c2, ax=ax)
            ax.set_title("GP Error")
            plt.tight_layout()
            plt.savefig(f'{self.acquisition.__class__.__name__}_cand_{self.batch_size}_{step_i}.png')
            plt.close()
            

# Multi-fidelity Bayesian Optimization class
class MultiFidelityBayesianOptimization(BaseOptimization):
    def __init__(
        self, 
        objective_high: Callable[[jnp.ndarray], float],
        objective_low: Callable[[jnp.ndarray], float],
        bounds: Tuple[Tuple[float, float], ...],
        acquisition: Callable = None,
        gp: AutoRegressiveMFGP=None,  # This should be your multi-fidelity GP instance (e.g., AutoRegressiveMFGP)
        batch_size: int = None,
        n_iter: int = 20,
        cost_low: float = 1.0,
        cost_high: float = 10.0
    ):
        # We use the high-fidelity objective as the main objective for BaseOptimization.
        super().__init__(objective_high, bounds, n_iter)
        if gp is None:
            raise ValueError("Please provide a multi-fidelity GP instance.")
        self.gp = gp
        self.cost_low = cost_low
        self.cost_high = cost_high
        self.objective_high = objective_high
        self.objective_low = objective_low

        if acquisition is None or batch_size is None:
            warnings.warn("No acquisition function and batch size provided. Using default qMixedFidelityCostAwareEIJoint with batch_size 1.")
            self.batch_size = 1
            self.acquisition = qMixedFidelityCostAwareEIJoint(self.gp, cost_low, cost_high)
        else:
            self.batch_size = batch_size
            self.acquisition = acquisition(self.gp, cost_low, cost_high)

        # Datasets for low- and high-fidelity evaluations.
        self.dataset_low = None  # will hold a Dataset instance for low fidelity
        self.dataset_high = None # for high fidelity

    def optimize_acquisition(self, key: jax.random.PRNGKey) -> jnp.ndarray:
        """Proposes a mixed batch of candidates.
        The returned array is of shape (batch_size, d+1), where the last dimension is the fidelity indicator.
        """
        return self.acquisition.propose_candidates(key, self.batch_size)

    def step(self, key: jax.random.PRNGKey, step_i: int):
        print(f"Running step {step_i}")
        d = len(self.bounds)
        # If no initial datasets exist, generate initial data for both fidelities.
        if self.dataset_low is None or self.dataset_high is None:
            num_init_low = 16
            num_init_high = 4
            self.dataset_low = generate_dataset(self.objective_low, self.bounds, num_init_low, seed=1)
            self.dataset_high = generate_dataset(self.objective_high, self.bounds, num_init_high, seed=4)
            self.gp.fit(self.dataset_low.x_train, self.dataset_low.y_train,
                    self.dataset_high.x_train, self.dataset_high.y_train)
        else:
            # Propose a batch of candidates (each candidate: d inputs + 1 fidelity indicator)
            X_mixed = self.optimize_acquisition(key)
            # Separate input and fidelity indicator.
            X_candidate = X_mixed[:, :d]
            fid_indicator = X_mixed[:, d]
            # Denormalize the input candidates.
            lower = jnp.array([b[0] for b in self.bounds])
            upper = jnp.array([b[1] for b in self.bounds])
            X_denorm = denormalize(X_candidate, lower, upper)
            # Assign fidelity: if indicator < 0.5 choose low fidelity; otherwise high.
            # (You can adjust the threshold as needed.)
            fidelities = jnp.where(fid_indicator < 0.5, 0, 1)
            # fidelities = jnp.array([1, *[0 for _ in range(self.batch_size - 1)]])
            print(X_denorm)
            print(fidelities)
            # Evaluate each candidate at the chosen fidelity and update the corresponding dataset.
            for i in range(self.batch_size):
                x = X_denorm[i:i+1, :]  # shape (1,d)
                if fidelities[i] == 1:
                    y = self.objective_high(x).flatten()
                    # Append to high-fidelity dataset.
                    self.dataset_high = Dataset(
                        x = jnp.concatenate([self.dataset_high.x, x], axis=0),
                        x_train = jnp.concatenate([self.dataset_high.x_train, X_candidate[i:i+1, :]], axis=0),
                        y_train = jnp.concatenate([self.dataset_high.y_train, y], axis=0)
                    )
                else:
                    y = self.objective_low(x).flatten()
                    # Append to low-fidelity dataset.
                    self.dataset_low = Dataset(
                        x = jnp.concatenate([self.dataset_low.x, x], axis=0),
                        x_train = jnp.concatenate([self.dataset_low.x_train, X_candidate[i:i+1, :]], axis=0),
                        y_train = jnp.concatenate([self.dataset_low.y_train, y], axis=0)
                    )
        # After updating, re-fit the multi-fidelity GP.
        # kernel_low = MaternKernel(lengthscale=jnp.array([1.0, 1.0]), variance=1.0)
        # kernel_high = MaternKernel(lengthscale=jnp.array([1.0, 1.0]), variance=1.0)
        # gp = AutoRegressiveMFGP(kernel_low, kernel_high)
        # self.gp = AutoRegressiveMFGP(kernel_low, kernel_high)
        self.gp.fit(self.dataset_low.x_train, self.dataset_low.y_train, self.dataset_high.x_train, self.dataset_high.y_train)
        
        if PLOT:
            self.plot_iteration(step_i)

    def plot_iteration(self, step_i: int):
        # Plotting similar to the single-fidelity case, but with separate markers for low and high fidelity.
        if len(self.bounds) == 1:
            test_dataset = generate_dataset(self.objective_high, self.bounds, 200, random=False)
            x = test_dataset.x.flatten()
            mu, cov, mu_low, cov_low, mu_delta, cov_delta = self.gp.joint_predict(test_dataset.x_train)
            sigma = jnp.diag(cov)
            sigma_low = jnp.diag(cov_low)
            plt.figure(figsize=(10, 5))
            plt.plot(x, test_dataset.y_train, color="black", linestyle="--", label="True Function")
            plt.plot(x, mu, label="MF mean", color="blue")
            plt.fill_between(x, mu - 1.96 * jnp.sqrt(sigma), mu + 1.96 *
                            jnp.sqrt(sigma), alpha=0.2, label="MF Confidence Interval", color="blue")
            plt.scatter(self.dataset_high.x, self.dataset_high.y_train,
                        color="blue", label="High-fidelity Points")
            
            plt.plot(x, mu_low, label="LF mean", color="green")
            plt.fill_between(x, mu_low - 1.96 * jnp.sqrt(sigma_low), mu_low + 1.96 *
                            jnp.sqrt(sigma_low), alpha=0.2, label="LF Confidence Interval", color="green")
            # Plot low-fidelity and high-fidelity points in different colors.
            plt.scatter(self.dataset_low.x, self.dataset_low.y_train,
                        color="green", label="Low-fidelity Points")
            
            # plt.plot(x, mu_delta, label="Delta mean", color="red")
            # plt.fill_between(x, mu_delta - 1.96 * jnp.sqrt(sigma_delta), mu_delta + 1.96 *
            #                 jnp.sqrt(sigma_delta), alpha=0.2, label="Delta Confidence Interval", color="red")
            
            plt.legend(loc="upper left")
            plt.grid()
            plt.title(f"MF-BO with {self.acquisition.__class__.__name__} (Step {step_i})")
            plt.xlabel("X")
            plt.ylabel("Objective Function")
            plt.tight_layout()
            plt.savefig(f'{self.acquisition.__class__.__name__}_cand_{self.batch_size}_{step_i}.png')
            plt.close()
        elif len(self.bounds) == 2:
            x1_range = jnp.linspace(self.bounds[0][0], self.bounds[0][1], 50)
            x2_range = jnp.linspace(self.bounds[1][0], self.bounds[1][1], 50)
            X1, X2 = jnp.meshgrid(x1_range, x2_range)
            X_test = jnp.column_stack((X1.ravel(), X2.ravel()))
            y_true = self.objective_high(X_test).reshape(X1.shape)
            X_test_norm = normalize(X_test, jnp.array([self.bounds[0][0], self.bounds[1][0]]),
                                      jnp.array([self.bounds[0][1], self.bounds[1][1]]))
            mu, sigma = self.gp.predict(X_test_norm)
            mu = mu.reshape(X1.shape)
            error = jnp.abs(mu - y_true).reshape(X1.shape)
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            ax = axes[0]
            c0 = ax.contourf(X1, X2, y_true, levels=50, cmap="viridis", extend="both")
            fig.colorbar(c0, ax=ax)
            ax.set_title("True Function")
            
            ax = axes[1]
            c1 = ax.contourf(X1, X2, mu, levels=50, cmap="viridis", extend="both")
            fig.colorbar(c1, ax=ax)
            if self.dataset_low is not None:
                ax.scatter(self.dataset_low.x[:, 0], self.dataset_low.x[:, 1], c="black", label="Low-fidelity", s=20)
            if self.dataset_high is not None:
                ax.scatter(self.dataset_high.x[:, 0], self.dataset_high.x[:, 1], c="red", label="High-fidelity", s=20)
            ax.set_title("GP Prediction")
            ax.legend(loc="upper left")
            
            ax = axes[2]
            c2 = ax.contourf(X1, X2, error, levels=50, cmap="Greys", extend="both")
            if self.dataset_low is not None:
                ax.scatter(self.dataset_low.x[:, 0], self.dataset_low.x[:, 1], c="black", label="Low-fidelity", s=20)
            if self.dataset_high is not None:
                ax.scatter(self.dataset_high.x[:, 0], self.dataset_high.x[:, 1], c="red", label="High-fidelity", s=20)
            fig.colorbar(c2, ax=ax)
            ax.set_title("GP Error")
            plt.tight_layout()
            plt.savefig(f'{self.acquisition.__class__.__name__}_cand_{self.batch_size}_{step_i}.png')
            plt.close()