import jax.numpy as jnp
import jax
from baox.surrogate import AutoRegressiveMFGP, MaternKernel
from baox.acquisition.qei_cost_mf import qCostAwareMultiFidelityEI
from baox.data_types import Dataset
from baox.utils import denormalize, normalize
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt_cols = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray", "cyan", "black"]

class MultiFidelityBayesianOptimization:
    """
    Multi-Fidelity Bayesian Optimization (MFBO) framework that uses an auto-regressive 
    multi-fidelity Gaussian Process (AR-MFGP) model and a cost-aware acquisition function 
    based on Expected Improvement (EI) to sequentially select candidate points for evaluation.

    The candidate points are represented as (d+1)-dimensional vectors:
        - The first d components correspond to the input (normalized in [0,1]^d).
        - The last component is a continuous fidelity indicator, which is later mapped 
          to a discrete fidelity level.

    The optimization loop proposes candidates, evaluates the corresponding objective 
    functions at different fidelities, updates the dataset, and re-fits the GP model.

    Attributes:
        objectives (dict): A dictionary mapping fidelity levels to their corresponding 
                           objective functions.
        bounds (tuple): A tuple specifying the lower and upper bounds for the input space.
        gp (AutoRegressiveMFGP): The multi-fidelity GP model used for modeling the objectives.
        acquisition (qCostAwareMultiFidelityEI): The acquisition function used to propose candidates.
        batch_size (int): Number of candidates to propose per iteration.
        n_iter (int): Total number of optimization iterations.
        dataset (Dataset): The dataset managed by the GP model.
        num_fidelities (int): Number of fidelity levels.
        step_fidelity: Stores the fidelity level chosen at the current optimization step.
    """

    def __init__(
        self,
        objectives: dict,  
        bounds: tuple,     
        gp: AutoRegressiveMFGP,
        acquisition: qCostAwareMultiFidelityEI = None,
        batch_size: int = 1,
        n_iter: int = 20
    ):
        """
        Initialize the Multi-Fidelity Bayesian Optimization framework.

        Args:
            objectives (dict): Dictionary mapping fidelity levels (int) to objective functions.
            bounds (tuple): Tuple of lower and upper bounds for the input space (e.g., ((low,), (high,))).
            gp (AutoRegressiveMFGP): The auto-regressive multi-fidelity GP model.
            acquisition (qCostAwareMultiFidelityEI, optional): The acquisition function instance. 
                If None, a default qCostAwareMultiFidelityEI is created using the provided GP.
            batch_size (int, optional): Number of candidate points to propose per iteration. Default is 1.
            n_iter (int, optional): Number of iterations for the optimization loop. Default is 20.
        """
        self.objectives = objectives
        self.bounds = bounds
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.gp = gp
        
        self.dataset = self.gp.dataset  
        self.num_fidelities = len(self.dataset.list_fidelities())
        
        # Use the cost-aware acquisition function that supports multiple fidelities.
        if acquisition is None:
            self.acquisition = qCostAwareMultiFidelityEI(gp)
        else:
            self.acquisition = acquisition(gp)
        self.step_fidelity = None

    def propose_candidates(self, key: jax.random.PRNGKey) -> jnp.ndarray:
        """
        Propose a batch of candidate points for evaluation.

        Each candidate is a (d+1)-dimensional vector where the first d dimensions are the input
        location (normalized in [0, 1]^d) and the last dimension is a continuous fidelity indicator.

        Args:
            key (jax.random.PRNGKey): PRNG key for candidate initialization and optimization.

        Returns:
            jnp.ndarray: Proposed candidate points of shape (batch_size, d+1).
        """
        return self.acquisition.propose_candidates(key, self.batch_size)

    def step(self, key: jax.random.PRNGKey):
        """
        Perform one iteration of the MFBO loop.

        The step consists of:
          - Proposing a batch of candidate points.
          - Mapping the continuous fidelity indicator to a discrete fidelity level.
          - Evaluating each candidate using the corresponding fidelity objective.
          - Updating the dataset with new observations.
          - Re-fitting the GP model with the updated dataset and reinitializing the acquisition function.

        Args:
            key (jax.random.PRNGKey): PRNG key for generating candidate points and evaluations.

        Returns:
            None
        """
        d = len(self.bounds) 
        
        X_mixed = self.propose_candidates(key)
        X_candidate = X_mixed[:, :d]
        fid_ind = X_mixed[:, -1]
        
        lower = jnp.array([b[0] for b in self.bounds])
        upper = jnp.array([b[1] for b in self.bounds])
        X_denorm = denormalize(X_candidate, lower, upper)
        
        # Map the continuous fidelity indicator to a discrete fidelity level.
        fidelity = jnp.floor(fid_ind * self.num_fidelities).astype(int)
        fidelity = jnp.clip(fidelity, 0, self.num_fidelities - 1)
        self.step_fidelity = fidelity
        for i in range(self.batch_size):
            f = int(fidelity[i])
            x_eval = X_denorm[i:i+1, :]  
            y_eval = self.objectives[f](x_eval).flatten()
            
            ds = self.dataset.get_data(f)
            updated_ds = Dataset(
                x = jnp.concatenate([ds.x, x_eval], axis=0),
                x_train = jnp.concatenate([ds.x_train, X_candidate[i:i+1, :]], axis=0),
                y_train = jnp.concatenate([ds.y_train, y_eval], axis=0),
                cost = ds.cost
            )
            self.dataset.set_data(f, updated_ds)
        
        # TODO: more elegant way to update the GP
        self.gp = AutoRegressiveMFGP(self.dataset, MaternKernel(lengthscale=jnp.array([1.0]), variance=1.0), noise=0.01)
        self.gp.fit()  
        self.acquisition = qCostAwareMultiFidelityEI(self.gp)


    def run(self, key: jax.random.PRNGKey):
        """
        Run the MFBO optimization loop for a specified number of iterations.

        At each iteration, the method:
          - Proposes candidate points.
          - Evaluates the objectives at the selected fidelity levels.
          - Updates the dataset and GP model.
          - Plots the current state of the optimization.

        Args:
            key (jax.random.PRNGKey): PRNG key for the optimization process.

        Returns:
            None
        """
        for i in range(self.n_iter):
            key, subkey = jax.random.split(key)
            print(f"Starting step {i}")
            self.step(subkey)
            for f in range(self.num_fidelities):
                num = jnp.sum(self.step_fidelity == f)
                if num != 0:
                    print(f"Fidelity {f}: {self.dataset.get_data(f).x[-num:].flatten()}")
            print(self.dataset)  
            self.plot_iteration(i)
            

    def plot_iteration(self, step_i: int):
        """
        Plot the current state of the MFBO optimization at a given iteration.

        The plot shows:
          - The objective curves for each fidelity level.
          - The predicted mean and confidence interval from the multi-fidelity GP.
          - The current sample locations.
          - An inset bar chart displaying the number of samples per fidelity level.

        The plot is saved as an image file.

        Args:
            step_i (int): The current iteration number.

        Returns:
            None
        """
        x_grid = jnp.linspace(self.bounds[0][0], self.bounds[0][1], 200).reshape(-1, 1)
        x_norm = normalize(x_grid, self.bounds[0][0], self.bounds[0][1])
        mu, var = self.gp.predict(x_norm)
        fig, ax = plt.subplots(figsize=(10, 6))

        for i in range(self.num_fidelities):
            ds = self.dataset.get_data(i)
            num_new = jnp.sum(self.step_fidelity == i)
            y = self.objectives[i](x_grid).flatten()
            ax.plot(x_grid, y, linestyle="--", label=f'Fidelity {i}', color=plt_cols[i])
            if num_new != 0:
                ax.scatter(ds.x[:-num_new], ds.y_train[:-num_new], color=plt_cols[i], s=40, facecolors='none', edgecolors=plt_cols[i])
                ax.scatter(ds.x[-num_new:], ds.y_train[-num_new:], color=plt_cols[i], marker='s', s=80)
            else:
                ax.scatter(ds.x, ds.y_train, color=plt_cols[i], s=40, facecolors='none', edgecolors=plt_cols[i])
        
        ax.plot(x_grid, mu, label='MF GP Mean', color=plt_cols[-1])
        ax.fill_between(x_grid.flatten(), mu - 1.96 * jnp.sqrt(var), mu + 1.96 * jnp.sqrt(var), color=plt_cols[-1], alpha=0.2, label='Confidence Interval')

        ax.set_xlabel('X')
        ax.set_ylabel('Objective')
        ax.set_title(f'MF-BO (Step {step_i})')
        ax.legend(ncol=1, loc='upper right')
        ax.grid(True)

        axins = inset_axes(ax, width="20%", height="25%", loc='lower right', borderpad=2)
        fidelities = list(range(self.num_fidelities))
        sample_counts = [len(self.dataset.get_data(i).x) for i in fidelities]
        axins.bar(fidelities, sample_counts, color=plt_cols[:self.num_fidelities], alpha=1)
        axins.set_ylim(0, 60)
        axins.set_title('Samples', fontsize=10)
        axins.tick_params(axis='both', which='major', labelsize=8)

        plt.tight_layout()
        plt.savefig(f'mfbo_1D_step_{step_i}.png')
        plt.close()
