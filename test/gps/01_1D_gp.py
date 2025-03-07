from baox.utils import generate_dataset, normalize
from baox.data_types import Dataset
from baox.surrogate.gp import SingleOuputGP
import matplotlib.pyplot as plt
from baox.surrogate.kernel import RBFKernel, MaternKernel
import jax.numpy as jnp
import jax
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"

jax.config.update("jax_enable_x64", True)


# Generate synthetic data
data = generate_dataset(jnp.cos, [[-4, 4]], random=True, num_samples=6)

# Define Test Points
x = jnp.linspace(-4, 4, 200).reshape(-1, 1)
y_true = jnp.cos(x).flatten()

kernel_configs = [
    ("blue", RBFKernel(lengthscale=jnp.array([1.0]), variance=1.0)),
    ("green", MaternKernel(lengthscale=jnp.array(
        [1.0]), variance=1.0, nu=1.5)),
    ("red", MaternKernel(lengthscale=jnp.array([1.0]), variance=1.0, nu=2.5))
]

plt.figure(figsize=(10, 5))
plt.scatter(data.x, data.y_train, color="black", label="Training Data")
plt.plot(x, y_true, color="black", linestyle="--", label="True Function")

x_test = normalize(x, jnp.array([-4]), jnp.array([4]))
for color, kernel in kernel_configs:
    gp = SingleOuputGP(data.x_train, data.y_train, kernel=kernel, noise=1e-3)
    gp.fit()
    
    mu, var = gp.predict(x_test)

    label = f"{kernel.__class__.__name__}"
    if isinstance(kernel, MaternKernel):
        label += f" (nu={kernel.nu})"
    print(gp)
    plt.plot(x, mu, label=label, color=color)
    plt.fill_between(x.flatten(), mu - 1.96 * jnp.sqrt(var),
                     mu + 1.96 * jnp.sqrt(var), alpha=0.2, color=color)

plt.legend()
plt.title("Gaussian Process Regression")
plt.grid()
plt.tight_layout()
plt.savefig("01_1D_gp.png")
