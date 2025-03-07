import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from baox.surrogate import MaternKernel, SingleOuputGP
import matplotlib.pyplot as plt
from baox.test_functions import currin
from baox.utils import generate_dataset

num_train = 32
data = generate_dataset(currin, [[0, 1], [0, 1]], random=True, num_samples=num_train)

# Generate test data using meshgrid
x1_range = jnp.linspace(0., 1.0, 50)
x2_range = jnp.linspace(0., 1.0, 50)
X1, X2 = jnp.meshgrid(x1_range, x2_range)
X_test = jnp.column_stack((X1.ravel(), X2.ravel()))

# Compute true function values
y_true = currin(X_test).reshape(X1.shape)

# Use Matern Kernel
kernel = MaternKernel(lengthscale=jnp.array([1.0, 1.0]), variance=1.0)
gp = SingleOuputGP(data.x_train, data.y_train, kernel)
gp.fit()
print(gp)

# GP predictions
mu_s, var_s = gp.predict(X_test)
std_s = jnp.sqrt(var_s).reshape(X1.shape)
mu_s = mu_s.reshape(X1.shape)

# Compute absolute error
error = jnp.abs(y_true - mu_s)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot True Function
contour1 = axes[0, 0].contourf(X1, X2, y_true, cmap='coolwarm', levels=50)
plt.colorbar(contour1, ax=axes[0, 0], label="True Function Value")
axes[0, 0].scatter(data.x[:, 0], data.x[:, 1], c='black', marker='x', label="Training Data")
axes[0, 0].set_title("True Function")
axes[0, 0].set_xlabel("x1")
axes[0, 0].set_ylabel("x2")
axes[0, 0].legend()

# Plot GP Predicted Mean
contour2 = axes[0, 1].contourf(X1, X2, mu_s, cmap='coolwarm', levels=50)
plt.colorbar(contour2, ax=axes[0, 1], label="Predicted Mean")
axes[0, 1].scatter(data.x[:, 0], data.x[:, 1], c='black', marker='x', label="Training Data")
axes[0, 1].set_title("GP Predicted Mean")
axes[0, 1].set_xlabel("x1")
axes[0, 1].set_ylabel("x2")
axes[0, 1].legend()

# Plot GP Variance (Uncertainty)
contour3 = axes[1, 0].contourf(X1, X2, std_s, cmap='plasma', levels=50)
plt.colorbar(contour3, ax=axes[1, 0], label="Predicted Variance")
axes[1, 0].scatter(data.x[:, 0], data.x[:, 1], c='black', marker='x', label="Training Data")
axes[1, 0].set_title("GP Variance (Uncertainty)")
axes[1, 0].set_xlabel("x1")
axes[1, 0].set_ylabel("x2")
axes[1, 0].legend()

# Plot Absolute Error (|True - GP|)
contour4 = axes[1, 1].contourf(X1, X2, error, cmap='viridis', levels=50)
plt.colorbar(contour4, ax=axes[1, 1], label="Absolute Error")
axes[1, 1].scatter(data.x[:, 0], data.x[:, 1], c='black', marker='x', label="Training Data")
axes[1, 1].set_title("Absolute Error (|True - GP|)")
axes[1, 1].set_xlabel("x1")
axes[1, 1].set_ylabel("x2")
axes[1, 1].legend()

plt.tight_layout()
plt.savefig("02_2D_gp.png")
