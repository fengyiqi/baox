import jax.numpy as jnp
from typing import Callable, Optional


def currin(X: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the Currin function for the given input.

    The Currin function is a benchmark function used in multi-fidelity modeling.
    It uses the first two columns of X to compute the output.

    Args:
        X (jnp.ndarray): Input array of shape [n, 2] where each row is a 2D point.

    Returns:
        jnp.ndarray: An array of computed Currin function values with shape [n].
    """
    x1, x2 = X[:, 0], X[:, 1]
    return (1 - jnp.exp(-1 / (2 * x2))) * (2300 * x1**3 + 1900 * x1**2 +
            2092 * x1 + 60) / (100 * x1**3 + 500 * x1**2 + 4 * x1 + 20)


class MFLinearA:
    """
    Multi-fidelity function implementation for Model A.

    This class provides two static methods for high-fidelity and low-fidelity evaluations.
    The high-fidelity function is defined as a transformation of the input X,
    and the low-fidelity function applies a linear correction to the high-fidelity output.
    """

    @staticmethod
    def high_f(X: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the high-fidelity function for Model A.

        The function is defined as:
            y = (6.0 * X - 2.0)**2 * sin(12.0 * X - 4.0)
        and the output is negated and flattened.

        Args:
            X (jnp.ndarray): Input array.

        Returns:
            jnp.ndarray: Flattened output array of the high-fidelity function.
        """
        y = (6.0 * X - 2.0)**2 * jnp.sin(12.0 * X - 4.0)
        return y.flatten()

    @staticmethod
    def low_f(X: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the low-fidelity function for Model A.

        The low-fidelity function applies a linear correction to the high-fidelity function:
            y_low = 0.5 * y_high + 10.0 * (X - 0.5) + 5.0
        and the output is negated and flattened.

        Args:
            X (jnp.ndarray): Input array.

        Returns:
            jnp.ndarray: Flattened output array of the low-fidelity function.
        """
        y = (6.0 * X - 2.0)**2 * jnp.sin(12.0 * X - 4.0)
        y = 0.5 * y + 10.0 * (X - 0.5) + 5.0
        return y.flatten()


class MFLinearB:
    """
    Multi-fidelity function implementation for Model B.

    This class provides high-fidelity and low-fidelity evaluations. The high-fidelity function
    is a quadratic-sinusoidal transformation of the input, and the low-fidelity function is a combination
    of a scaled version of the high-fidelity output and additional nonlinear corrections.
    """

    @staticmethod
    def high_f(X: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the high-fidelity function for Model B.

        The function is defined as:
            y = 5.0 * X**2 * sin(12.0 * X)
        and the output is flattened.

        Args:
            X (jnp.ndarray): Input array.

        Returns:
            jnp.ndarray: Flattened output array of the high-fidelity function.
        """
        y = 5.0 * X**2 * jnp.sin(12.0 * X)
        return y.flatten()

    @staticmethod
    def low_f(X: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the low-fidelity function for Model B.

        The low-fidelity function is defined as:
            y_low = 2.0 * high_f(X) + (X**3 - 0.5) * sin(3.0 * X - 0.5) + 4.0 * cos(2.0 * X)
        and the output is flattened.

        Args:
            X (jnp.ndarray): Input array.

        Returns:
            jnp.ndarray: Flattened output array of the low-fidelity function.
        """
        y = 2.0 * MFLinearB.high_f(X).reshape(X.shape) + (X**3 - 0.5) * \
            jnp.sin(3.0 * X - 0.5) + 4.0 * jnp.cos(2.0 * X)
        return y.flatten()


class MFBranin:
    """
    Multi-fidelity implementation for the Branin function.

    Provides high-fidelity and low-fidelity evaluations of the Branin function, which is a common
    benchmark in optimization.
    """

    @staticmethod
    def high_f(X: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the high-fidelity Branin function.

        The Branin function is defined as:
            f(x1, x2) = (x2 - (5.1 / (4 * pi^2)) * x1^2 + (5 / pi) * x1 - 6)^2 +
                        10 * (1 - 1 / (8 * pi)) * cos(x1) + 10

        Args:
            X (jnp.ndarray): Input array of shape [n, 2] where each row contains (x1, x2).

        Returns:
            jnp.ndarray: An array of function values with shape [n].
        """
        x1, x2 = X[:, 0], X[:, 1]
        return (x2 - (5.1 / (4 * jnp.pi**2)) * x1**2 + (5 / jnp.pi) * x1 - 6) ** 2 + \
               10 * (1 - 1 / (8 * jnp.pi)) * jnp.cos(x1) + 10

    @staticmethod
    def low_f(X: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the low-fidelity Branin function.

        The low-fidelity function is defined as:
            f_low(x1, x2) = 0.5 * high_f(x1, x2) + 2*(x1 - 0.5) + 2*(x2 - 0.5)

        Args:
            X (jnp.ndarray): Input array of shape [n, 2].

        Returns:
            jnp.ndarray: An array of low-fidelity function values with shape [n].
        """
        x1, x2 = X[:, 0], X[:, 1]
        return 0.5 * MFBranin.high_f(X) + 2 * (x1 - 0.5) + 2 * (x2 - 0.5)


class MFCurrin:
    """
    Multi-fidelity implementation for the Currin function.

    Provides high-fidelity and low-fidelity evaluations based on the Currin function,
    which is used in multi-fidelity modeling benchmarks.
    """

    @staticmethod
    def high_f(X: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the high-fidelity Currin function.

        This simply calls the `currin` function with the provided inputs.

        Args:
            X (jnp.ndarray): Input array of shape [n, 2].

        Returns:
            jnp.ndarray: An array of high-fidelity Currin function values with shape [n].
        """
        return currin(X)

    @staticmethod
    def low_f(X: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the low-fidelity Currin function.

        The low-fidelity function is computed as the average of Currin function values
        evaluated at four perturbed versions of the input X.

        Args:
            X (jnp.ndarray): Input array of shape [n, 2].

        Returns:
            jnp.ndarray: An array of low-fidelity Currin function values with shape [n].
        """
        x_low_1 = X + jnp.array([0.05, 0.05])
        x_low_2 = X + jnp.array([0.05, -0.05])
        x_low_2 = x_low_2.at[:, 1:].set(jnp.maximum(x_low_2[:, 1:], 0.0))
        x_low_3 = X + jnp.array([-0.05, 0.05])
        x_low_4 = X + jnp.array([-0.05, -0.05])
        x_low_4 = x_low_4.at[:, 1:].set(jnp.maximum(x_low_4[:, 1:], 0.0))
        return 0.25 * (currin(x_low_1) + currin(x_low_2) + currin(x_low_3) + currin(x_low_4))


class MFHimmelblau:
    """
    Multi-fidelity implementation for the Himmelblau function.

    Provides high-fidelity and low-fidelity evaluations of the Himmelblau function,
    which is a popular test function for optimization.
    """

    @staticmethod
    def high_f(X: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the high-fidelity Himmelblau function.

        The Himmelblau function is defined as:
            f(x1, x2) = -( (x1^2 + x2 - 11)^2 + (x1 + x2^2 - 7)^2 )

        Args:
            X (jnp.ndarray): Input array of shape [n, 2] where each row contains (x1, x2).

        Returns:
            jnp.ndarray: An array of high-fidelity Himmelblau function values with shape [n].
        """
        x1, x2 = X[:, 0], X[:, 1]
        return -(x1**2 + x2 - 11)**2 - (x1 + x2**2 - 7)**2

    @staticmethod
    def low_f(X: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the low-fidelity Himmelblau function.

        The low-fidelity function is defined as a scaled version of the high-fidelity
        function with additional nonlinear corrections:
            f_low(x1, x2) = 0.5 * high_f( scaled X ) - x2^3 + (x1 + 1)^2,
        where X is scaled element-wise by [0.5, 0.8] before computing high_f.

        Args:
            X (jnp.ndarray): Input array of shape [n, 2].

        Returns:
            jnp.ndarray: An array of low-fidelity Himmelblau function values with shape [n].
        """
        x1, x2 = X[:, 0], X[:, 1]
        y = MFHimmelblau.high_f(X * jnp.array([0.5, 0.8]))
        return 0.5 * y - x2**3 + (x1 + 1)**2
    

class MFForrester:
    @staticmethod
    def f_3(X: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the high-fidelity Forrester function.

        The Forrester function is defined as:
            f(x) = (6x - 2)^2 * sin(12x - 4)

        Args:
            X (jnp.ndarray): Input array of shape [n, 1].

        Returns:
            jnp.ndarray: An array of high-fidelity Forrester function values with shape [n].
        """
        return (6 * X - 2)**2 * jnp.sin(12 * X - 4)

    @staticmethod
    def f_2(X: jnp.ndarray) -> jnp.ndarray:
        """
        Compute a medium-fidelity Forrester function.

        This variant of the Forrester function uses slightly adjusted parameters:
            f(x) = (5.5x - 2.5)^2 * sin(12x - 4)

        Args:
            X (jnp.ndarray): Input array of shape [n, 1].

        Returns:
            jnp.ndarray: An array of medium-fidelity Forrester function values with shape [n].
        """
        return (5.5 * X - 2.5)**2 * jnp.sin(12.0 * X - 4.0)

    @staticmethod
    def f_1(X: jnp.ndarray) -> jnp.ndarray:
        """
        Compute a low-fidelity Forrester function (variant 1).

        This low-fidelity function is defined as a scaled and shifted version of the high-fidelity function:
            f(x) = 0.75 * f_3(x) + 5(x - 0.5) - 2

        Args:
            X (jnp.ndarray): Input array of shape [n, 1].

        Returns:
            jnp.ndarray: An array of low-fidelity Forrester function values with shape [n].
        """
        return 0.75 * MFForrester.f_3(X) + 5.0 * (X - 0.5) - 2.0

    @staticmethod
    def f_0(X: jnp.ndarray) -> jnp.ndarray:
        """
        Compute a low-fidelity Forrester function (variant 2).

        This variant is defined as:
            f(x) = 0.5 * f_3(x) + 10(x - 0.5) - 5

        Args:
            X (jnp.ndarray): Input array of shape [n, 1].

        Returns:
            jnp.ndarray: An array of low-fidelity Forrester function values with shape [n].
        """
        return 0.5 * MFForrester.f_3(X) + 10.0 * (X - 0.5) - 5.0

        
        
