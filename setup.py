from setuptools import setup, find_packages

setup(
    name="baox",
    version="0.1.0",
    description="A JAX-based Bayesian Optimization framework",
    author="Yiqi Feng",
    author_email="yiqi.feng@hotmail.com",
    url="https://github.com/fengyiqi/baox",
    packages=find_packages(),
    install_requires=[
        "jax",
        "jaxlib",
        "matplotlib",
        "numpy",
        "scipy",
        "setuptools",
    ],
    python_requires=">=3.12",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)