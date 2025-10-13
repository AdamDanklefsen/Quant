from setuptools import setup, find_packages

setup(
    name="quant-finance",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "torch",
        "yfinance",
    ],
    author="Adam Danklefsen",
    description="Quantitative Finance Package",
    python_requires=">=3.7",
)