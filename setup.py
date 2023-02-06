from setuptools import setup, find_packages

setup(
    name="pytorch_approximate_retrieval",
    version="0.2",
    description="Approximate KNN retrieval for PyTorch",
    url="http://github.com/itsdaniele/pytorch_approximate_retrieval",
    long_description_content_type="text/markdown",
    author="Daniele Paliotta",
    author_email="daniele.paliotta@unige.ch",
    license="MIT",
    packages=find_packages(exclude=[]),
    install_requires=[
        "numpy",
        "torch>=1.6",
    ],
)
