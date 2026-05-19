from setuptools import setup, find_packages
from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension

extensions = [
    Extension(
        "pdmlabs.evaluation.anomaly_evaluator",
        ["pdmlabs/evaluation/anomaly_evaluator.pyx"],
        language="c++",
        extra_compile_args=["-std=c++11"],
    )
]

import os
import re

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

def get_version():
    init_path = os.path.join(os.path.dirname(__file__), "pdmlabs", "__init__.py")
    with open(init_path, "r", encoding="utf-8") as f:
        match = re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", f.read())
        if match:
            return match.group(1)
        raise RuntimeError("Unable to find version string.")

setup(
    name="pdmlabs",
    version=get_version(),
    author="Anastasios Papadopoulos, Apostolos Giannoulidis, DataLab AUTh",
    description="PdMLabs is an open-source Python automated machine learning benchmarking platform designed to navigate industrial time-series data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PdM-Labs/PdMLabs",
    project_urls={
        "Documentation": "https://pdm-labs.github.io/PdMLabs/",
        "Source": "https://github.com/PdM-Labs/PdMLabs",
        "Tracker": "https://github.com/PdM-Labs/PdMLabs/issues",
    },
    python_requires=">=3.11",
    packages=find_packages(include=["pdmlabs", "pdmlabs.*"]),
    ext_modules=cythonize(extensions),
    setup_requires=[
        "Cython>=0.29.0",
        "numpy>=1.24.3",
    ],
    install_requires=[
        "arch>=6.3.0",
        "auto_mix_prep>=0.2.0",
        "celery>=5.4.0",
        "hurst>=0.0.5",
        "joblib>=1.2.0",
        "locket>=1.0.0",
        "matplotlib>=3.8.4",
        "mlflow>=2.7.2",
        "mypy_extensions>=1.0.0",
        "numpy>=1.24.3",
        "pandas>=1.5",
        "patsy>=1.0.1",
        "prts>=1.0.0.3",
        "scikit_learn>=1.2.0",
        "scipy>=1.15.2",
        "six>=1.16.0",
        "statsmodels>=0.14.0",
        "tqdm>=4.66.2",
        "tsfresh>=0.21.0",
        "tslearn>=0.6.3",
        "scikit-survival>=0.25.0",
        # "torch",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
)

