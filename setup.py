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

setup(
    name="pdmlabs",
    version="0.1",
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
)

