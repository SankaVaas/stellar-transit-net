from setuptools import setup, find_packages

setup(
    name="stellar-transit-net",
    version="0.1.0",
    description=(
        "Exoplanet transit detection from Kepler/TESS light curves using "
        "ensemble deep learning with uncertainty quantification and active learning."
    ),
    author="SankaVaas",
    author_email="you@email.com",
    url="https://github.com/SankaVaas/stellar-transit-net",
    license="MIT",
    python_requires=">=3.10",

    # Makes `import src` work from anywhere after `pip install -e .`
    packages=find_packages(exclude=["tests*", "notebooks*", "data*", "reports*"]),

    install_requires=[
        "numpy>=1.26",
        "scipy>=1.13",
        "pandas>=2.2",
        "scikit-learn>=1.4",
        "lightkurve>=2.4",
        "astropy>=6.1",
        "astroquery>=0.4",
        "torch>=2.3",
        "shap>=0.45",
        "captum>=0.7",
        "mlflow>=2.13",
        "matplotlib>=3.9",
        "pyyaml>=6.0",
        "tqdm>=4.66",
        "crepes>=0.6",
    ],

    extras_require={
        # pip install -e ".[dev]"  — everything needed to develop & test
        "dev": [
            "pytest>=8.2",
            "pytest-cov>=5.0",
            "jupyterlab>=4.2",
            "ipywidgets>=8.1",
            "plotly>=5.22",
            "seaborn>=0.13",
        ],
    },

    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)