from setuptools import setup, find_packages
#conventionally read version from a file, not by importing
def read_version():
    for line in open("src/credit_risk/__init__.py"):
        if line.startswith("__version__"):
            # __version__ = "0.1.0"
            return line.split("=")[1].strip().strip('"\'')
        raise RuntimeError("cannot find version")


setup(
    name="credit-risk",
    version="0.1.0",
    author="Glenn Asher",
    description="Credit-risk modelling pipeline with fairness auditing",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas",
        "scikit-learn",
        "fairlearn",
        "shap",
        "joblib",
        "streamlit",
        "numpy",
        "aif360",
        "fastapi",
        "uvicorn",
        "pytest",
        "black",
        "flake8"
    ],
    entry_points = {
        "console_scripts":[
            "credit-risk = credit_risk.cli:cli",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8"
)