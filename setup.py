from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="aml-fraud-detection",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Anti-Money Laundering fraud detection with dynamic thresholds",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/aml-fraud-detection",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.1.0",
            "pytest-cov>=3.0.0",
            "black>=22.6.0",
            "flake8>=5.0.0",
            "isort>=5.10.0",
            "mypy>=0.971",
            "pre-commit>=2.20.0",
        ],
        "docs": [
            "sphinx>=5.1.0",
            "sphinx-rtd-theme>=1.0.0",
            "nbsphinx>=0.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "aml-train=aml_fraud_detection.cli:train",
            "aml-predict=aml_fraud_detection.cli:predict",
            "aml-api=aml_fraud_detection.api:run",
        ],
    },
    include_package_data=True,
    package_data={
        "aml_fraud_detection": ["config/*.yaml"],
    },
)
